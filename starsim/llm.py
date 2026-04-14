"""
LLM-powered modules for integrating large language model decisions into simulations.

Uses the OpenRouter API to make LLM calls during the simulation loop.
"""
import concurrent.futures
import requests
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


__all__ = ['LLMIntervention']

def _call_openrouter(prompt, model, api_key, max_tokens, timeout, seed=None, session=None):
    """
    Send a prompt to OpenRouter and return the response text.

    Pass ``seed`` (int) to enable deterministic outputs across runs (requires
    ``temperature=0``; not all models honour this field).
    Pass ``session`` (requests.Session) to reuse connections across calls.
    """
    _OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
    payload = {
        'model':       model,
        'messages':    [{'role': 'user', 'content': prompt}],
        'temperature': 0,
        'provider':    {'order': ['Groq'], 'allow_fallbacks': True},
    }
    if max_tokens is not None:
        payload['max_tokens'] = max_tokens
    if seed is not None:
        payload['seed'] = seed

    caller  = session if session is not None else requests
    resp    = caller.post(
        _OPENROUTER_URL,
        json    = payload,
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type':  'application/json',
        },
        timeout = timeout,
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


class LLMIntervention(ss.Intervention):
    """
    LLM-driven quarantine intervention using the Health Belief Model (HBM).

    Once per decision round, one LLM call is made per active agent. Each call
    asks whether that agent should quarantine (yes = quarantine). Quarantined
    agents have their ``rel_sus`` and ``rel_trans`` zeroed on the target disease
    before ``disease.step()`` runs, effectively removing them from the contact
    network for that timestep. These are restored at the start of the next
    decision round.

    With daily (or coarser) timesteps, decisions are made every ``interval``
    steps. With sub-daily timesteps, decisions are made once per calendar day
    on the first timestep at or after ``decision_hour``.

    Each agent carries an HBM profile sampled at initialisation: perceived
    infection risk, health severity, quarantine self-efficacy, and response
    efficacy (Likert 1–6).

    Args:
        low_reward (float): Points awarded per decision round for quarantining. Default 5.
        high_reward (float): Points awarded per decision round for not quarantining (if uninfected). Default 10.
        model (str): OpenRouter model identifier. Defaults to a free model.
        api_key (str): OpenRouter API key (or set ``OPENROUTER_API_KEY`` env var).
        interval (int): Timesteps between decision rounds (daily or coarser dt only). Default 1.
        decision_hour (float): Hour of day to make decisions for sub-daily sims (9.5 = 09:30). Default 9.5.
        build_prompt (callable): Optional ``fn(mod, uid, disease) -> str`` that
            builds the per-agent prompt. Defaults to ``ss.default_agent_prompt``.
        init_beliefs (callable): Optional ``fn(mod) -> None`` that populates per-agent
            HBM belief states in-place. Must be provided; no built-in default exists.
        max_tokens (int): Max LLM response tokens. Default None (model default).
        timeout (int): HTTP request timeout in seconds. Default 20.
        verbose (bool): Print LLM errors and summary statistics. Default False.
        max_workers (int): Max parallel LLM calls per batch. Default 12.
        rate_limit (int): Max requests per minute (None = unlimited). Default 100.
        agent_uids (array-like): UIDs of agents to include; None means all agents. Default None.
    """

    def __init__(self, low_reward=5, high_reward=10,
                 model=None, api_key=None, interval=1, decision_hour=9.5,
                 build_prompt=None, init_beliefs=None,
                 max_tokens=None, timeout=20, verbose=False,
                 max_workers=12, rate_limit=100, agent_uids=None, **kwargs):
        super().__init__(**kwargs)

        self.low_reward       = low_reward
        self.high_reward      = high_reward
        self.model            = model
        self.api_key          = api_key
        self.interval         = interval
        self.decision_hour    = decision_hour   # Hour of day to make decisions (9.5 = 09:30)
        self.build_prompt_fn  = build_prompt if build_prompt is not None else ss.default_agent_prompt
        self.init_beliefs_fn  = init_beliefs
        self.max_tokens       = max_tokens
        self.timeout          = timeout
        self.verbose          = verbose
        self.max_workers      = max_workers
        self.rate_limit       = rate_limit  # Max requests/min (None = unlimited); free models: 20/min
        self.agent_uids       = np.asarray(agent_uids, dtype=int) if agent_uids is not None else None
        self._session = requests.Session()  # Shared across all LLMIntervention instances
        self._last_decision_date = None  # Track last day decisions were made (sub-daily sims)

        # Per-agent states
        self.define_states(
            ss.BoolState('quarantined',   label='Quarantined'),
            ss.FloatArr('perceived_infection_risk', default=3.0, label='HBM perceived susceptibility (1-6)'),
            ss.FloatArr('perceived_health_severity',       default=3.0, label='HBM perceived severity (1-6)'),
            ss.FloatArr('quarantine_self_efficacy',  default=3.0, label='HBM perceived self-efficacy (1-6)'),
            ss.FloatArr('quarantine_response_efficacy',       default=3.0, label='HBM perceived response-efficacy (1-6)'),
            ss.FloatArr('points',         default=0.0, label='Accumulated game points'),
            ss.FloatArr('n_quarantine_steps', default=0.0, label='Number of steps agent quarantined'),
            ss.FloatArr('reward_high',    default=float(high_reward), label='Per-agent high reward (staying active)'),
            ss.BoolState('has_been_infected', default=False, label='Agent has been infected?'),
        )

        self.log              = []   # Per-step sc.objdict: {t, ti, n_agents, n_quarantined, error}
        self.decision_log     = []   # Per-agent per-day: {date, uid, quarantined}
        self.agent_summary    = None  # Populated by finalize()
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('quarantine_rate', dtype=float, scale=False, label='Quarantine rate'),
            ss.Result('mean_points',     dtype=float, scale=False, label='Mean points'),
        )
        return

    def init_post(self):
        """ Initialize per-agent HBM belief scores via ``init_beliefs_fn`` """
        super().init_post()
        self.init_beliefs_fn(self)
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _target_disease(self):
        """ Return the first disease module in the sim, or None """
        vals = list(self.sim.diseases.values())
        return vals[0] if vals else None

    def _local_prevalence(self, uid, disease):
        """
        Fraction of uid's direct network contacts that were infected at the
        start of this step (i.e. before disease.step() runs).

        Since intv.step() executes before disease.step(), disease.infected
        reflects the previous step's state — matching the AUIB game design
        where agents see local prevalence from the prior timestep.
        """
        if disease is None or not hasattr(disease, 'infected'):
            return "0/0"
        contacts = set()
        for net in self.sim.networks.values():
            contacts.update(net.find_contacts(ss.uids([uid])))
        contacts.discard(int(uid))  # exclude self if present
        if not contacts:
            return "0/0"
        contact_uids = ss.uids(list(contacts))
        return f"{disease.infected[contact_uids].sum()}/{len(contact_uids)}"

    def _agent_status(self, uid, disease):
        if disease is None:
            return 'unknown'
        if not self.sim.people.alive[uid]:
            return 'dead'
        if hasattr(disease, 'symptom_cat') and disease.symptom_cat[uid]:
            # asymptomatic
            if 0 == disease.symptom_cat[uid]:
                return 'healthy'
            # ONLY KNOW IF THEY ARE INFECTED
            # "mild"
            if 1 == disease.symptom_cat[uid]:
                if hasattr(disease, 'infected') and disease.infected[uid]:
                    self.has_been_infected[uid] = True
                    return 'mild symptom'
            # "severe"
            if 2 == disease.symptom_cat[uid]:
                if hasattr(disease, 'infected') and disease.infected[uid]:
                    self.has_been_infected[uid] = True
                    return 'severe symptom'
        return 'healthy'

    def _call_llm_agent(self, uid, disease):
        """ Ask the LLM whether this agent should quarantine. Returns (bool, response_text). """
        fn     = self.build_prompt_fn
        prompt = fn(self, uid, disease)

        seed    = int(self.sim.pars.rand_seed)
        content = _call_openrouter(prompt, self.model, self.api_key, self.max_tokens, self.timeout, seed=seed, session=self._session)
        content = (content or '').strip().lower()

        return 'yes' in content, content

    def _zero_transmission(self, q_uids, disease):
        """ Set rel_sus and rel_trans to 0 for quarantined agents """
        if disease is None:
            return
        if hasattr(disease, 'rel_sus'):
            disease.rel_sus[q_uids]   = 0.0
        if hasattr(disease, 'rel_trans'):
            disease.rel_trans[q_uids] = 0.0
        return

    def _restore_transmission(self, q_uids, disease):
        """ Restore rel_sus and rel_trans to 1 for previously quarantined agents """
        if disease is None or len(q_uids) == 0:
            return
        if hasattr(disease, 'rel_sus'):
            disease.rel_sus[q_uids]   = 1.0
        if hasattr(disease, 'rel_trans'):
            disease.rel_trans[q_uids] = 1.0
        return

    def _is_decision_time(self):
        """
        Return True if a decision round should run this timestep.

        With daily (or coarser) resolution the ``interval`` parameter governs
        frequency exactly as before.  With sub-daily resolution the intervention
        fires exactly once per calendar day, on the first timestep whose
        wall-clock hour is >= ``self.decision_hour`` (default 9.5 = 09:30).
        """
        dt_days = float(self.sim.t.dt)
        if dt_days >= 1.0:
            # Daily or coarser: use interval as usual
            return self.ti % self.interval == 0

        # Sub-daily: fire once per day at decision_hour
        now  = pd.Timestamp(self.now)
        hour = now.hour + now.minute / 60.0 + now.second / 3600.0
        if hour < self.decision_hour:
            return False
        today = now.date()
        if self._last_decision_date == today:
            return False
        self._last_decision_date = today
        return True

    # ------------------------------------------------------------------
    # Module lifecycle
    # ------------------------------------------------------------------

    def step(self):
        """
        For each active agent:
        1. Ask the LLM whether they quarantine (once per day at decision_hour).
        2. Zero transmission for quarantined agents.
        3. Award points.
        """
        import time as _time
        _t0 = _time.perf_counter()
        def _elapsed(): return f'{_time.perf_counter() - _t0:.3f}s'

        if not self._is_decision_time():
            return

        disease = self._target_disease()
        q_uids = ss.uids(np.intersect1d(self.quarantined.uids, self.sim.people.auids))
        self._restore_transmission(q_uids, disease)
        self.quarantined[:] = False

        all_uids = self.sim.people.auids
        if self.agent_uids is not None:
            uids = ss.uids(np.intersect1d(all_uids, self.agent_uids))
        else:
            uids = all_uids
        disease = self._target_disease()
        entry   = sc.objdict(t=self.ti, ti=self.ti,
                             n_agents=len(uids), n_quarantined=0, error=None)

        if len(uids) == 0:
            self.log.append(entry)
            return

        # LLM calls in parallel batches; each batch waits up to 60s, then rate-limits before the next
        BATCH_TIMEOUT  = 60  # seconds to wait per batch before marking non-responders as no-quarantine
        min_batch_secs = (self.max_workers / self.rate_limit * 60) if self.rate_limit else 0

        decisions  = {}
        errors     = []
        uid_list   = list(uids)
        batch_size = self.max_workers
        n_batches  = (len(uid_list) + batch_size - 1) // batch_size

        for b, start in enumerate(range(0, len(uid_list), batch_size)):
            chunk      = uid_list[start:start + batch_size]
            batch_t0   = _time.perf_counter()
            print(f'\n[{self.name} t={self.ti}] Batch {b+1}/{n_batches} — {len(chunk)} agents ({_elapsed()})', flush=True)

            pool    = concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk))
            futures = {pool.submit(self._call_llm_agent, uid, disease): int(uid) for uid in chunk}
            try:
                for fut in concurrent.futures.as_completed(futures, timeout=BATCH_TIMEOUT):
                    uid_int = futures[fut]
                    try:
                        result, content = fut.result()
                        decisions[uid_int] = result
                        print(f'  Agent {uid_int}: {content} ({_elapsed()})', flush=True)
                    except Exception as e:
                        decisions[uid_int] = False
                        errors.append(f'agent {uid_int}: {e}')
                        print(f'  Agent {uid_int}: ERROR {e} ({_elapsed()})', flush=True)
            except concurrent.futures.TimeoutError:
                for fut, uid_int in futures.items():
                    if uid_int not in decisions:
                        fut.cancel()
                        decisions[uid_int] = False
                        errors.append(f'agent {uid_int}: batch timeout')
                        print(f'  Agent {uid_int}: TIMEOUT ({_elapsed()})', flush=True)
            finally:
                pool.shutdown(wait=False)  # don't block on any still-running threads

            # Rate-limit: ensure we don't exceed rate_limit req/min across batches
            if min_batch_secs > 0 and b < n_batches - 1:
                elapsed_batch = _time.perf_counter() - batch_t0
                wait = min_batch_secs - elapsed_batch
                if wait > 0:
                    _time.sleep(wait)

        if errors:
            entry.error = '; '.join(errors)
            if self.verbose:
                print(f'[LLMIntervention t={self.ti}] LLM errors: {entry.error}')

        q_list      = ss.uids([uid for uid, q in decisions.items() if     q])
        active_list = ss.uids([uid for uid, q in decisions.items() if not q])

        # Record per-agent decision for this day
        current_date = str(self.now)
        for uid, did_quarantine in decisions.items():
            self.decision_log.append(
                dict(
                    date=current_date,
                    uid=uid,
                    quarantined=did_quarantine,
                    status=self._agent_status(uid, disease),
                    points=self.points[uid]
                )
            )

        # Add a row for each dead tracked agent
        if self.agent_uids is not None:
            dead_tracked = np.setdiff1d(self.agent_uids, np.array(self.sim.people.auids))
            for uid in dead_tracked:
                self.decision_log.append(
                    dict(date=current_date, uid=int(uid), quarantined=False, status='dead', points=0.0)
                )

        if len(q_list):
            self.quarantined[q_list] = True
            self._zero_transmission(q_list, disease)
            self.points[q_list]             += self.low_reward
            self.n_quarantine_steps[q_list] += 1

        if len(active_list):
            if disease is not None and hasattr(disease, 'infected'):
                not_infected = active_list[~disease.infected[active_list]]
            else:
                not_infected = active_list
            self.points[not_infected] += self.reward_high[not_infected]

        entry.n_quarantined = int(len(q_list))
        self.log.append(entry)
        return

    def finish_step(self):
        """ Zero points for agents who died this timestep """
        dead_uids = self.sim.people.dead.uids
        if len(dead_uids):
            self.points[dead_uids] = 0.0
        super().finish_step()
        return

    def update_results(self):
        super().update_results()
        uids = self.sim.people.auids
        n    = max(len(uids), 1)
        n_q  = int(self.quarantined.sum())
        pts  = float(self.points[uids].mean()) if len(uids) else 0.0
        self.results['quarantine_rate'][self.ti] = n_q / n
        self.results['mean_points'][self.ti]     = pts
        return

    def finalize(self):
        super().finalize()
        n_calls  = len(self.log)
        n_errors = sum(1 for e in self.log if e.error)
        if self.verbose or n_errors:
            total_q = sum(e.n_quarantined for e in self.log)
            total_a = sum(e.n_agents      for e in self.log)
            rate    = total_q / max(total_a, 1)
            print(f'[LLMIntervention] {n_calls} steps | overall quarantine rate: {rate:.1%} | {n_errors} errors')

        self._build_agent_summary()
        self.decision_log = pd.DataFrame(self.decision_log)  # date | uid | quarantined
        return

    def _build_agent_summary(self):
        """
        Build a per-agent summary DataFrame stored as ``self.agent_summary``.

        Columns: uid, status, points, n_quarantine_steps, quarantine_rate,
                 susceptibility, severity, self_efficacy, benefits.
        """
        disease    = self._target_disease()
        n_decisions = max(len(self.log), 1)
        rows = []
        for uid in self.sim.people.auids:
            uid_int = int(uid)
            rows.append(dict(
                uid                = uid_int,
                status             = self._agent_status(uid, disease),
                points             = float(self.points[uid]),
                n_quarantine_steps = int(self.n_quarantine_steps[uid]),
                quarantine_rate    = float(self.n_quarantine_steps[uid]) / n_decisions,
                susceptibility     = float(self.perceived_infection_risk[uid]),
                severity           = float(self.perceived_health_severity[uid]),
                self_efficacy      = float(self.quarantine_self_efficacy[uid]),
                benefits           = float(self.quarantine_response_efficacy[uid]),
            ))
        self.agent_summary = pd.DataFrame(rows).set_index('uid')
        return

"""
Epigame LLM intervention study.

This study simulates an epidemic game in which autonomous LLM-driven agents
decide each day whether to quarantine or remain active. 

Agents interact via real-world proximity contact networks derived 
from the Epigames dataset and are infected according to a SEIR disease model. 

The LLM intervention uses the Health Belief Model (HBM) to inform each agent's decision, 
weighting perceived susceptibility, severity, self-efficacy, and benefits. 

Research Questions: 

1. How does LLM-guided quarantine behavior, conditioned on 
individual health status and reward incentives, alter epidemic curve trajectories 
and cumulative scoring outcomes across distinct payoff structures?

2. Results are benchmarked against epigames how are these scores different to what 
was done in practice?

Usage:

OPENROUTER_API_KEY=... uv run python tests/test_epigame.py

GET GRAPHS:
python starsim/plot_results.py run_outputs/20260324T170428Z/results_sim.csv
"""
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import starsim as ss

def initial_infection(csv_path: str, user_id_map: dict):
    df_initial = pd.read_csv(csv_path, index_col=0)
    df_initial = df_initial[df_initial["inf"] == "CASE0[0]"].copy()
    df_initial["uid"] = df_initial["user_id"].map(user_id_map)
    df_initial = df_initial.dropna(subset=["uid"]).copy()
    df_initial["uid"] = df_initial["uid"].astype(int)

    initial_infection_uids = df_initial["uid"].tolist()
    return initial_infection_uids

def main():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError('Set OPENROUTER_API_KEY env var before running')

    run_dir = Path("run_outputs") / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    MODEL = 'openai/gpt-oss-120b'
    net, n_agents, start_date, stop_date, id_map = ss.build_network("data_ingestion/histories.csv")
    group_a_uids, group_b_uids = ss.group_split("data_ingestion/participants.csv", id_map)
    all_participant_uids = group_a_uids + group_b_uids
    initial_infected_ids = initial_infection("data_ingestion/histories.csv", id_map)

    seir = ss.SEIR_AMS(
        init_prev      = ss.bernoulli(p=0.01),
        beta           = ss.perday(0.0907*24),
        dur_exp        = ss.expon(scale=ss.days(10/24)),
        dur_inf        = ss.expon(scale=ss.days(77/24)),
        p_symp         = ss.choice(a=3, p=[0.30, 0.42, 0.28]),
        p_death_mild   = ss.bernoulli(p=0.25),
        p_death_severe = ss.bernoulli(p=0.70),
        
    )

    #Random Net
    network = ss.RandomNet(n_contacts=ss.lognorm_ex(mean=2.4, std=1.55), dur=ss.days(1/(24*60*6)))

    sim = ss.Sim(
        n_agents      = n_agents,
        start         = start_date,
        stop          = stop_date,
        dt            = ss.days(1/8640),
        rand_seed     = 42,
        diseases      = seir,
        networks      = network,
        interventions = ss.make_intervention(
            high_reward    = 10,
            agent_uids     = all_participant_uids,
            model          = MODEL,
            api_key        = api_key,
            name           = 'epigame',
            id_map         = id_map,
            answers_path   = "data_ingestion/survey-answers.csv",
            group_b_uids   = group_b_uids,
            group_b_reward = 15,
        ),
    )
    # --- Seed initial infections from data ---

    # Get the disease module
    sim.init()  # make sure the internal module objects are registered

    disease = sim.get_module(ss.SEIR_AMS)   # or ss.Infection
    disease.set_prognoses(ss.uids(initial_infected_ids))

    sim.run()

    # Save a manifest of the run configuration.
    run_metadata = {
        "model":         MODEL,
        "rand_seed":     42,
        "n_agents":      n_agents,
        "start_date":    str(start_date),
        "stop_date":     str(stop_date),
        "group_a_uids":  group_a_uids,
        "group_b_uids":  group_b_uids,
        "id_map":        {str(k): int(v) for k, v in id_map.items()},
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, default=str)

    # Save the full simulation object if possible.
    try:
        with open(run_dir / "sim.pkl", "wb") as f:
            pickle.dump(sim, f)
    except Exception as e:
        with open(run_dir / "sim_pickle_error.txt", "w", encoding="utf-8") as f:
            f.write(repr(e))

    # Save sim.results — one CSV per module.
    results = getattr(sim, "results", None)
    if results is not None:
        try:
            dfs = results.to_df(descend=True)
            # to_df(descend=True) returns either a single DataFrame or an sc.objdict
            # keyed by module name (e.g. 'sim', 'seir', 'demographics', …).
            if isinstance(dfs, pd.DataFrame):
                dfs.to_csv(run_dir / "results_sim.csv")
            else:
                for key, df in dfs.items():
                    safe_key = str(key).replace("/", "_")
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(run_dir / f"results_{safe_key}.csv")
                    elif isinstance(df, pd.Series):
                        df.to_csv(run_dir / f"results_{safe_key}.csv")
        except Exception as e:
            with open(run_dir / "sim_results_error.txt", "w", encoding="utf-8") as f:
                f.write(repr(e))

    # Save everything for every intervention that exists
    for label, mod in sim.interventions.items():
        # Save intervention-specific logs and summaries
        try:
            step_log_df = pd.DataFrame(mod.log)
            step_log_df_a = step_log_df[step_log_df.uid.isin(group_a_uids)]            
            step_log_df_b = step_log_df[step_log_df.uid.isin(group_b_uids)]  
            step_log_df_a.to_csv(run_dir / f"{label}_step_log_a.csv", index=False)
            step_log_df_b.to_csv(run_dir / f"{label}_step_log_b.csv", index=False)

        except Exception:
            pass

        try:
            decision_log_df = mod.decision_log
            if isinstance(decision_log_df, list):
                decision_log_df = pd.DataFrame(decision_log_df)

            decision_log_df_a = decision_log_df[decision_log_df.uid.isin(group_a_uids)]            
            decision_log_df_b = decision_log_df[decision_log_df.uid.isin(group_b_uids)]  
            decision_log_df_a.to_csv(run_dir / f"{label}_decision_log_a.csv", index=False)
            decision_log_df_b.to_csv(run_dir / f"{label}_decision_log_b.csv", index=False)
            
        except Exception:
            pass

        try:
            if isinstance(mod.agent_summary, pd.DataFrame):
                mod.agent_summary.to_csv(run_dir / f"{label}_agent_summary.csv", index=False)
            else:
                pd.DataFrame(mod.agent_summary).to_csv(run_dir / f"{label}_agent_summary.csv", index=False)
        except Exception:
            pass

        try:
            quarantine_rate = sim.results[label].quarantine_rate
            if hasattr(quarantine_rate, "to_csv"):
                quarantine_rate.to_csv(run_dir / f"{label}_quarantine_rate.csv")
            else:
                pd.DataFrame(quarantine_rate).to_csv(run_dir / f"{label}_quarantine_rate.csv", index=False)
        except Exception:
            pass

        print(f'\n{"=" * 50}')
        print(f'Intervention: {label}  (high={getattr(mod, "high_reward", None)}, n_agents={len(getattr(mod, "agent_uids", []))})')
        print('=' * 50)

        print('\n--- Step log ---')
        for e in mod.log:
            status = f"{e.n_quarantined}/{e.n_agents} quarantined"
            err = f"  ERROR: {e.error}" if getattr(e, "error", None) else ""
            print(f"  t={e.t}: {status}{err}")

        print('\n--- Per-agent quarantine decisions ---')
        dl = mod.decision_log if not isinstance(mod.decision_log, list) else pd.DataFrame(mod.decision_log)
        if len(dl):
            print(dl.to_string(index=False))

        print('\n--- Per-agent summary ---')
        print(mod.agent_summary)

        print('\n--- Quarantine rate over time ---')
        print(sim.results[label].quarantine_rate)
    
    fig = sim.plot()
    fig.savefig(run_dir / "sim_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved run artifacts to: {run_dir.resolve()}")

if __name__ == '__main__':
    main()
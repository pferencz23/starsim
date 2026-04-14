"""
Quick check that OpenRouter API calls succeed (SSL, auth, model).

Run with:
    OPENROUTER_API_KEY=<your-key> uv run python tests/test_api.py
"""
import json
import os
import ssl
import urllib.request
import certifi
import starsim as ss


_URL   = 'https://openrouter.ai/api/v1/chat/completions'
_MODEL = 'nvidia/nemotron-3-super-120b-a12b:free'

def check_api(api_key=None):
    key = api_key or os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('Set OPENROUTER_API_KEY env var or pass api_key=')

    payload = json.dumps({
        'model':      _MODEL,
        'messages':   [{'role': 'user', 'content': 'Tell me a joke.'}],
    }).encode('utf-8')

    req = urllib.request.Request(
        _URL,
        data    = payload,
        headers = {
            'Authorization': f'Bearer {key}',
            'Content-Type':  'application/json',
        },
        method = 'POST',
    )

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    print(f'Sending request to {_URL} ...')
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))

    reply = data['choices'][0]['message']['content']
    print(f'Response: {reply!r}')
    print('API call succeeded.')
    return reply

def toy_example():
    """
    Minimal 12-agent test that doesn't need any CSV data files.

    One LLMIntervention covers all agents. Group membership is encoded as a
    per-agent reward: group_a earns 10 pts for staying active, group_b earns 15.
    Uses daily dt and runs for 5 days (5 decision rounds, 12 LLM calls each).
    """
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError('Set OPENROUTER_API_KEY env var before running')

    MODEL    = 'openai/gpt-oss-20b'
    N        = 12
    all_uids = list(range(N))
    group_a  = all_uids[:6]   # high_reward = 10
    group_b  = all_uids[6:]   # high_reward = 15

    seir = ss.SEIR_AMS(
        init_prev      = ss.bernoulli(p=0.2),
        beta           = ss.perday(0.5),
        dur_exp        = ss.lognorm_ex(mean=ss.days(1), std=ss.days(0.1)),
        dur_inf        = ss.lognorm_ex(mean=ss.days(2), std=ss.days(0.2)),
        p_symp         = ss.choice(a=3, p=[0.50, 0.35, 0.15]),
        p_death_mild   = ss.bernoulli(p=0.05),
        p_death_severe = ss.bernoulli(p=0.20),
    )

    def init_beliefs(mod):
        """Set group_b agents' high reward; group_a keeps the default of 10."""
        for uid in group_b:
            mod.reward_high[uid] = 15.0

    sim = ss.Sim(
        n_agents      = N,
        start         = '2024-01-01',
        stop          = '2024-01-05',
        dt            = ss.days(1),
        rand_seed     = 42,
        diseases      = seir,
        networks      = ss.RandomNet(n_contacts=3),
        interventions = ss.LLMIntervention(
            low_reward   = 5,
            high_reward  = 10,
            agent_uids   = all_uids,
            model        = MODEL,
            api_key      = api_key,
            interval     = 1,
            init_beliefs = init_beliefs,
            verbose      = True,
            name         = 'epigame',
            max_workers  = 12,
            rate_limit   = 100,
        ),
    )
    sim.run()

    mod = sim.interventions['epigame']

    print(f'\n{"=" * 50}')
    print('Step log')
    print('=' * 50)
    for e in mod.log:
        err = f'  ERROR: {e.error}' if e.error else ''
        print(f'  t={e.t}: {e.n_quarantined}/{e.n_agents} quarantined{err}')

    print(f'\n{"=" * 50}')
    print('Per-agent summary')
    print('=' * 50)
    print(mod.agent_summary.to_string())

    print(f'\n{"=" * 50}')
    print('Group breakdown')
    print('=' * 50)
    summary = mod.agent_summary
    for label, uids in [('group_a (reward=10)', group_a), ('group_b (reward=15)', group_b)]:
        grp = summary.loc[summary.index.isin(uids)]
        print(f'  {label}: quarantine_rate={grp.quarantine_rate.mean():.1%}  mean_points={grp.points.mean():.1f}')

    return sim


if __name__ == '__main__':
    import sys
    if '--toy' in sys.argv:
        toy_example()
    else:
        check_api()
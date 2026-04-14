"""
Game-specific prompt building, belief initialisation, and factory helpers
for LLM-driven epidemic simulations.
"""
import numpy as np
import pandas as pd
import starsim as ss


__all__ = ['default_agent_prompt', 'build_pregame_beliefs', 'init_beliefs_from_survey', 'make_intervention']

CHOICE_TO_SCORE = {c: i + 1 for i, c in enumerate(["a", "b", "c", "d", "e", "f"])}

QUESTION_TO_STATE = {
    35: "perceived_infection_risk",
    41: "perceived_infection_risk",
    36: "perceived_health_severity",
    42: "perceived_health_severity",
    37: "quarantine_self_efficacy",
    43: "quarantine_self_efficacy",
    38: "quarantine_response_efficacy",
    44: "quarantine_response_efficacy",
}

CORE_STATES = [
    "perceived_infection_risk",
    "perceived_health_severity",
    "quarantine_self_efficacy",
    "quarantine_response_efficacy",
]


def default_agent_prompt(mod, uid, disease):
    """
    Build the per-agent quarantine prompt for LLMIntervention.

    Args:
        mod (LLMIntervention): The calling module (gives access to ``ti``,
            ``low_reward``, ``high_reward``, and HBM belief states).
        uid (int): Agent UID.
        disease: The target disease module (used to determine health status).

    Returns:
        str: Prompt text sent to the LLM. Must end with a question that the
            LLM answers with only ``'yes'`` or ``'no'``.
    """
    status             = mod._agent_status(uid, disease)
    local_prev         = mod._local_prevalence(uid, disease)
    _has_been_infected = mod.has_been_infected[uid]
    return (
    f"You are playing an epidemic decision game where your goal is to maximise your total points.\n"
    f"\n"
    f"Game mechanics:\n"
    f"- A disease spreads through a contact network: interacting with others exposes you to infection risk.\n"
    f"- Your infection risk increases with local prevalence and your contacts.\n"
    f"- If infected, you may lose points (reduced rewards, possible large penalties).\n"
    f"- You move through health states (susceptible → infected → recovered).\n"
    f"\n"
    f"Decision each round:\n"
    f"- Quarantine: {mod.low_reward} pts. No infection risk this round.\n"
    f"- Stay active: {mod.reward_high[uid]:.0f} pts. Risk infection from contacts.\n"
    f"\n"
    f"This is a trade-off between:\n"
    f"- Short-term reward (staying active)\n"
    f"- Long-term risk (infection causing point losses)\n"
    f"\n"
    f"Your objective:\n"
    f"Maximise your total points over time. Consider expected future losses from infection, not just immediate reward.\n"
    f"\n"
    f"To help you make this decision, you are given initial beliefs related to epidemics and real-time epidemic information in the form of local prevalence, as well as your previous infection history.\n"
    f"\n"
    f"Your initial beliefs:\n"
    f"- These values come from your pregame survey responses.\n"
    f"- Each belief is on a 1 to 6 scale.\n"
    f"- 1 means the weakest possible belief.\n"
    f"- 6 means the strongest possible belief.\n"
    f"- Higher values mean stronger agreement or confidence.\n"
    f"\n"
    f"Belief framework:\n"
    f"- Perceived infection risk: how likely you think it is that you will get infected.\n"
    f"- Perceived health severity: how serious you think infection would be for your health.\n"
    f"- Quarantine self-efficacy: how confident you are that you can successfully follow quarantine.\n"
    f"- Quarantine response efficacy: how effective you think quarantine is at preventing spread.\n"
    f"\n"
    f"Local prevalence (0-1):\n"
    f"- This is the fraction of people you interacted with in the previous timestep who were infected.\n"
    f"- If local prevalence is high, your infection risk from staying active is high.\n"
    f"- If local prevalence is low, the risk from staying active is lower.\n"
    f"- Combine this with your beliefs and current points when deciding.\n"
    f"\n"
    f"Your current state:\n"
    f"- Time: {mod.ti}\n"
    f"- Status: {status}\n"
    f"- Infection History: {str(_has_been_infected)}\n"
    f"- Points: {mod.points[uid]:.0f}\n"
    f"- Local prevalence: {local_prev}\n"
    f"- Initial perceived infection risk (1-6): {mod.perceived_infection_risk[uid]:.2f}\n"
    f"- Initial perceived health severity (1-6): {mod.perceived_health_severity[uid]:.2f}\n"
    f"- Initial quarantine self-efficacy (1-6): {mod.quarantine_self_efficacy[uid]:.2f}\n"
    f"- Initial quarantine response efficacy (1-6): {mod.quarantine_response_efficacy[uid]:.2f}\n"
    f"\n"
    f"Use this framework to guide your decision. Should you quarantine this round? Reply with only 'yes' or 'no'."
)


def build_pregame_beliefs(
    answers_path: str,
    user_id_map: dict,
    default_value: float = 3.0,
) -> pd.DataFrame:
    """
    Return a user-indexed dataframe with one column per core belief/state.
    """
    answers = pd.read_csv(answers_path)

    answers = answers[answers["survey_id"].isin([3, 4])].copy()
    answers["score"] = (
        answers["value"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(CHOICE_TO_SCORE)
    )

    answers["dimension"] = answers["question_id"].map(QUESTION_TO_STATE)
    answers = answers.dropna(subset=["score", "dimension"])
    beliefs = (
        answers
        .groupby(["user_id", "dimension"])["score"]
        .mean()
        .unstack()
    )

    beliefs = beliefs.reindex(columns=CORE_STATES)

    # Fill missing values with the column median, then default_value if needed
    for col in CORE_STATES:
        if col in beliefs.columns:
            beliefs[col] = beliefs[col].fillna(beliefs[col].median())
        else:
            beliefs[col] = np.nan

    beliefs = beliefs.fillna(default_value)

    # Remap user_id -> sim uid
    beliefs = beliefs.reset_index()
    beliefs["uid"] = beliefs["user_id"].map(user_id_map)
    beliefs = beliefs.dropna(subset=["uid"]).copy()
    beliefs["uid"] = beliefs["uid"].astype(int)

    return beliefs.set_index("uid")[CORE_STATES]


def init_beliefs_from_survey(mod, answers_path: str, user_id_map: dict):
    beliefs = build_pregame_beliefs(answers_path, user_id_map)
    n = len(mod.sim.people)

    # Overwrite with survey-derived beliefs
    for uid, row in beliefs.iterrows():
        if uid >= n:
            continue
        mod.perceived_infection_risk[uid]     = float(row["perceived_infection_risk"])
        mod.perceived_health_severity[uid]    = float(row["perceived_health_severity"])
        mod.quarantine_self_efficacy[uid]     = float(row["quarantine_self_efficacy"])
        mod.quarantine_response_efficacy[uid] = float(row["quarantine_response_efficacy"])


def make_intervention(high_reward, agent_uids, name, model, api_key,
                      id_map=None, answers_path=None, group_b_uids=None, group_b_reward=None,
                      max_workers=12, rate_limit=100):
    def _init_beliefs(mod):
        if answers_path is not None and id_map is not None:
            init_beliefs_from_survey(mod, answers_path, user_id_map=id_map)
        if group_b_uids is not None:
            for uid in group_b_uids:
                mod.reward_high[uid] = group_b_reward

    return ss.LLMIntervention(
        low_reward    = 5,
        high_reward   = high_reward,
        agent_uids    = agent_uids,
        model         = model,
        api_key       = api_key,
        interval      = 1,
        decision_hour = 9.5,
        build_prompt  = default_agent_prompt,
        init_beliefs  = _init_beliefs,
        verbose       = True,
        name          = name,
        max_workers   = max_workers,
        rate_limit    = rate_limit,
    )

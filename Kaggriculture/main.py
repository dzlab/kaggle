from kagriculture_agent.policy import Policy


_policy = Policy()


def agent(obs):
    """Return a deterministic, legality-first action for a Kaggriculture turn.

    The stateful policy plans crop, animal, structure, routing, shed, and
    market work while preserving the Kaggle action schema and using ``PASS``
    whenever a requested task is not currently executable.
    """
    return _policy.act(obs)

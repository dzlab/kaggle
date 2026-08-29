from kagriculture_agent.policy import Policy


_policy = Policy()


def agent(obs):
    """Return the placeholder action for a Kaggle environment observation."""
    return _policy.act(obs)

import pytest


def test_native_rollout_manifest_contains_training_identity():
    from scripts.collect_trajectories import _manifest

    manifest = _manifest(
        seeds=[3], opponents=["pass"], seats=[0, 1], steps=4,
        source_policy_identity="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )

    assert manifest["source_policy_identity"] == "orbit-context-test"
    assert manifest["feature_variant"] == "experimental_context_v1"
    assert manifest["training_mode"] == "reduced_behavior_clone_then_ppo"


@pytest.mark.parametrize(
    ("field", "value"),
    [("feature_variant", "unknown"), ("training_mode", "unknown")],
)
def test_native_rollout_manifest_rejects_unknown_identity(field, value):
    from scripts.collect_trajectories import _manifest

    with pytest.raises(ValueError, match=field):
        _manifest(
            seeds=[3], opponents=["pass"], seats=[0], steps=4,
            source_policy_identity="orbit-policy-v1",
            **{field: value},
        )

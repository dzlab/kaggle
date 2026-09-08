import pytest


def test_colab_config_resolves_device_and_rejects_seed_overlap(monkeypatch, tmp_path):
    from scripts import colab_train

    monkeypatch.setattr(colab_train, "resolve_device", lambda value: "cuda")
    config = colab_train.build_config(
        run_directory=tmp_path, device="auto", workers=2,
        development_seeds=(1, 2), holdout_seeds=(3,),
    )
    assert config.device == "cuda"
    assert config.workers == 2
    with pytest.raises(ValueError, match="overlap"):
        colab_train.build_config(development_seeds=(1,), holdout_seeds=(1,))

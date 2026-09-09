import pytest


torch = pytest.importorskip("torch")


def _outputs(*, batch_size=2, worker_count=2):
    return {
        "worker_act_logits": torch.tensor(
            [[[0.5, -0.5], [0.2, 1.2]], [[-0.3, 0.7], [1.1, -0.1]]],
            dtype=torch.float32,
        )[:batch_size, :worker_count],
        "worker_target_logits": torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [[-0.1, -0.2, -0.3, -0.4], [1.0, 0.0, -1.0, -2.0]],
            ],
            dtype=torch.float32,
        )[:batch_size, :worker_count],
        "worker_kind_logits": torch.tensor(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[-0.1, -0.2, -0.3], [1.0, 0.0, -1.0]],
            ],
            dtype=torch.float32,
        )[:batch_size, :worker_count],
        "market_active_logits": torch.tensor(
            [[0.3, -0.3], [-0.2, 0.8]], dtype=torch.float32,
        )[:batch_size],
        "market_item_logits": torch.tensor(
            [[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]], dtype=torch.float32,
        )[:batch_size],
        "market_quantity_logits": torch.tensor(
            [[0.1, 0.2], [0.3, -0.4]], dtype=torch.float32,
        )[:batch_size],
    }


def _labels(*, batch_size=2, worker_count=2):
    return {
        "worker_active": torch.tensor([[0, 1], [0, 1]], dtype=torch.long)[:batch_size, :worker_count],
        "worker_target": torch.tensor([[0, 1], [2, 3]], dtype=torch.long)[:batch_size, :worker_count],
        "worker_kind": torch.tensor([[0, 1], [2, 0]], dtype=torch.long)[:batch_size, :worker_count],
        "market_active": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
        "market_item": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
        "market_quantity": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
    }


def _objective(outputs, labels, **masks):
    from kagriculture_agent.action_objectives import conditional_action_objectives

    return conditional_action_objectives(outputs, **labels, **masks)


def test_conditional_objective_ignores_inactive_worker_and_market_branches():
    outputs = _outputs()
    labels = _labels()
    log_probs, entropy = _objective(outputs, labels)

    changed = {name: value.clone() for name, value in outputs.items()}
    changed["worker_target_logits"][0, 0] = torch.tensor([1000.0, -1000.0, 500.0, -500.0])
    changed["worker_kind_logits"][0, 0] = torch.tensor([-1000.0, 1000.0, 500.0])
    changed["market_item_logits"][0] = torch.tensor([1000.0, -1000.0, 500.0])
    changed["market_quantity_logits"][0] = torch.tensor([1000.0, -1000.0])

    changed_log_probs, changed_entropy = _objective(changed, labels)

    torch.testing.assert_close(changed_log_probs, log_probs)
    torch.testing.assert_close(changed_entropy, entropy)


def test_no_market_order_excludes_item_and_quantity_objectives():
    outputs = _outputs()
    labels = _labels()
    labels["market_active"] = torch.zeros(2, dtype=torch.long)
    log_probs, entropy = _objective(outputs, labels)

    changed = {name: value.clone() for name, value in outputs.items()}
    changed["market_item_logits"] = torch.tensor([[1000.0, -1000.0, 500.0]] * 2)
    changed["market_quantity_logits"] = torch.tensor([[1000.0, -1000.0]] * 2)
    changed_log_probs, changed_entropy = _objective(changed, labels)

    torch.testing.assert_close(changed_log_probs, log_probs)
    torch.testing.assert_close(changed_entropy, entropy)


def test_market_active_probability_is_scored():
    outputs = _outputs(batch_size=1)
    labels = _labels(batch_size=1)
    baseline_log_probs, baseline_entropy = _objective(outputs, labels)

    changed = {name: value.clone() for name, value in outputs.items()}
    changed["market_active_logits"] = torch.tensor([[1000.0, -1000.0]])
    changed_log_probs, changed_entropy = _objective(changed, labels)

    assert not torch.allclose(changed_log_probs, baseline_log_probs)
    assert not torch.allclose(changed_entropy, baseline_entropy)


def test_conditional_objective_matches_factorized_log_probability_and_entropy():
    outputs = _outputs()
    labels = _labels()
    log_probs, entropy = _objective(outputs, labels)

    act_log = outputs["worker_act_logits"].log_softmax(dim=-1)
    target_log = outputs["worker_target_logits"].log_softmax(dim=-1)
    kind_log = outputs["worker_kind_logits"].log_softmax(dim=-1)
    market_active_log = outputs["market_active_logits"].log_softmax(dim=-1)
    item_log = outputs["market_item_logits"].log_softmax(dim=-1)
    quantity_log = outputs["market_quantity_logits"].log_softmax(dim=-1)
    worker_active = labels["worker_active"].bool()
    market_active = labels["market_active"].bool()

    expected_log_probs = act_log.gather(-1, labels["worker_active"].unsqueeze(-1)).squeeze(-1).sum(dim=1)
    expected_log_probs += (
        target_log.gather(-1, labels["worker_target"].unsqueeze(-1)).squeeze(-1)
        + kind_log.gather(-1, labels["worker_kind"].unsqueeze(-1)).squeeze(-1)
    ).masked_fill(~worker_active, 0.0).sum(dim=1)
    expected_log_probs += market_active_log.gather(-1, labels["market_active"].unsqueeze(-1)).squeeze(-1)
    expected_log_probs += (
        item_log.gather(-1, labels["market_item"].unsqueeze(-1)).squeeze(-1)
        + quantity_log.gather(-1, labels["market_quantity"].unsqueeze(-1)).squeeze(-1)
    ).masked_fill(~market_active, 0.0)

    def entropy_for(logits, active):
        values = -(logits.exp() * logits).sum(dim=-1)
        return values.masked_fill(~active, 0.0)

    expected_entropy = (
        entropy_for(act_log, torch.ones_like(worker_active, dtype=torch.bool)).sum(dim=1)
        + entropy_for(target_log, worker_active).sum(dim=1)
        + entropy_for(kind_log, worker_active).sum(dim=1)
        + entropy_for(market_active_log, torch.ones_like(market_active, dtype=torch.bool))
        + entropy_for(item_log, market_active)
        + entropy_for(quantity_log, market_active)
    ).mean()

    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)


def test_mask_is_applied_before_log_softmax_and_masks_inactive_rows_safely():
    outputs = _outputs(batch_size=1, worker_count=1)
    labels = _labels(batch_size=1, worker_count=1)
    masks = {
        "worker_target_mask": torch.tensor([[[True, False, False, False]]]),
        "worker_kind_mask": torch.tensor([[[True, False, False]]]),
        "market_item_mask": torch.tensor([[True, False, False]]),
        "market_quantity_mask": torch.tensor([[True, False]]),
    }

    log_probs, entropy = _objective(outputs, labels, **masks)
    expected = (
        outputs["worker_act_logits"].log_softmax(dim=-1)[0, 0, 0]
        + outputs["market_active_logits"].log_softmax(dim=-1)[0, 0]
    )

    torch.testing.assert_close(log_probs, expected.reshape(1))
    worker_entropy = -(
        outputs["worker_act_logits"].log_softmax(dim=-1).exp()
        * outputs["worker_act_logits"].log_softmax(dim=-1)
    ).sum()
    market_active_log = outputs["market_active_logits"].log_softmax(dim=-1)
    market_active_entropy = -(market_active_log.exp() * market_active_log).sum()
    torch.testing.assert_close(entropy, worker_entropy + market_active_entropy)


def test_active_market_legality_masks_change_the_active_distribution():
    outputs = _outputs(batch_size=1)
    labels = _labels(batch_size=1)
    labels["market_active"] = torch.ones(1, dtype=torch.long)
    unmasked_log_probs, unmasked_entropy = _objective(outputs, labels)

    masked_log_probs, masked_entropy = _objective(
        outputs,
        labels,
        market_item_mask=torch.tensor([[True, False, False]]),
        market_quantity_mask=torch.tensor([[True, False]]),
    )

    assert masked_log_probs.item() > unmasked_log_probs.item()
    assert masked_entropy.item() < unmasked_entropy.item()


@pytest.mark.parametrize(
    ("mask_name", "error_match"),
    [
        ("market_item_mask", "no legal market item"),
        ("market_quantity_mask", "no legal market quantity"),
    ],
)
def test_active_market_row_rejects_empty_legal_choices(mask_name, error_match):
    outputs = _outputs(batch_size=1)
    labels = _labels(batch_size=1)
    labels["market_active"] = torch.ones(1, dtype=torch.long)

    with pytest.raises(ValueError, match=error_match):
        _objective(outputs, labels, **{mask_name: torch.zeros_like(outputs[mask_name.replace("_mask", "_logits")], dtype=torch.bool)})


@pytest.mark.parametrize("bad_case", ["market_active_logits", "market_active", "worker_target"])
def test_malformed_market_or_label_shapes_are_rejected(bad_case):
    outputs = _outputs(batch_size=1, worker_count=1)
    labels = _labels(batch_size=1, worker_count=1)
    if bad_case == "market_active_logits":
        outputs[bad_case] = torch.zeros(1, 3)
        error_match = "market_active_logits"
    elif bad_case == "market_active":
        labels[bad_case] = torch.zeros(1, 1, dtype=torch.long)
        error_match = "market_active.*shape"
    else:
        labels[bad_case] = torch.zeros(1, dtype=torch.long)
        error_match = "worker_target.*shape"

    with pytest.raises(ValueError, match=error_match):
        _objective(outputs, labels)


@pytest.mark.parametrize(
    ("mask_name", "mask"),
    [
        ("worker_target_mask", torch.ones(1, 1, 3, dtype=torch.bool)),
        ("worker_kind_mask", torch.ones(1, 1, 2, dtype=torch.bool)),
        ("market_item_mask", torch.ones(1, 2, dtype=torch.bool)),
        ("market_quantity_mask", torch.ones(1, 1, dtype=torch.bool)),
    ],
)
def test_legality_masks_validate_dimensions(mask_name, mask):
    outputs = _outputs(batch_size=1, worker_count=1)
    labels = _labels(batch_size=1, worker_count=1)

    with pytest.raises(ValueError, match="mask.*shape|shape.*mask"):
        _objective(outputs, labels, **{mask_name: mask})


@pytest.mark.parametrize(
    ("mask_name", "mask", "error_match"),
    [
        ("worker_target_mask", torch.tensor([[[False, False, False, False]]]), "legal target"),
        ("worker_kind_mask", torch.tensor([[[False, False, False]]]), "legal kind"),
    ],
)
def test_active_worker_requires_a_legal_target_and_kind(mask_name, mask, error_match):
    outputs = _outputs(batch_size=1, worker_count=1)
    labels = _labels(batch_size=1, worker_count=1)
    labels["worker_active"] = torch.ones((1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match=error_match):
        _objective(outputs, labels, **{mask_name: mask})


def test_mask_rejects_an_illegal_selected_active_label():
    outputs = _outputs(batch_size=1, worker_count=1)
    labels = _labels(batch_size=1, worker_count=1)
    labels["worker_active"] = torch.ones((1, 1), dtype=torch.long)
    mask = torch.tensor([[[True, False, False, False]]])
    labels["worker_target"] = torch.tensor([[1]], dtype=torch.long)

    with pytest.raises(ValueError, match="worker target label"):
        _objective(outputs, labels, worker_target_mask=mask)


def test_empty_batch_returns_empty_log_probs_and_finite_zero_entropy():
    outputs = _outputs(batch_size=0)
    labels = _labels(batch_size=0)

    log_probs, entropy = _objective(outputs, labels)

    assert log_probs.shape == (0,)
    assert torch.isfinite(log_probs).all()
    assert entropy.shape == torch.Size([])
    assert torch.isfinite(entropy)
    assert entropy.item() == 0.0


def test_finite_logits_produce_finite_objectives():
    outputs = _outputs()
    for logits in outputs.values():
        logits.copy_(logits * 1_000_000.0)

    log_probs, entropy = _objective(outputs, _labels())

    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy)


def test_target_first_objective_scores_kind_conditioned_on_selected_target():
    from kagriculture_agent.action_objectives import conditional_action_objectives

    outputs = _outputs(batch_size=1, worker_count=1)
    outputs["worker_target_logits"] = torch.tensor([[[0.0, 2.0]]])
    outputs["worker_kind_logits"] = torch.tensor([[[
        [0.0, 10.0],
        [4.0, 0.0],
    ]]])
    outputs["market_item_logits"] = outputs["market_item_logits"][:, :1]
    outputs["market_quantity_logits"] = outputs["market_quantity_logits"][:, :1]
    labels = {
        "worker_active": torch.tensor([[1]]),
        "worker_target": torch.tensor([[1]]),
        "worker_kind": torch.tensor([[0]]),
        "market_active": torch.tensor([0]),
        "market_item": torch.tensor([0]),
        "market_quantity": torch.tensor([0]),
    }

    log_probs, _entropy = conditional_action_objectives(
        outputs, **labels, action_representation="target_first_v1",
    )
    expected = (
        outputs["worker_act_logits"].log_softmax(-1)[0, 0, 1]
        + outputs["worker_target_logits"].log_softmax(-1)[0, 0, 1]
        + outputs["worker_kind_logits"].log_softmax(-1)[0, 0, 1, 0]
        + outputs["market_active_logits"].log_softmax(-1)[0, 0]
    )
    torch.testing.assert_close(log_probs, expected.reshape(1))

    changed = {name: value.clone() for name, value in outputs.items()}
    changed["worker_kind_logits"][0, 0, 0] = torch.tensor([1000.0, -1000.0])
    changed_log_probs, _ = conditional_action_objectives(
        changed, **labels, action_representation="target_first_v1",
    )
    torch.testing.assert_close(changed_log_probs, log_probs)

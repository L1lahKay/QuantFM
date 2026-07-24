from quant_fm.pretrain.train import TrainState, _restore_train_state, cosine_lr


def test_new_train_state_round_trip_and_legacy_conversion() -> None:
    checkpoint = {
        "train_state": {
            "micro_step": 20,
            "update_step": 5,
            "samples_seen": 80,
            "non_pad_tokens_seen": 1234,
            "best_val": 1.2,
            "best_update_step": 4,
        }
    }
    assert _restore_train_state(checkpoint, grad_accum=4) == TrainState(
        micro_step=20,
        update_step=5,
        samples_seen=80,
        non_pad_tokens_seen=1234,
        best_val=1.2,
        best_update_step=4,
    )
    legacy = _restore_train_state(
        {"step": 20, "train_state": {"step": 20, "best_step": 16}},
        grad_accum=4,
    )
    assert legacy.micro_step == 20
    assert legacy.update_step == 5
    assert legacy.best_update_step == 4


def test_lr_is_indexed_by_optimizer_updates() -> None:
    values = [cosine_lr(step, warmup=2, max_steps=10, base_lr=1.0) for step in range(4)]
    assert values[0] == 0.5
    assert values[1] == 1.0
    assert values[2] == 1.0
    assert values[3] < values[2]

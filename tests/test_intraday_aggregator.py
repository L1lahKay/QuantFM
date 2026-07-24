from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quant_fm.embedding.intraday_aggregator import IntradayAggregator  # noqa: E402


def _model(*, n_layers: int = 2) -> IntradayAggregator:
    torch.manual_seed(19)
    model = IntradayAggregator(
        4,
        d_model=6,
        n_layers=n_layers,
        num_sessions=4,
        session_dim=3,
        time_frequencies=2,
        dropout=0.0,
    )
    return model.eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    summaries = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5, -0.5],
                [0.0, 2.0, 0.2, 0.1],
                [-1.0, 0.3, 1.5, 0.7],
            ]
        ]
    )
    time = torch.tensor([[0.10, 0.50, 0.90]])
    session = torch.tensor([[0, 1, 3]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    return summaries, time, session, mask


@pytest.mark.parametrize("n_layers", [1, 2])
def test_output_contract(n_layers: int) -> None:
    model = _model(n_layers=n_layers)
    output = model(*_inputs())

    assert tuple(output) == IntradayAggregator.OUTPUT_KEYS
    assert all(value.shape == (1, 6) for value in output.values())
    assert all(torch.isfinite(value).all() for value in output.values())


def test_padding_values_and_layout_do_not_change_summaries() -> None:
    model = _model()
    summaries, time, session, mask = _inputs()
    expected = model(summaries, time, session, mask)

    # 在有效 chunk 中间和末尾插入任意 padding；有效序列保持不变。
    padded_summaries = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5, -0.5],
                [999.0, -999.0, 5.0, 8.0],
                [0.0, 2.0, 0.2, 0.1],
                [-1.0, 0.3, 1.5, 0.7],
                [123.0, 456.0, 789.0, 10.0],
            ]
        ]
    )
    padded_time = torch.tensor([[0.10, float("nan"), 0.50, 0.90, float("nan")]])
    padded_session = torch.tensor([[0, -100, 1, 3, 999]])
    padded_mask = torch.tensor([[True, False, True, True, False]])
    actual = model(padded_summaries, padded_time, padded_session, padded_mask)

    for name in IntradayAggregator.OUTPUT_KEYS:
        assert torch.allclose(actual[name], expected[name], atol=1e-6)


def test_chunk_order_changes_ordered_summary() -> None:
    model = _model()
    summaries, time, session, mask = _inputs()
    ordered = model(summaries, time, session, mask)
    reordered = model(summaries.flip(1), time, session, mask)

    assert not torch.allclose(
        ordered["close_summary"],
        reordered["close_summary"],
    )
    assert not torch.allclose(
        ordered["intraday_trend_summary"],
        reordered["intraday_trend_summary"],
    )


def test_future_chunk_cannot_change_earlier_causal_hidden() -> None:
    model = _model()
    summaries, time, session, mask = _inputs()
    original = model.encode_chunks(summaries, time, session, mask)

    changed_summaries = summaries.clone()
    changed_summaries[:, 2] = torch.tensor([500.0, -700.0, 900.0, 200.0])
    changed_time = time.clone()
    changed_time[:, 2] = 0.99
    changed_session = session.clone()
    changed_session[:, 2] = 2
    changed = model.encode_chunks(
        changed_summaries,
        changed_time,
        changed_session,
        mask,
    )

    assert torch.equal(original[:, :2], changed[:, :2])
    assert not torch.equal(original[:, 2], changed[:, 2])

    # 即使未来位置从有效变为 padding，历史位置也不能感知全日序列长度。
    shortened_mask = mask.clone()
    shortened_mask[:, 2] = False
    shortened = model.encode_chunks(summaries, time, session, shortened_mask)
    assert torch.equal(original[:, :2], shortened[:, :2])


def test_all_padding_returns_finite_zeros() -> None:
    model = _model()
    summaries = torch.randn(2, 4, 4)
    time = torch.full((2, 4), float("nan"))
    session = torch.full((2, 4), -1)
    mask = torch.zeros((2, 4), dtype=torch.bool)

    hidden = model.encode_chunks(summaries, time, session, mask)
    output = model(summaries, time, session, mask)

    assert torch.equal(hidden, torch.zeros_like(hidden))
    for value in output.values():
        assert torch.equal(value, torch.zeros_like(value))


def test_rejects_noncausal_depth_and_invalid_valid_metadata() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        IntradayAggregator(4, n_layers=3)

    model = _model()
    summaries, time, session, mask = _inputs()
    bad_session = session.clone()
    bad_session[:, 1] = model.num_sessions
    with pytest.raises(ValueError, match="chunk_session"):
        model(summaries, time, bad_session, mask)

    bad_time = time.clone()
    bad_time[:, 1] = float("nan")
    with pytest.raises(ValueError, match="chunk_time"):
        model(summaries, bad_time, session, mask)

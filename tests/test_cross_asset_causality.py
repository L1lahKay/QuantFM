from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from quant_fm.cross_asset.clock_grid import (  # noqa: E402
    add_clock_interval,
    clock_interval_id,
)
from quant_fm.cross_asset.context_pool import (  # noqa: E402
    build_synchronous_context,
)


def _ms(hour: int, minute: int) -> int:
    return (hour * 60 + minute) * 60_000


def test_clock_grid_does_not_bridge_lunch_break() -> None:
    times = np.array(
        [
            _ms(9, 20),
            _ms(9, 30),
            _ms(9, 35),
            _ms(11, 30),
            _ms(12, 0),
            _ms(13, 0),
            _ms(15, 0),
        ]
    )
    assert clock_interval_id(times).tolist() == [0, 1, 2, 24, -1, 25, 48]


def test_packed_exchange_time_maps_deterministically() -> None:
    frame = pl.DataFrame({"int_time": [92_000_000, 93_000_000, 130_000_000]})
    actual = add_clock_interval(frame)["clock_interval"].to_list()
    assert actual == [0, 1, 25]


def test_industry_context_strictly_excludes_self() -> None:
    own = torch.tensor([[[1.0], [3.0], [10.0]]])
    context = build_synchronous_context(own, torch.tensor([0, 0, 1]))

    assert torch.allclose(context.market[0], torch.full((3, 1), 14.0 / 3.0))
    assert torch.equal(
        context.industry_leave_one_out[0],
        torch.tensor([[3.0], [1.0], [0.0]]),
    )
    assert context.industry_has_peer[0].tolist() == [True, True, False]
    assert context.own_minus_industry[0, 2].item() == 0.0


def test_future_peer_changes_do_not_change_current_interval() -> None:
    own = torch.tensor(
        [
            [[1.0], [3.0]],
            [[2.0], [4.0]],
        ]
    )
    changed = own.clone()
    changed[1, 1] = 10_000.0
    industries = torch.tensor([0, 0])

    baseline = build_synchronous_context(own, industries)
    modified = build_synchronous_context(changed, industries)

    assert torch.equal(baseline.market[0], modified.market[0])
    assert torch.equal(
        baseline.industry_leave_one_out[0],
        modified.industry_leave_one_out[0],
    )


def test_inactive_stock_is_excluded_from_all_pools() -> None:
    own = torch.tensor([[[1.0], [999.0], [3.0]]])
    active = torch.tensor([[True, False, True]])
    context = build_synchronous_context(
        own,
        torch.tensor([0, 0, 0]),
        active_mask=active,
    )

    assert context.market[0, 0].item() == pytest.approx(2.0)
    assert context.market[0, 1].item() == 0.0
    assert context.industry_leave_one_out[0, 0].item() == 3.0
    assert context.industry_leave_one_out[0, 2].item() == 1.0

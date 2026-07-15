from dataclasses import asdict, dataclass


@dataclass(slots=True, frozen=True)
class ColumnIndices:
    """列索引管理 - 不可变."""

    int_time: int
    exchange_time: int
    serial: int
    type: int
    trade_type: int
    order_type: int
    orderorino: int
    sell_id: int
    buy_id: int
    bsflag: int
    trade_price: int
    order_price: int
    trade_volume: int
    order_volume: int

    @classmethod
    def from_columns(
        cls, cols: list[str], *, time_col: str = "int_time"
    ) -> "ColumnIndices":
        """
        从列名列表创建索引.

        Parameters
        ----------
        cols : list[str]
            合并后 DataFrame 的列名列表
        time_col : str
            时间列的实际列名，默认 ``"int_time"``
        """

        def _idx(name: str) -> int:
            if name not in cols:
                msg = f"缺少必要列 '{name}', 实际列={cols}"
                raise KeyError(msg)
            return cols.index(name)

        return cls(
            int_time=_idx(time_col),
            exchange_time=_idx("exchange_time"),
            serial=_idx("serial"),
            type=_idx("type"),
            trade_type=_idx("trade_type"),
            order_type=_idx("order_type"),
            orderorino=_idx("orderorino"),
            sell_id=_idx("sell_id"),
            buy_id=_idx("buy_id"),
            bsflag=_idx("bsflag"),
            trade_price=_idx("trade_price"),
            order_price=_idx("order_price"),
            trade_volume=_idx("trade_volume"),
            order_volume=_idx("order_volume"),
        )

    def as_dict(self) -> dict[str, int]:
        """Return field name to column index mapping as a dict."""
        return asdict(self)

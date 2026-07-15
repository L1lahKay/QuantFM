from dataclasses import dataclass
from enum import Enum


@dataclass(slots=True)
class Order:
    """订单结构体."""

    order_id: int
    side: "Side"
    price: int
    quantity: int
    order_time: int  # int_time 格式，如 93000000
    order_type: str  # '0'表示限价单，'1'表示市价单


@dataclass(slots=True)
class Trade:
    """交易记录结构体."""

    trade_id: int
    buy_order_id: int
    sell_order_id: int
    price: int
    quantity: int
    trade_time: int  # int_time 格式，如 93000000


@dataclass(slots=True)
class Cancel:
    """取消记录结构体."""

    cancel_id: int
    order_id: int
    side: "Side"
    price: int
    quantity: int
    order_time: int
    cancel_time: int


class Side(Enum):
    """订单方向."""

    BUY = "买入"
    SELL = "卖出"


class TradingPhase(Enum):
    """交易阶段."""

    CALL_AUCTION = "集合竞价"
    CONTINUOUS = "连续竞价"

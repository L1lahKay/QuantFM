import numpy as np
import polars as pl

from pylob._utils import normalize_dtypes
from pylob.cols_mapping import ColumnIndices
from pylob.data_types import TradingPhase
from pylob.event_ordering import (
    DEFAULT_EVENT_ORDERING_VERSION,
    order_market_events,
)
from pylob.matching_engine import MatchingEngine
from pylob.result_mixin import ResultMixin


class OrderBookSH(ResultMixin, MatchingEngine):
    """上海订单簿实现，支持限价单."""

    def __init__(self, matching_date=None):
        super().__init__()
        self.matching_date = matching_date
        self.market_orders = {}
        self.market_trades = {}
        self.data_status = "O"
        self.current_trade_id = 0
        self.current_trade_id_old = 0

    def prepare_market_data(
        self,
        trade_df_SH,
        order_df_SH,
        symbol: str,
        cut_time=150100000,
        cut_serial=None,
        *,
        event_ordering_version: str = DEFAULT_EVENT_ORDERING_VERSION,
    ):
        """
        准备市场数据处理环境.

        Args:
            trade_df_SH: 交易数据DataFrame (polars)
            order_df_SH: 订单数据DataFrame (polars)
            symbol: 股票代码
            cut_time: int_time 93500000
            cut_serial: serial
            event_ordering_version: 合并事件排序契约；默认交易所时间与序号。

        Returns
        -------
            order_df: 合并排序后的pandas DataFrame
        """
        # 统一列类型（UInt64→Int64, Binary→Utf8），防止下游 concat SchemaError
        trade_df_SH = normalize_dtypes(trade_df_SH)
        order_df_SH = normalize_dtypes(order_df_SH)

        # 数据预处理
        trade_df_test = (
            trade_df_SH.with_columns(
                pl.col("symbol").cast(pl.String).str.slice(0, 6),
                pl.col("int_time").cast(pl.Int64),
            )
            .filter(pl.col("symbol") == symbol)
            .with_columns(pl.lit("T").alias("type"))
            .filter(
                (pl.col("int_time") < int(cut_time))
                & (pl.col("serial") < cut_serial if cut_serial is not None else True)
            )
        )
        self.set_trade_df(trade_df_test)

        order_df_test = (
            order_df_SH.with_columns(
                pl.col("symbol").cast(pl.String).str.slice(0, 6),
                pl.col("int_time").cast(pl.Int64),
            )
            .filter(pl.col("symbol") == symbol)
            .with_columns(pl.lit("O").alias("type"))
            .filter(
                (pl.col("int_time") < int(cut_time))
                & (pl.col("serial") < cut_serial if cut_serial is not None else True)
            )
        )

        # local_time 是行情接收时间，不能决定交易所事件因果顺序。旧产物可通过
        # local_time_v1 显式复现；新产物默认按 int_time + serial 稳定排序。
        order_df = order_market_events(
            pl.concat([trade_df_test, order_df_test], how="diagonal"),
            version=event_ordering_version,
        ).to_pandas()

        cols = list(order_df.columns)

        self.indices = ColumnIndices.from_columns(cols)
        self.column_indices = self.indices.as_dict()

        self._update_logger_for_symbol(symbol)

        # 重置状态
        self.reset_market_state()

        return order_df

    def finalize_trading_session(self):
        """结束交易会话，处理最后的pending trade."""
        # 处理最后一笔未完成的交易
        if (
            self.data_status == "T"
            and self.current_trade_id in self.market_trades
            and self.trading_phase == TradingPhase.CONTINUOUS
        ):
            self._finalize_pending_trade()

        # 如果还在集合竞价阶段，执行最后的撮合
        if self.trading_phase == TradingPhase.CALL_AUCTION:
            self.call_auction_match()

        # 静默处理交易会话结束

    def _auto_detect_trading_phase(self, row_data):
        """
        基于时间自动检测并切换交易阶段.

        Args:
            row_data: np.ndarray - 单条记录数据
        """
        if not hasattr(self, "column_indices"):
            self.logger.error("请先调用 prepare_market_data 初始化列索引")
            return

        # 从数据中获取时间
        int_time = row_data[self.column_indices["int_time"]]

        # 根据时间判断交易阶段
        if int_time < 93000000:
            # 开盘集合竞价
            if self.trading_phase != TradingPhase.CALL_AUCTION:
                self.set_trading_phase(TradingPhase.CALL_AUCTION)
        elif 93000000 <= int_time < 145700000:
            # 连续竞价
            if self.trading_phase != TradingPhase.CONTINUOUS:
                # 从集合竞价切换到连续竞价前，先执行集合竞价撮合
                if self.trading_phase == TradingPhase.CALL_AUCTION:
                    self.logger.debug("执行开盘集合竞价撮合")
                    self.call_auction_match()

                self.logger.debug("切换到连续竞价阶段")
                self.set_trading_phase(TradingPhase.CONTINUOUS)
                self.first_continuous = True
        else:  # >= 145700000
            # 收盘集合竞价
            if self.trading_phase == TradingPhase.CONTINUOUS:
                # 从连续竞价切换到集合竞价前，处理pending trade
                if (
                    self.data_status == "T"
                    and self.current_trade_id in self.market_trades
                ):
                    self._finalize_pending_trade()

                self.logger.debug("切换到收盘集合竞价阶段")
                self.set_trading_phase(TradingPhase.CALL_AUCTION)

    def reset_market_state(self):
        """重置市场数据处理状态，防止同一对象处理多个 symbol 时状态污染."""
        super().reset_market_state()
        # SH 特有状态
        self.market_orders = {}
        self.market_trades = {}
        self.data_status = "O"
        self.current_trade_id = 0
        self.current_trade_id_old = 0

    def _process_single_record(
        self, row_data: np.ndarray, *, is_continuous: bool = False
    ):
        """处理单条记录."""
        indices = self.column_indices
        record_type = row_data[indices["type"]]

        if record_type == "T":
            self._process_trade_record(row_data, is_continuous=is_continuous)
        elif record_type == "O":
            self._process_order_record(row_data, is_continuous=is_continuous)

    def _process_trade_record(self, row_data: np.ndarray, *, is_continuous: bool):
        """处理交易记录."""
        indices = self.column_indices
        trade_id = int(max(row_data[indices["sell_id"]], row_data[indices["buy_id"]]))

        if is_continuous:
            # 连续竞价阶段的复杂逻辑
            # 只有在确实有pending trade且是不同trade_id时才finalize
            if (
                self.data_status == "T"
                and trade_id != self.current_trade_id
                and not self.first_continuous
                and self.current_trade_id in self.market_trades
            ):
                # 上一笔trade完全成交，需要作为order输入
                self._finalize_pending_trade()
                self.data_status = "O"

        # 创建或更新trade记录
        if trade_id not in self.market_trades:
            self.market_trades[trade_id] = row_data.copy()
            self.current_trade_id_old = self.current_trade_id
            self.current_trade_id = trade_id
        else:
            # 累加数量，更新价格（买单取最高价，卖单取最低价）
            trade_order = self.market_trades[trade_id]
            trade_order[indices["trade_volume"]] += row_data[indices["trade_volume"]]

            if trade_order[indices["bsflag"]] == "B":
                trade_order[indices["trade_price"]] = max(
                    trade_order[indices["trade_price"]],
                    row_data[indices["trade_price"]],
                )
            else:
                trade_order[indices["trade_price"]] = min(
                    trade_order[indices["trade_price"]],
                    row_data[indices["trade_price"]],
                )

        self.data_status = "T"

    def _process_order_record(self, row_data: np.ndarray, *, is_continuous: bool):
        """处理订单记录."""
        indices = self.column_indices
        order_id = int(row_data[indices["orderorino"]])

        # 处理取消订单
        if row_data[indices["order_type"]] == "D":
            if (
                is_continuous
                and self.data_status == "T"
                and order_id != self.current_trade_id
                and not self.first_continuous
                and self.current_trade_id in self.market_trades
            ):
                self._finalize_pending_trade()

            self.cancel_order(order_id, cancel_time=row_data[indices["int_time"]])
            if order_id in self.market_orders:
                del self.market_orders[order_id]
            self.data_status = "O"
            return

        # 处理普通订单
        if (
            is_continuous
            and self.data_status == "T"
            and not self.first_continuous
            and self.current_trade_id in self.market_trades
        ):
            if order_id != self.current_trade_id:
                # 完全成交的情况
                self._finalize_pending_trade()
                self._add_new_order(row_data, order_id)
            else:
                # 部分成交的情况
                self._handle_partial_trade(row_data, order_id)
        else:
            # 集合竞价阶段或连续竞价的新订单
            self._add_new_order(row_data, order_id)

        self.data_status = "O"

    def _finalize_pending_trade(self):
        """完成待处理的交易."""
        if self.current_trade_id in self.market_trades:
            indices = self.column_indices
            trade_order = self.market_trades[self.current_trade_id]

            order_time = trade_order[indices["int_time"]]
            side = self._parse_side(str(trade_order[indices["bsflag"]]))

            self.add_order(
                price=int(trade_order[indices["trade_price"]]),
                quantity=int(trade_order[indices["trade_volume"]]),
                side=side,
                order_id=self.current_trade_id,
                order_time=order_time,
                order_type="0",
            )

            self.market_orders[self.current_trade_id] = trade_order
            del self.market_trades[self.current_trade_id]

    def _add_new_order(self, row_data: np.ndarray, order_id: int):
        """添加新订单."""
        indices = self.column_indices
        market_order = row_data
        self.market_orders[order_id] = market_order
        order_time = row_data[indices["int_time"]]
        side = self._parse_side(str(row_data[indices["bsflag"]]))
        self.add_order(
            price=int(market_order[indices["order_price"]]),
            quantity=int(market_order[indices["order_volume"]]),
            side=side,
            order_id=order_id,
            order_time=order_time,
            order_type="0",
        )

    def _handle_partial_trade(self, row_data: np.ndarray, order_id: int):
        """处理部分成交的订单."""
        indices = self.column_indices

        # 这笔order没完全成交，总报单量加上之前的trade
        if self.current_trade_id in self.market_trades:
            trade_order = self.market_trades[self.current_trade_id]
            trade_order[indices["trade_volume"]] += row_data[indices["order_volume"]]
            trade_order[indices["trade_price"]] = row_data[indices["order_price"]]

            self.market_orders[order_id] = trade_order

            order_time = trade_order[indices["int_time"]]

            side = self._parse_side(str(trade_order[indices["bsflag"]]))

            self.add_order(
                price=int(trade_order[indices["trade_price"]]),
                quantity=int(trade_order[indices["trade_volume"]]),
                side=side,
                order_id=order_id,
                order_time=order_time,
                order_type="0",
            )

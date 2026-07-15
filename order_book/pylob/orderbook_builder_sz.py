import numpy as np
import pandas as pd
import polars as pl

from pylob._utils import normalize_dtypes
from pylob.cols_mapping import ColumnIndices
from pylob.data_types import Order, Side, TradingPhase
from pylob.matching_engine import MatchingEngine
from pylob.result_mixin import ResultMixin


class OrderBookSZ(ResultMixin, MatchingEngine):
    """深圳订单簿实现，支持限价单、市价单和本地最优单."""

    def __init__(self, matching_date=None):
        super().__init__()
        self.matching_date = matching_date

    def reset_market_state(self):
        """重置市场数据处理状态，防止同一对象处理多个 symbol 时状态污染."""
        super().reset_market_state()

    def prepare_market_data(
        self,
        trade_df_SZ: pl.DataFrame,
        order_df_SZ: pl.DataFrame,
        symbol: str,
        cut_time=150100000,
        cut_serial=None,
    ):
        """
        准备市场数据处理环境.

        Args:
            trade_df_SZ: 交易数据DataFrame (polars)
            order_df_SZ: 订单数据DataFrame (polars)
            symbol: 股票代码
            cut_time: int_time 93500000
            cut_serial: serial int

        Returns
        -------
            order_df: 合并排序后的pandas DataFrame
        """
        # 统一列类型（UInt64→Int64, Binary→Utf8），防止下游 concat SchemaError
        trade_df_SZ = normalize_dtypes(trade_df_SZ)
        order_df_SZ = normalize_dtypes(order_df_SZ)

        # --- trade df: symbol/int_time/serial 强制类型 ---
        tdf = (
            trade_df_SZ.with_columns(
                pl.col("symbol").cast(pl.String).str.slice(0, 6),
                pl.col("int_time").cast(pl.Int64),
                pl.col("serial").cast(pl.Int64),
            )
            .filter(pl.col("symbol") == symbol)
            .with_columns(pl.lit("T").alias("type"))
        )

        t_filters = [pl.col("int_time") < cut_time]
        if cut_serial is not None:
            t_filters.append(pl.col("serial") < cut_serial)
        trade_df_test = tdf.filter(pl.all_horizontal(t_filters))

        trade_df_full = trade_df_test.filter(pl.col("trade_type") != "C")

        self.trade_df = trade_df_full.to_pandas()
        self.trade_df_with_c = trade_df_test.to_pandas()

        # --- order df: symbol/int_time/serial 强制类型 ---
        odf = (
            order_df_SZ.with_columns(
                pl.col("symbol").cast(pl.String).str.slice(0, 6),
                pl.col("int_time").cast(pl.Int64),
                pl.col("serial").cast(pl.Int64),
            )
            .filter(pl.col("symbol") == symbol)
            .with_columns(pl.lit("O").alias("type"))
        )

        o_filters = [pl.col("int_time") < cut_time]
        if cut_serial is not None:
            o_filters.append(pl.col("serial") < cut_serial)
        order_df_test = odf.filter(pl.all_horizontal(o_filters))

        # 合并、按 serial 排序
        order_df = (
            pl.concat([trade_df_test, order_df_test], how="diagonal")
            .sort("local_time")
            .to_pandas()
        )

        cols = list(order_df.columns)

        self.indices = ColumnIndices.from_columns(cols)
        self.column_indices = self.indices.as_dict()

        self._update_logger_for_symbol(symbol)

        # 重置市场状态，防止重复使用同一对象时的污染
        self.reset_market_state()

        return order_df

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
                self.logger.debug("切换到收盘集合竞价阶段")
                self.set_trading_phase(TradingPhase.CALL_AUCTION)

    def _process_single_record(
        self, row_data: np.ndarray, *, is_continuous: bool = False
    ):
        """处理单条记录."""
        indices = self.column_indices
        record_type = row_data[indices["type"]]
        trade_type = row_data[indices["trade_type"]]

        if record_type == "T" and trade_type == "C":
            cancel_order_id = int(
                max(row_data[indices["sell_id"]], row_data[indices["buy_id"]])
            )
            cancel_quantity = int(row_data[indices["trade_volume"]])
            self.cancel_order(
                cancel_order_id,
                cancel_time=row_data[indices["int_time"]],
                cancel_quantity=cancel_quantity,
            )

        elif record_type == "O":
            side = self._parse_side(row_data[indices["bsflag"]])
            self.add_order(
                price=int(row_data[indices["order_price"]]),
                quantity=int(row_data[indices["order_volume"]]),
                side=side,
                order_id=int(row_data[indices["orderorino"]]),
                order_time=row_data[indices["int_time"]],
                order_type=row_data[indices["order_type"]],
            )

    def finalize_trading_session(self):
        """结束交易会话，处理最后的pending trade."""
        # 如果还在集合竞价阶段，执行最后的撮合
        if self.trading_phase == TradingPhase.CALL_AUCTION:
            self.call_auction_match()

        if self.row_data_time > 145700000:
            self.logger.debug(
                f"收盘集合竞价处理完成，处理了{self.process_num - self.cont_count}条记录({self.process_percent:.2f}%)"
            )
            self.logger.debug("*************** 交易会话处理完成 ***************")

    def _query_market_order_type(self, order_id: int, side: Side) -> int:
        """
        查询市价单类型.

        Args:
            order_id: 订单ID
            side: 订单方向

        Returns
        -------
            0: 市转限（只有一个成交价格）或符合限价单逻辑
            1: 全吃单（有多个成交价格）或不符合限价单逻辑
            -1: 未找到对应交易记录
            -2: 市转撤（部分成交后撤单）
        """
        if self.trade_df_with_c is None:
            self.logger.debug(f"未设置trade_df，订单 {order_id} 将使用默认逻辑")
            return -1, None

        try:
            # 根据订单方向确定查询列
            if side == Side.BUY:
                # 买单查询buy_order_id列
                order_trades = self.trade_df_with_c[
                    self.trade_df_with_c["buy_id"] == order_id
                ]
            else:
                # 卖单查询sell_order_id列
                order_trades = self.trade_df_with_c[
                    self.trade_df_with_c["sell_id"] == order_id
                ]

            if len(order_trades) == 0:
                self.logger.debug(f"未找到订单 {order_id} 的交易记录")
                return -1, None

            # ✅ 修复：验证交易方向是否匹配（只对正常成交记录过滤，撤单记录无条件保留）
            if "bsflag" in order_trades.columns:
                expected_flag = "B" if side == Side.BUY else "S"

                # 分离正常成交记录和撤单记录
                normal_trades = order_trades[order_trades["trade_type"] != "C"]
                cancel_trades = order_trades[order_trades["trade_type"] == "C"]

                # 只对正常成交记录过滤 bsflag
                valid_normal_trades = normal_trades[
                    normal_trades["bsflag"] == expected_flag
                ]

                if len(valid_normal_trades) == 0 and len(cancel_trades) == 0:
                    self.logger.warning(
                        f"订单 {order_id} 的成交记录中没有方向匹配的记录！"
                        f"预期方向: {expected_flag}, 实际记录数: {len(order_trades)}"
                    )
                    return -1, None

                # ✅ 合并：方向匹配的正常成交 + 所有撤单记录

                order_trades = pd.concat(
                    [valid_normal_trades, cancel_trades], ignore_index=True
                )

                self.logger.debug(
                    f"订单 {order_id} 筛选后有 {len(valid_normal_trades)} 条方向匹配的成交记录，"
                    f"{len(cancel_trades)} 条撤单记录"
                )

            unique_prices = order_trades["trade_price"].nunique()

            self.logger.debug(
                f"订单 {order_id} 在trade_df中找到 {len(order_trades)} 条交易记录，涉及 {unique_prices} 个不同价格"
            )

            # 判断市价单类型
            if int(order_trades["trade_price"].min()) == 0:  # 市转撤
                unique_prices = order_trades[order_trades["trade_price"] != 0][
                    "trade_price"
                ].nunique()
                if unique_prices == 1:
                    self.logger.debug(
                        f"订单 {order_id} 只有一个成交价格，判定为市转限模式"
                    )
                    return 0, None  # 市转限
                else:
                    self.logger.debug(
                        f"订单 {order_id} 最后一笔为撤单，判定为市转撤模式"
                    )
                    new_quantity = order_trades["trade_volume"].iloc[-1]
                    return -2, new_quantity
            else:
                if unique_prices == 1:
                    self.logger.debug(
                        f"订单 {order_id} 只有一个成交价格，判定为市转限模式"
                    )
                    return 0, None  # 市转限
                else:
                    self.logger.debug(f"订单 {order_id} 有多个成交价格，判定为全吃模式")
                    return 1, None  # 全吃单

        except Exception:
            self.logger.exception("查询订单 %s 的交易记录时出错", order_id)
            return -1, None

    def _get_local_optimal_price(self, side: Side) -> float | None:
        """
        获取本地最优价格.

        买单：当前最优买价（买一价格）
        卖单：当前最优卖价（卖一价格）.
        """
        if side == Side.BUY:
            # 买单：获取当前最高买价
            if self.bids:
                # 获取有效订单的最高买价
                for price in reversed(self.bids):
                    if any(order.quantity > 0 for order in self.bids[price]):
                        return price
            return None
        else:  # Side.SELL
            # 卖单：获取当前最低卖价
            if self.asks:
                # 获取有效订单的最低卖价
                for price in self.asks:
                    if any(order.quantity > 0 for order in self.asks[price]):
                        return price
            return None

    def _match_market_order(
        self, market_order: Order, left_quantity: int | None = None
    ):
        """
        市价单撮合逻辑（支持两种模式）.

        模式1: price=0 - 价格锁定模式
        - 第一次成交后价格锁定，只在该价格上继续成交
        - 未成交部分转挂限价单

        模式2: price=-2 - 后续撤单
        - 市价成交部分后撤单

        模式3: 其他price值 - 立即成交模式
        - 按市场最优价格依次成交，能成交多少就成交多少
        - 不转挂限价单，剩余部分直接失效
        """
        self.logger.debug(f"开始处理市价单 {market_order.order_id}...")

        original_quantity = market_order.quantity

        if market_order.price == 0:
            # 模式1: 价格锁定模式（市转限）
            self._match_market_order_lock_price(market_order, original_quantity)
        elif market_order.price == -2:
            # 模式2: 立即成交模式（市转撤）
            self._match_market_order_cancel(
                market_order, original_quantity, left_quantity
            )
        else:
            # 模式3: 立即成交模式
            self._match_market_order_immediate(market_order, original_quantity)

    def _match_market_order_lock_price(
        self, market_order: Order, original_quantity: int
    ):
        """
        市价单价格锁定模式（price=0）.

        第一次成交后价格锁定，未成交部分转挂限价单.
        """
        locked_price = None

        if market_order.side == Side.BUY:
            # 买入市价单：找到最低价卖单进行成交，价格锁定
            ask_prices = list(self.asks.keys())

            # 找到第一个有效的价格档位
            for price in ask_prices:
                price_level = self.asks[price]
                if price_level and any(order.quantity > 0 for order in price_level):
                    locked_price = price
                    break

            if locked_price is None:
                self.logger.debug(
                    f"市价买单 {market_order.order_id} 无法成交：没有可用的卖单"
                )
                return

            self.logger.debug(
                f"市价买单 {market_order.order_id} 锁定成交价格: {locked_price:.2f}"
            )

            # 只在锁定价格上成交
            price_level = self.asks[locked_price]
            orders_to_remove = []

            while market_order.quantity > 0 and price_level:
                maker_order = price_level[0]

                if maker_order.quantity <= 0:
                    orders_to_remove.append(maker_order)
                    price_level.popleft()
                    continue

                trade_quantity = min(market_order.quantity, maker_order.quantity)

                if trade_quantity <= 0:
                    break

                # 市价单以锁定价格成交
                self._execute_trade(
                    maker_order, market_order, locked_price, trade_quantity
                )

                # 更新订单数量
                maker_order.quantity -= trade_quantity
                market_order.quantity -= trade_quantity

                if maker_order.quantity <= 0:
                    orders_to_remove.append(maker_order)
                    price_level.popleft()
                    self.logger.debug(f"订单 {maker_order.order_id} 已完全成交。")

            # 清理完全成交的订单
            for order in orders_to_remove:
                if order.order_id in self.orders:
                    del self.orders[order.order_id]

            # 如果价格层级为空，删除该价格层级
            if not price_level:
                del self.asks[locked_price]

        else:  # Side.SELL
            # 卖出市价单：找到最高价买单进行成交，价格锁定
            bid_prices = list(reversed(self.bids.keys()))

            # 找到第一个有效的价格档位
            for price in bid_prices:
                price_level = self.bids[price]
                if price_level and any(order.quantity > 0 for order in price_level):
                    locked_price = price
                    break

            if locked_price is None:
                self.logger.debug(
                    f"市价卖单 {market_order.order_id} 无法成交：没有可用的买单"
                )
                return

            self.logger.debug(
                f"市价卖单 {market_order.order_id} 锁定成交价格: {locked_price:.2f}"
            )

            # 只在锁定价格上成交
            price_level = self.bids[locked_price]
            orders_to_remove = []

            while market_order.quantity > 0 and price_level:
                maker_order = price_level[0]

                if maker_order.quantity <= 0:
                    orders_to_remove.append(maker_order)
                    price_level.popleft()
                    continue

                trade_quantity = min(market_order.quantity, maker_order.quantity)

                if trade_quantity <= 0:
                    break

                # 市价单以锁定价格成交
                self._execute_trade(
                    maker_order, market_order, locked_price, trade_quantity
                )

                # 更新订单数量
                maker_order.quantity -= trade_quantity
                market_order.quantity -= trade_quantity

                if maker_order.quantity <= 0:
                    orders_to_remove.append(maker_order)
                    price_level.popleft()
                    self.logger.debug(f"订单 {maker_order.order_id} 已完全成交。")

            # 清理完全成交的订单
            for order in orders_to_remove:
                if order.order_id in self.orders:
                    del self.orders[order.order_id]

            # 如果价格层级为空，删除该价格层级
            if not price_level:
                del self.bids[locked_price]

        # 处理未完全成交的部分（转挂限价单）
        if market_order.quantity > 0:
            if locked_price is not None:
                # 有成交记录，将剩余部分转挂限价单
                traded_quantity = original_quantity - market_order.quantity
                self.logger.debug(
                    f"市价单 {market_order.order_id} 部分成交：{traded_quantity}手@{locked_price:.2f}，"
                    f"剩余 {market_order.quantity}手 转挂限价单@{locked_price:.2f}"
                )

                # 修改订单类型为限价单，价格为成交价格
                market_order.order_type = "0"  # 转为限价单
                market_order.price = locked_price  # 设置限价

                # 将剩余部分挂单
                self._add_to_book(market_order)
                self.logger.debug(f"订单 {market_order.order_id} 剩余部分已转挂限价单")
            else:
                # 没有任何成交，完全无法成交
                self.logger.debug(
                    f"市价单 {market_order.order_id} 无法成交，剩余数量 {market_order.quantity}（市场无流动性）"
                )
                if market_order.order_id in self.orders:
                    del self.orders[market_order.order_id]
        else:
            # 完全成交
            self.logger.debug(f"市价单 {market_order.order_id} 已完全成交。")
            if market_order.order_id in self.orders:
                del self.orders[market_order.order_id]

    def _match_market_order_cancel(
        self, market_order: Order, original_quantity: int, left_quantity: int
    ):
        """
        市价单立即成交模式.

        按市场最优价格依次成交，成交过程中直接撤单.
        """
        order_quantity = market_order.quantity - left_quantity
        if market_order.side == Side.BUY:
            # 买入市价单：从最低价卖单开始依次成交
            ask_prices = list(self.asks.keys())

            self.logger.debug(f"市价买单 {market_order.order_id} 开始立即成交模式")

            for price in ask_prices:
                if order_quantity <= 0:
                    break

                price_level = self.asks[price]
                orders_to_remove = []

                self.logger.debug(f"开始成交价格档位 {price:.2f} 的卖单...")

                while order_quantity > 0 and price_level:
                    maker_order = price_level[0]

                    if maker_order.quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        continue

                    trade_quantity = min(order_quantity, maker_order.quantity)

                    if trade_quantity <= 0:
                        break

                    # 按当前价格档位成交
                    self._execute_trade(
                        maker_order, market_order, price, trade_quantity
                    )

                    # 更新订单数量
                    maker_order.quantity -= trade_quantity
                    order_quantity -= trade_quantity
                    market_order.quantity -= trade_quantity

                    if order_quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        self.logger.debug(
                            f"订单 {maker_order.order_id} 非撤单部分已完全成交。"
                        )

                # 如果价格层级为空，删除该价格层级
                if not price_level:
                    del self.asks[price]

        else:  # Side.SELL
            # 卖出市价单：从最高价买单开始依次成交
            bid_prices = list(reversed(self.bids.keys()))

            self.logger.debug(f"市价卖单 {market_order.order_id} 开始立即成交模式")

            for price in bid_prices:
                if order_quantity <= 0:
                    break

                price_level = self.bids[price]
                orders_to_remove = []

                self.logger.debug(f"开始成交价格档位 {price:.2f} 的买单...")

                while order_quantity > 0 and price_level:
                    maker_order = price_level[0]

                    if maker_order.quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        continue

                    trade_quantity = min(order_quantity, maker_order.quantity)

                    if trade_quantity <= 0:
                        break

                    # 按当前价格档位成交
                    self._execute_trade(
                        maker_order, market_order, price, trade_quantity
                    )

                    # 更新订单数量
                    maker_order.quantity -= trade_quantity
                    order_quantity -= trade_quantity
                    market_order.quantity -= trade_quantity

                    if order_quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        self.logger.debug(
                            f"订单 {maker_order.order_id} 非撤单部分已完全成交。"
                        )

                # 如果价格层级为空，删除该价格层级
                if not price_level:
                    del self.bids[price]

        # 处理未完全成交的部分
        traded_quantity = original_quantity - market_order.quantity

        if market_order.quantity > 0:
            self.logger.debug(
                f"市价单 {market_order.order_id} 立即成交模式完成：已成交 {traded_quantity}手，"
                f"剩余 {market_order.quantity}手，等待后续 T/C 撤单记录"
            )
            self.pending_exchange_cancels[market_order.order_id] = {
                "side": market_order.side,
                "price": 0.0,
                "quantity": market_order.quantity,
                "order_time": market_order.order_time,
            }
            # 剩余部分不再参与后续撮合，但撤单记录以交易所 T/C 为准
            if market_order.order_id in self.orders:
                del self.orders[market_order.order_id]
            return
        else:
            self.logger.debug(
                f"市价单 {market_order.order_id} 立即成交模式完成：已完全成交 {traded_quantity}手"
            )
            if market_order.order_id in self.orders:
                del self.orders[market_order.order_id]

    def _match_market_order_immediate(
        self, market_order: Order, original_quantity: int
    ):
        """
        市价单立即成交模式.

        按市场最优价格依次成交，能成交多少就成交多少
        不转挂限价单，剩余部分直接失效.
        """
        if market_order.side == Side.BUY:
            # 买入市价单：从最低价卖单开始依次成交
            ask_prices = list(self.asks.keys())

            self.logger.debug(f"市价买单 {market_order.order_id} 开始立即成交模式")

            for price in ask_prices:
                if market_order.quantity <= 0:
                    break

                price_level = self.asks[price]
                orders_to_remove = []

                self.logger.debug(f"开始成交价格档位 {price:.2f} 的卖单...")

                while market_order.quantity > 0 and price_level:
                    maker_order = price_level[0]

                    if maker_order.quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        continue

                    trade_quantity = min(market_order.quantity, maker_order.quantity)

                    if trade_quantity <= 0:
                        break

                    # 按当前价格档位成交
                    self._execute_trade(
                        maker_order, market_order, price, trade_quantity
                    )

                    # 更新订单数量
                    maker_order.quantity -= trade_quantity
                    market_order.quantity -= trade_quantity

                    if maker_order.quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        self.logger.debug(f"订单 {maker_order.order_id} 已完全成交。")

                # 清理完全成交的订单
                for order in orders_to_remove:
                    if order.order_id in self.orders:
                        del self.orders[order.order_id]

                # 如果价格层级为空，删除该价格层级
                if not price_level:
                    del self.asks[price]

        else:  # Side.SELL
            # 卖出市价单：从最高价买单开始依次成交
            bid_prices = list(reversed(self.bids.keys()))

            self.logger.debug(f"市价卖单 {market_order.order_id} 开始立即成交模式")

            for price in bid_prices:
                if market_order.quantity <= 0:
                    break

                price_level = self.bids[price]
                orders_to_remove = []

                self.logger.debug(f"开始成交价格档位 {price:.2f} 的买单...")

                while market_order.quantity > 0 and price_level:
                    maker_order = price_level[0]

                    if maker_order.quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        continue

                    trade_quantity = min(market_order.quantity, maker_order.quantity)

                    if trade_quantity <= 0:
                        break

                    # 按当前价格档位成交
                    self._execute_trade(
                        maker_order, market_order, price, trade_quantity
                    )

                    # 更新订单数量
                    maker_order.quantity -= trade_quantity
                    market_order.quantity -= trade_quantity

                    if maker_order.quantity <= 0:
                        orders_to_remove.append(maker_order)
                        price_level.popleft()
                        self.logger.debug(f"订单 {maker_order.order_id} 已完全成交。")

                # 清理完全成交的订单
                for order in orders_to_remove:
                    if order.order_id in self.orders:
                        del self.orders[order.order_id]

                # 如果价格层级为空，删除该价格层级
                if not price_level:
                    del self.bids[price]

        # 处理未完全成交的部分（立即成交模式不转挂限价单）
        traded_quantity = original_quantity - market_order.quantity

        if market_order.quantity > 0:
            self.logger.debug(
                f"市价单 {market_order.order_id} 立即成交模式完成：已成交 {traded_quantity}手，"
                f"剩余 {market_order.quantity}手 无法成交（市场流动性不足）"
            )
        else:
            self.logger.debug(
                f"市价单 {market_order.order_id} 立即成交模式完成：已完全成交 {traded_quantity}手"
            )

        # 无论是否完全成交，都从订单系统中移除（不转挂限价单）
        if market_order.order_id in self.orders:
            del self.orders[market_order.order_id]

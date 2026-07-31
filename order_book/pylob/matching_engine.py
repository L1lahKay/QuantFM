import logging
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from sortedcontainers import SortedDict

from pylob.data_types import Cancel, Order, Side, Trade, TradingPhase

logger = logging.getLogger(__name__)


class _SymbolLoggerAdapter(logging.LoggerAdapter):
    """在日志消息前自动添加 [symbol] 前缀."""

    def process(self, msg, kwargs):
        return f"[{self.extra['symbol']}] {msg}", kwargs


class MatchingEngine(ABC):
    """
    订单簿撮合引擎：状态管理 + 回放流程 + 撮合逻辑.

    子类需实现 4 个抽象方法以适配不同交易所的数据格式和阶段规则。
    """

    def __init__(self):
        # 使用 SortedDict 来自动维护价格排序
        self.bids = SortedDict()
        self.asks = SortedDict()

        # 使用哈希表快速通过订单ID查找订单
        self.orders = {}

        # 用于跟踪使用过的order_id，防止重复
        self.used_order_ids = set()
        self.next_auto_order_id = 1
        self.trades = []
        self.next_trade_id = 1  # 交易ID计数器
        self.cancels = []
        self.next_cancel_id = 1  # 取消ID计数器
        # 市价单等待后续交易所 T/C 正式落账
        self.pending_exchange_cancels = {}

        # 交易阶段控制
        self.trading_phase = TradingPhase.CALL_AUCTION

        # 新增：保存所有订单的初始信息（用于查询历史）
        self.order_history = {}  # order_id -> 初始订单信息

        # 初始化运行时状态，防止空数据时 finalize 报 AttributeError
        self.row_data_time = 0
        self.cont_count = 0
        self.first_continuous = True

        self.symbol: str | None = None
        self.logger: logging.Logger | logging.LoggerAdapter = logger

    # ------------------------------------------------------------------
    # 状态重置
    # ------------------------------------------------------------------

    def reset_market_state(self):
        """重置订单簿核心状态."""
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.orders = {}
        self.used_order_ids = set()
        self.next_auto_order_id = 1
        self.trades = []
        self.next_trade_id = 1
        self.cancels = []
        self.next_cancel_id = 1
        self.pending_exchange_cancels = {}
        self.order_history = {}

        self.trading_phase = TradingPhase.CALL_AUCTION
        self.row_data_time = 0
        self.cont_count = 0
        self.first_continuous = True

    # ------------------------------------------------------------------
    # Logger / symbol
    # ------------------------------------------------------------------

    def _update_logger_for_symbol(self, symbol: str):
        """用 LoggerAdapter 为所有日志自动添加 [symbol] 前缀."""
        self.symbol = symbol
        self.logger = _SymbolLoggerAdapter(logger, {"symbol": symbol})

    # ------------------------------------------------------------------
    # 数据设置
    # ------------------------------------------------------------------

    def set_trade_df(self, trade_df):
        """设置真实交易数据表."""
        self.trade_df = trade_df
        self.logger.debug(f"已设置真实交易数据表，包含{len(trade_df):,}条记录")

    def set_trading_phase(self, phase: TradingPhase):
        """设置交易阶段."""
        self.trading_phase = phase
        self.logger.debug(f"交易阶段切换为: {phase.value}")

    def _parse_side(self, side_str: str) -> Side:
        """解析买卖方向字符串."""
        # 处理字节字符串的字符串表示形式
        if side_str in ["b'B'", "B", "买入"]:
            return Side.BUY
        elif side_str in ["b'S'", "S", "卖出"]:
            return Side.SELL
        else:
            msg = f"未知的交易方向: {side_str}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # 抽象方法（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_market_data(self):
        """准备市场数据处理环境."""

    @abstractmethod
    def _process_single_record(
        self, row_data: np.ndarray, *, is_continuous: bool = False
    ):
        """处理单条记录."""

    @abstractmethod
    def _auto_detect_trading_phase(self, row_data):
        """基于时间自动检测并切换交易阶段."""

    @abstractmethod
    def finalize_trading_session(self):
        """结束交易会话，处理最后的pending trade."""

    # ------------------------------------------------------------------
    # 回放流程
    # ------------------------------------------------------------------

    def process_single_market_record(self, row_data):
        """
        处理单条市场数据记录（模拟实盘数据流）.

        Args:
            row_data: np.ndarray - 单条记录数据
        """
        trading_phase_old = self.trading_phase
        self._auto_detect_trading_phase(row_data)
        if (
            trading_phase_old == TradingPhase.CONTINUOUS
            and self.trading_phase == TradingPhase.CALL_AUCTION
        ):
            self.cont_count = self.process_num

        # 处理单条记录
        is_continuous = self.trading_phase == TradingPhase.CONTINUOUS
        self._process_single_record(row_data, is_continuous=is_continuous)

        # 如果是连续竞价第一条数据，标记为已处理
        if is_continuous and self.first_continuous:
            self.first_continuous = False

    def process_workflow(self, order_df):
        """Replay ``order_df`` row-by-row like live feed, then finalize session."""
        self.reset_market_state()
        self.process_num = 0

        total_rows = len(order_df)
        order_array = order_df.to_numpy()

        for row in range(total_rows):
            row_data = order_array[row]
            self.row_data = row_data
            self.row_data_time = row_data[self.column_indices["int_time"]]
            self.process_single_market_record(row_data)
            self.process_num += 1

            self.process_percent = self.process_num / total_rows * 100

            if (self.trading_phase == TradingPhase.CONTINUOUS) and (
                self.process_num % 10000 == 0
            ):
                self.logger.debug(
                    f"处理进度: {self.process_num:,}/{total_rows:,} ({self.process_percent:.2f}%)"
                )
                # 周期性压缩 tombstone，避免单向堆积时簿深度虚高、内存膨胀。
                self._compact_book()

        # 结束交易会话
        self.finalize_trading_session()

    # ------------------------------------------------------------------
    # 订单管理
    # ------------------------------------------------------------------

    def _finalize_pending_trade(self):
        """完成待处理的交易."""
        if self.current_trade_id in self.market_trades:
            trade_order = self.market_trades[self.current_trade_id]

            self.add_order(
                price=trade_order.price,
                quantity=trade_order.quantity,
                side=trade_order.side,
                order_id=self.current_trade_id,
                order_time=trade_order.order_time,
                order_type="0",
            )

            self.market_orders[self.current_trade_id] = trade_order

    def _add_new_order(self, row_data: np.ndarray, order_id: int):
        """添加新订单."""
        indices = self.column_indices
        order_time = row_data[indices["int_time"]]
        side = self._parse_side(str(row_data[indices["bsflag"]]))
        market_order = Order(
            order_id=order_id,
            side=side,
            price=int(row_data[indices["order_price"]]),
            quantity=int(row_data[indices["order_volume"]]),
            order_time=order_time,
            order_type="O",
        )

        self.market_orders[order_id] = market_order

        self.add_order(
            price=market_order.price,
            quantity=market_order.quantity,
            side=side,
            order_id=order_id,
            order_time=market_order.order_time,
            order_type="0",
        )

    def _handle_partial_trade(self, row_data: np.ndarray, order_id: int):
        """处理部分成交的订单."""
        indices = self.column_indices

        # 这笔order没完全成交，总报单量加上之前的trade
        if self.current_trade_id in self.market_trades:
            trade_order = self.market_trades[self.current_trade_id]
            trade_order.quantity += int(row_data[indices["order_volume"]])
            trade_order.price = int(row_data[indices["order_price"]])

            self.market_orders[order_id] = trade_order

            self.add_order(
                price=trade_order.price,
                quantity=trade_order.quantity,
                side=trade_order.side,
                order_id=order_id,
                order_time=trade_order.order_time,
                order_type="0",
            )

    def add_order(
        self,
        price: float,
        quantity: int,
        side: Side,
        order_type: str = "0",
        order_id: int | None = None,
        order_time=None,
    ):
        """
        添加新订单.

        Args:
            price: 订单价格
                  - 限价单：具体的限价价格
                  - 市价单：如果设置了 trade_df，此参数将被自动覆盖；
                    否则 0=价格锁定模式，其他值=立即成交模式
                  - 本地最优单：任意值（会被自动覆盖为最优价格）
            quantity: 订单数量
            side: 订单方向
            order_type: 订单类型，'0'表示限价单，'1'表示市价单，'U'表示本地最优单
            order_id: 自定义订单ID，如果不提供则自动生成
            order_time: 订单时间，int: 93000000 (int_time 格式)
        """
        # 验证订单类型
        if order_type not in ["0", "1", "U"]:
            self.logger.error(
                f"订单类型 '{order_type}' 无效，应为 '0'(限价单)、'1'(市价单) 或 'U'(本地最优单)"
            )
            return None

        # 处理不同订单类型的价格
        left_quantity = None  # 默认值，仅市价单走 _query_market_order_type 时被赋值
        if order_type == "1":
            # 市价单仅 SZ 支持（依赖 _query_market_order_type / _match_market_order）
            if not hasattr(self, "_query_market_order_type"):
                self.logger.error(
                    f"当前订单簿不支持市价单 (order_type='1'), 订单 {order_id} 已拒绝"
                )
                return None
            # 市价单：根据trade_df查询结果动态设置价格
            if order_id is not None:
                market_type, left_quantity = self._query_market_order_type(
                    order_id, side
                )
                if market_type == 0:
                    # 市转限模式
                    price = 0
                    side_desc = "买入" if side == Side.BUY else "卖出"
                    self.logger.debug(
                        f"基于trade_df分析，市价{side_desc}单 {order_id} 设置为价格锁定模式（市转限）"
                    )
                elif market_type == 1:
                    # 全吃模式
                    price = 1
                    side_desc = "买入" if side == Side.BUY else "卖出"
                    self.logger.debug(
                        f"基于trade_df分析，市价{side_desc}单 {order_id} 设置为立即成交模式（全吃）"
                    )
                elif market_type == -2:
                    # 部分成交后，市转撤
                    price = -2
                    side_desc = "买入" if side == Side.BUY else "卖出"
                    self.logger.debug(
                        f"基于trade_df分析，市价{side_desc}单 {order_id} 成交部分后市转撤"
                    )
                else:
                    # 未找到记录，市转撤
                    price = -1
                    side_desc = "买入" if side == Side.BUY else "卖出"
                    self.logger.debug(
                        f"未找到trade_df记录，市价{side_desc}单 {order_id} 为市转撤"
                    )
            else:
                # 自定义的报单，按照普通市价单逻辑
                price = 1
                side_desc = "买入" if side == Side.BUY else "卖出"
                self.logger.debug(
                    f"自定义的报单，市价{side_desc}单 {order_id} 设置为立即成交模式（全吃）"
                )

        elif order_type == "U":
            # 本方最优单：取同方向最优价，无量时用极端价格兜底（ref: https://github.com/fpga2u/AXOrderBook）
            optimal_price = self._get_local_optimal_price(side)
            if optimal_price is None:
                optimal_price = 0 if side == Side.BUY else 999_999_999
                self.logger.warning(
                    f"本方最优单 {order_id} 无本方价格，以 {optimal_price} 挂单保留"
                )
            if price != optimal_price:
                self.logger.debug(
                    f"本方最优单价格自动调整: {price:.2f} -> {optimal_price:.2f}"
                )
            price = optimal_price

        elif order_type == "0" and price <= 0:
            # 限价单价格验证
            self.logger.error(f"限价单价格必须大于0，当前价格: {price}")
            return None

        # 集合竞价期间不支持市价单
        if self.trading_phase == TradingPhase.CALL_AUCTION and order_type == "1":
            self.logger.warning("集合竞价期间不支持市价单")
            return None

        # 处理订单ID
        if order_id is None:
            # 自动生成ID
            while self.next_auto_order_id in self.used_order_ids:
                self.next_auto_order_id += 1
            final_order_id = self.next_auto_order_id
            self.next_auto_order_id += 1
        else:
            # 使用自定义ID
            if order_id in self.used_order_ids:
                self.logger.error(f"订单ID {order_id} 已存在！")
                return None
            final_order_id = order_id

        self.used_order_ids.add(final_order_id)

        order = Order(
            order_id=final_order_id,
            side=side,
            price=price,
            quantity=quantity,
            order_time=order_time,
            order_type=order_type,
        )
        self.orders[final_order_id] = order

        # 保存订单初始信息到历史记录
        self.order_history[final_order_id] = {
            "order_id": final_order_id,
            "initial_side": side,
            "initial_price": price,
            "initial_quantity": quantity,
            "initial_order_type": order_type,
            "order_time": order_time,
        }

        # 订单接收信息用debug级别
        self.logger.debug(
            f"接收到新订单: ID={order.order_id}, 类型={order.order_type}, 方向={order.side.value}, "
            f"价格={order.price}, 数量={order.quantity}, 时间={order.order_time}"
        )

        # 根据交易阶段决定处理方式
        if self.trading_phase == TradingPhase.CONTINUOUS:
            if order_type == "1":  # 市价单
                self._match_market_order(order, left_quantity)
            else:  # 限价单或本地最优单
                self._match_order(order)
                # 如果订单没有完全成交，则将其放入订单簿
                if order.quantity > 0:
                    order_type_desc = "本地最优单" if order_type == "U" else "限价单"
                    self.logger.debug(
                        f"{order_type_desc} {order.order_id} 未完全成交，剩余数量 {order.quantity}，已挂单。"
                    )
                    self._add_to_book(order)
        else:  # 集合竞价阶段
            self.logger.debug(f"集合竞价阶段，订单 {order.order_id} 已加入待撮合队列。")
            self._add_to_book(order)

        return final_order_id

    def _add_to_book(self, order: Order):
        """将订单添加到订单簿."""
        if order.side == Side.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = deque()
            self.bids[order.price].append(order)
        else:  # Side.SELL
            if order.price not in self.asks:
                self.asks[order.price] = deque()
            self.asks[order.price].append(order)

    def _purge_level_front(self, price_level: deque) -> None:
        """弹出队头已失效或已不在活动订单索引中的订单，摊还 O(1)."""
        while price_level:
            order = price_level[0]
            indexed = self.orders.get(order.order_id)
            if order.quantity > 0 and indexed is order:
                break
            price_level.popleft()
            if indexed is order:
                self.orders.pop(order.order_id, None)

    def _compact_book(self) -> None:
        """防御性清理无效订单，保证价格档只包含活动订单."""
        for book in (self.bids, self.asks):
            empty_prices = []
            for price, level in list(book.items()):
                if not level:
                    empty_prices.append(price)
                    continue
                live = deque(
                    order
                    for order in level
                    if order.quantity > 0 and self.orders.get(order.order_id) is order
                )
                if not live:
                    empty_prices.append(price)
                elif len(live) < len(level):
                    book[price] = live
            for price in empty_prices:
                del book[price]

    def _remove_from_book(self, order: Order) -> bool:
        """从订单所在价格档物理删除订单，并在档位为空时删除价格档."""
        book = self.bids if order.side == Side.BUY else self.asks
        price_level = book.get(order.price)
        if price_level is None:
            return False

        try:
            price_level.remove(order)
        except ValueError:
            return False

        if not price_level:
            del book[order.price]
        return True

    def cancel_order(
        self,
        order_id: int,
        cancel_time=None,
        cancel_quantity: int | None = None,
    ):
        """
        取消订单.

        Args:
            order_id: 要取消的订单ID
            cancel_time: 取消时间，int_time 格式，如 93000000
        """
        if order_id not in self.orders:
            pending = self.pending_exchange_cancels.pop(order_id, None)
            if pending is not None:
                record_qty = (
                    cancel_quantity
                    if cancel_quantity is not None
                    else pending["quantity"]
                )
                record_price = pending["price"]

                cancel_record = Cancel(
                    cancel_id=self.next_cancel_id,
                    order_id=order_id,
                    side=pending["side"],
                    price=record_price,
                    quantity=record_qty,
                    order_time=pending["order_time"],
                    cancel_time=cancel_time,
                )
                self.cancels.append(cancel_record)
                self.next_cancel_id += 1
                self.used_order_ids.discard(order_id)

                self.logger.debug(
                    f"订单 {order_id} 在 O 行已退出活动簿，T/C 到达后补记撤单。"
                )
                return True

            self.logger.warning(f"无法找到订单 ID: {order_id}")
            return False

        order_to_cancel = self.orders[order_id]

        self.logger.debug(
            f"取消订单请求: 订单ID={order_id}, 取消时间={cancel_time}, 交易阶段={self.trading_phase.value}"
        )

        record_qty = (
            cancel_quantity if cancel_quantity is not None else order_to_cancel.quantity
        )
        record_price = order_to_cancel.price
        side = order_to_cancel.side
        order_time = order_to_cancel.order_time
        order_type = order_to_cancel.order_type

        # 撤单必须同步更新 deque 与 orders。deque.remove 为 O(n)，但可保证中间
        # 订单撤销后同价位其余订单的 FIFO 顺序不变，也不会留下 tombstone。
        if not self._remove_from_book(order_to_cancel):
            # 修复已存在的不一致状态，避免 orders 继续引用一个不在簿上的订单。
            del self.orders[order_id]
            order_to_cancel.quantity = 0
            self.logger.error(
                f"订单 {order_id} 存在于活动索引但不在对应价格档，已清理异常索引"
            )
            return False

        order_to_cancel.quantity = 0

        self.pending_exchange_cancels.pop(order_id, None)
        cancel_record = Cancel(
            cancel_id=self.next_cancel_id,
            order_id=order_id,
            side=side,
            price=record_price,
            quantity=record_qty,
            order_time=order_time,
            cancel_time=cancel_time,
        )
        self.cancels.append(cancel_record)
        self.next_cancel_id += 1

        del self.orders[order_id]
        self.used_order_ids.discard(order_id)

        order_type_str = (
            "市价单"
            if order_type == "1"
            else ("本地最优单" if order_type == "U" else "限价单")
        )

        if self.trading_phase == TradingPhase.CALL_AUCTION:
            self.logger.debug(
                f"集合竞价阶段：{order_type_str} {order_id} 已成功撤销，将不参与后续撮合。"
            )
        else:
            self.logger.debug(f"连续竞价阶段：{order_type_str} {order_id} 已成功撤销。")

        self.logger.debug(
            f"原订单信息: 类型: {order_type_str}, 方向: {side.value}, "
            f"价格: {record_price:.2f}, 数量: {record_qty}, "
            f"挂单时间: {order_time}"
        )

        return True

    # ------------------------------------------------------------------
    # 集合竞价
    # ------------------------------------------------------------------

    def call_auction_match(self):
        """
        执行集合竞价撮合.

        找到能成交最大数量的价格进行撮合.
        """
        if self.trading_phase != TradingPhase.CALL_AUCTION:
            self.logger.warning("当前不在集合竞价阶段")
            return

        self.logger.debug("开始集合竞价撮合")

        # 找到集合竞价价格
        auction_price, max_volume = self._find_auction_price()

        if auction_price is None:
            self.logger.debug("集合竞价撮合完成：无法找到合适的撮合价格")
            return

        self.logger.debug(
            f"集合竞价价格: {auction_price:.2f}, 预期成交量: {max_volume}"
        )

        # 在集合竞价价格上撮合
        self._execute_call_auction(auction_price)

        self.logger.debug("集合竞价撮合完成")

    def _find_auction_price(self) -> tuple[float | None, int]:
        """
        寻找集合竞价价格.

        返回: (价格, 最大成交量).
        """
        if not self.bids or not self.asks:
            return None, 0

        # 获取所有可能的价格点
        all_prices = set()
        all_prices.update(self.bids.keys())
        all_prices.update(self.asks.keys())
        all_prices = sorted(all_prices)

        # Aggregate every order once, then update cumulative executable volume
        # while walking prices from low to high.  The previous implementation
        # rescanned the complete bid/ask book for every candidate price, which
        # is quadratic in the number of price levels and dominates active-stock
        # replays around the opening/closing auctions.
        buy_at_price = {
            price: sum(order.quantity for order in orders if order.quantity > 0)
            for price, orders in self.bids.items()
        }
        sell_at_price = {
            price: sum(order.quantity for order in orders if order.quantity > 0)
            for price, orders in self.asks.items()
        }
        buy_volume = sum(buy_at_price.values())
        sell_volume = 0
        max_volume = 0
        best_price = None

        self.logger.debug("集合竞价价格分析:")
        for price in all_prices:
            sell_volume += sell_at_price.get(price, 0)

            volume = min(buy_volume, sell_volume)
            self.logger.debug(
                f"价格 {price:.2f}: 买量={buy_volume}, 卖量={sell_volume}, 可成交={volume}"
            )

            if volume > max_volume:
                max_volume = volume
                best_price = price

            # At the next (higher) candidate price, bids at this price are no
            # longer executable.  Update after evaluation so p >= price keeps
            # exactly the original boundary semantics and tie-breaking.
            buy_volume -= buy_at_price.get(price, 0)

        if best_price is not None:
            self.logger.debug(
                f"选定集合竞价价格: {best_price:.2f}, 最大成交量: {max_volume}"
            )
        else:
            self.logger.debug("集合竞价未能撮合出开盘价")
        return best_price, max_volume

    def _execute_call_auction(self, auction_price: float):
        """在集合竞价价格上执行撮合."""
        self.logger.debug(f"开始在价格 {auction_price:.2f} 执行集合竞价撮合...")

        # 收集所有能在此价格成交的订单并按价格和时间排序
        eligible_buys = []
        eligible_sells = []

        # 收集买单（价格 >= auction_price），确保订单仍在订单簿中且未被撤销
        for price, orders in self.bids.items():
            if price >= auction_price:
                for order in list(orders):
                    if order.order_id in self.orders and order.quantity > 0:
                        eligible_buys.append(order)
                        self.logger.debug(
                            f"符合条件的买单: ID={order.order_id}, 价格={order.price:.2f}, 数量={order.quantity}"
                        )
                    else:
                        self.logger.debug(f"订单 {order.order_id} 已被撤销或无效，跳过")

        # 收集卖单（价格 <= auction_price），确保订单仍在订单簿中且未被撤销
        for price, orders in self.asks.items():
            if price <= auction_price:
                for order in list(orders):
                    if order.order_id in self.orders and order.quantity > 0:
                        eligible_sells.append(order)
                        self.logger.debug(
                            f"符合条件的卖单: ID={order.order_id}, 价格={order.price:.2f}, 数量={order.quantity}"
                        )
                    else:
                        self.logger.debug(f"订单 {order.order_id} 已被撤销或无效，跳过")

        # 找到所有参与撮合的订单中的最晚时间作为成交时间
        all_eligible_orders = eligible_buys + eligible_sells
        if not all_eligible_orders:
            self.logger.warning("没有符合条件的有效订单进行撮合")
            return

        latest_order = max(all_eligible_orders, key=lambda x: x.order_time)
        auction_trade_time = latest_order.order_time
        self.logger.debug(
            f"集合竞价统一成交时间={auction_trade_time} (来源订单ID={latest_order.order_id})"
        )

        # 买单：价格从高到低，同价位按时间从早到晚
        eligible_buys.sort(key=lambda x: (-x.price, x.order_id))
        # 卖单：价格从低到高，同价位按时间从早到晚
        eligible_sells.sort(key=lambda x: (x.price, x.order_id))

        self.logger.debug(
            f"符合条件的有效买单数量: {len(eligible_buys)}, 卖单数量: {len(eligible_sells)}"
        )

        # 按排序后的顺序执行撮合
        buy_idx, sell_idx = 0, 0
        total_traded = 0

        while buy_idx < len(eligible_buys) and sell_idx < len(eligible_sells):
            buy_order = eligible_buys[buy_idx]
            sell_order = eligible_sells[sell_idx]

            if buy_order.order_id not in self.orders or buy_order.quantity <= 0:
                self.logger.debug(
                    f"买单 {buy_order.order_id} 在撮合过程中变为无效，跳过"
                )
                buy_idx += 1
                continue
            if sell_order.order_id not in self.orders or sell_order.quantity <= 0:
                self.logger.debug(
                    f"卖单 {sell_order.order_id} 在撮合过程中变为无效，跳过"
                )
                sell_idx += 1
                continue

            trade_quantity = min(buy_order.quantity, sell_order.quantity)

            if trade_quantity <= 0:
                if buy_order.quantity <= 0:
                    buy_idx += 1
                if sell_order.quantity <= 0:
                    sell_idx += 1
                continue

            self._execute_trade(
                sell_order, buy_order, auction_price, trade_quantity, auction_trade_time
            )
            total_traded += trade_quantity

            buy_order.quantity -= trade_quantity
            sell_order.quantity -= trade_quantity

            if buy_order.quantity <= 0:
                buy_idx += 1
            if sell_order.quantity <= 0:
                sell_idx += 1

        self.logger.debug(f"集合竞价撮合完成，总成交量: {total_traded}")

        # 清理完全成交的订单
        self._cleanup_filled_orders()

    def _cleanup_filled_orders(self):
        """清理已完全成交的订单."""
        orders_to_remove = []

        for order_id, order in self.orders.items():
            if order.quantity <= 0:
                orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            order = self.orders[order_id]
            self.logger.debug(f"订单 {order_id} 已完全成交。")

            if order.side == Side.BUY:
                price_level = self.bids.get(order.price)
                if price_level and order in price_level:
                    price_level.remove(order)
                    if not price_level:
                        del self.bids[order.price]
            else:
                price_level = self.asks.get(order.price)
                if price_level and order in price_level:
                    price_level.remove(order)
                    if not price_level:
                        del self.asks[order.price]

            del self.orders[order_id]

    # ------------------------------------------------------------------
    # 连续竞价撮合
    # ------------------------------------------------------------------

    def _match_order(self, taker_order: Order):
        """核心撮合逻辑（连续竞价）- 处理限价单."""
        if taker_order.side == Side.BUY:
            # SortedDict.peekitem(0) is O(log P) and avoids copying all P price
            # keys for every incoming order.  On liquid symbols the old
            # ``list(self.asks)`` made a non-crossing order O(P) before doing
            # any useful work.
            while taker_order.quantity > 0 and self.asks:
                best_price, price_level = self.asks.peekitem(0)
                self._purge_level_front(price_level)
                if not price_level:
                    del self.asks[best_price]
                    continue
                if taker_order.price < best_price:
                    break

                while taker_order.quantity > 0 and price_level:
                    maker_order = price_level[0]

                    if maker_order.quantity <= 0:
                        price_level.popleft()
                        continue

                    trade_quantity = min(taker_order.quantity, maker_order.quantity)

                    if trade_quantity <= 0:
                        break

                    self._execute_trade(
                        maker_order,
                        taker_order,
                        maker_order.price,
                        trade_quantity,
                        None,
                    )

                    maker_order.quantity -= trade_quantity
                    taker_order.quantity -= trade_quantity

                    if maker_order.quantity <= 0:
                        price_level.popleft()
                        self.logger.debug(f"订单 {maker_order.order_id} 已完全成交。")
                        self.orders.pop(maker_order.order_id, None)

                if not price_level:
                    del self.asks[best_price]

        else:  # Side.SELL
            while taker_order.quantity > 0 and self.bids:
                best_price, price_level = self.bids.peekitem(-1)
                self._purge_level_front(price_level)
                if not price_level:
                    del self.bids[best_price]
                    continue
                if taker_order.price > best_price:
                    break

                while taker_order.quantity > 0 and price_level:
                    maker_order = price_level[0]

                    if maker_order.quantity <= 0:
                        price_level.popleft()
                        continue

                    trade_quantity = min(taker_order.quantity, maker_order.quantity)

                    if trade_quantity <= 0:
                        break

                    self._execute_trade(
                        maker_order,
                        taker_order,
                        maker_order.price,
                        trade_quantity,
                        None,
                    )

                    maker_order.quantity -= trade_quantity
                    taker_order.quantity -= trade_quantity

                    if maker_order.quantity <= 0:
                        price_level.popleft()
                        self.logger.debug(f"订单 {maker_order.order_id} 已完全成交。")
                        self.orders.pop(maker_order.order_id, None)

                if not price_level:
                    del self.bids[best_price]

        if taker_order.quantity <= 0:
            self.logger.debug(f"订单 {taker_order.order_id} 已完全成交。")
            if taker_order.order_id in self.orders:
                del self.orders[taker_order.order_id]

    def _execute_trade(
        self,
        maker_order: Order,
        taker_order: Order,
        price: float,
        quantity: int,
        trade_time_override: int | None = None,
    ):
        """执行交易并记录（trade_time 统一存 int_time）."""
        if trade_time_override is not None:
            trade_time = int(trade_time_override)
        elif taker_order.order_time is not None:
            trade_time = int(taker_order.order_time)
        else:
            trade_time = 0

        if taker_order.side == Side.BUY:
            buy_order_id = taker_order.order_id
            sell_order_id = maker_order.order_id
        else:
            buy_order_id = maker_order.order_id
            sell_order_id = taker_order.order_id

        trade = Trade(
            trade_id=self.next_trade_id,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            price=price,
            quantity=quantity,
            trade_time=trade_time,
        )
        self.trades.append(trade)
        self.next_trade_id += 1

        self.logger.debug(
            f"撮合成功! 价格={trade.price:.2f}, 数量={trade.quantity}, "
            f"买单ID={trade.buy_order_id}, 卖单ID={trade.sell_order_id}, "
            f"成交时间={trade.trade_time}"
        )

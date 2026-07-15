from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


class ResultMixin:
    """
    Mixin: print, export, and compare order book and trade outputs as tables.

    Requires host class to provide: bids, asks, orders, trades, cancels,
    order_history, logger, trading_phase, matching_date (optional),
    以及 MatchingEngine 的回放方法.
    """

    def process_workflow_with_book(self, order_df, depth=None):
        """
        回放订单流并在每条连续竞价记录后生成订单簿快照.

        Parameters
        ----------
        order_df
            订单数据（pandas DataFrame）
        depth
            快照深度（档位数），None 表示全部
        """
        from pylob.data_types import TradingPhase

        self.process_num = 0
        self.all_orderbook = []

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

            if self.trading_phase == TradingPhase.CONTINUOUS:
                df = self.get_full_order_book_dataframe(depth=depth)
                df["serial"] = row_data[self.column_indices["serial"]]
                df["int_time"] = self.row_data_time

                self.all_orderbook.append(df)

        self.finalize_trading_session()

        all_orderbook = pd.concat(self.all_orderbook)

        return all_orderbook

    def format_trade_time_with_date(self, trade_time, matching_date=None) -> str:
        """
        将交易所成交时间 int 转为带毫秒字符串，可选拼接 matching_date.

        例如 int(91500020) 转为 '09:15:00.020'；有日期时得到 'YYYY-MM-DD HH:MM:SS.mmm'.
        """
        # 优先使用传入 matching_date，其次用对象属性 matching_date
        if matching_date is None:
            matching_date = getattr(self, "matching_date", None)

        t = str(int(trade_time)).zfill(9)
        hh = t[0:2]
        mm = t[2:4]
        ss = t[4:6]
        ms = t[6:9]
        time_str = f"{hh}:{mm}:{ss}.{ms}"

        return f"{matching_date} {time_str}"

    def print_order_book(self, depth=None):
        """打印当前订单簿的快照."""
        # 使用INFO级别输出订单簿快照
        lines = []
        lines.append("===================== 订单簿快照 =====================")
        lines.append(f"当前阶段: {self.trading_phase.value}")
        lines.append(f"{'买单 (Bids)':^21}|{'卖单 (Asks)':^21}")
        lines.append(f"{'数量':>10}{'价格':>10} | {'价格':>10}{'数量':>10}")
        lines.append("------------------------------------------------------")

        # 买单：选择最高的n档价格（SortedDict 升序，reversed 取降序）
        all_bid_prices = [
            p for p in reversed(self.bids) if any(o.quantity > 0 for o in self.bids[p])
        ]
        if depth is None:
            bid_prices = all_bid_prices.copy()
        else:
            bid_prices = all_bid_prices[:depth]

        # 卖单：选择最低的n档价格（SortedDict 已升序）
        all_ask_prices = [
            p for p in self.asks if any(o.quantity > 0 for o in self.asks[p])
        ]
        if depth is None:
            ask_prices = all_ask_prices.copy()
        else:
            ask_prices = all_ask_prices[:depth]

        # 卖单区域：从最高的卖价到最低的卖价（最低价靠近分界线）
        ask_display = []
        for price in reversed(ask_prices):  # 倒序，让最低价在底部
            total_quantity = sum(
                order.quantity for order in self.asks[price] if order.quantity > 0
            )
            if total_quantity > 0:
                ask_display.append((price, total_quantity))

        # 显示卖单部分
        for price, total_quantity in ask_display:
            lines.append(f"{' ' * 21}| {price / 10000:<10.2f}{total_quantity:<10}")

        # 显示分界线
        lines.append("------------------------------------------------------")

        # 买单区域：从最高的买价到最低的买价（最高价靠近分界线）
        for price in bid_prices:
            total_quantity = sum(
                order.quantity for order in self.bids[price] if order.quantity > 0
            )
            if total_quantity > 0:
                lines.append(f"{total_quantity:>10}{price / 10000:>10.2f} |")

        lines.append("======================================================")

        for line in lines:
            self.logger.info(line)

    def get_full_order_book_dataframe(self, depth=None):
        """获取当前订单簿的DataFrame格式数据."""
        # 买单：选择最高的n档价格（SortedDict 升序，reversed 取降序）
        all_bid_prices = [
            p for p in reversed(self.bids) if any(o.quantity > 0 for o in self.bids[p])
        ]
        if depth is None:
            bid_prices = all_bid_prices.copy()
        else:
            bid_prices = all_bid_prices[:depth]

        # 卖单：选择最低的n档价格（SortedDict 已升序）
        all_ask_prices = [
            p for p in self.asks if any(o.quantity > 0 for o in self.asks[p])
        ]
        if depth is None:
            ask_prices = all_ask_prices.copy()
        else:
            ask_prices = all_ask_prices[:depth]

        # 准备DataFrame数据
        av_array = []
        ap_array = []
        bv_array = []
        bp_array = []

        # 添加买单数据
        for _i, price in enumerate(bid_prices):
            total_quantity = sum(
                order.quantity for order in self.bids[price] if order.quantity > 0
            )
            if total_quantity > 0:
                bv_array.append(total_quantity)
                bp_array.append(price / 10000)

        # 添加卖单数据
        for _i, price in enumerate(ask_prices):
            total_quantity = sum(
                order.quantity for order in self.asks[price] if order.quantity > 0
            )
            if total_quantity > 0:
                av_array.append(total_quantity)
                ap_array.append(price / 10000)

        # 创建DataFrame
        data = {
            "av_array": av_array,  # list，不需要[]
            "ap_array": ap_array,  # list，不需要[]
            "bv_array": bv_array,  # list，不需要[]
            "bp_array": bp_array,  # list，不需要[]
        }

        # 转换为DataFrame（注意需要用[data]包装）
        df = pd.DataFrame([data])

        return df

    def print_trades(self):
        """打印成交记录."""
        lines = []
        lines.append("======================= 成交记录 =======================")
        if not self.trades:
            lines.append("暂无成交记录。")
        else:
            lines.append(
                f"{'成交ID':>8} {'成交时间':>29} {'买单ID':>8} {'卖单ID':>8} {'成交价':>10} {'成交量':>10}"
            )
            lines.append("-" * 85)
            for trade in self.trades:
                lines.append(
                    f"{trade.trade_id:>8} {trade.trade_time:>29} {trade.buy_order_id:>8} "
                    f"{trade.sell_order_id:>8} {trade.price:>10.2f} {trade.quantity:>10}"
                )
        lines.append("======================================================")

        for line in lines:
            self.logger.info(line)

    def get_trades_table(self):
        """获取成交记录的表格数据."""
        if not self.trades:
            return []

        table_data = []
        for trade in self.trades:
            table_data.append(
                {
                    "成交ID": trade.trade_id,
                    "成交时间": trade.trade_time,
                    "买单ID": trade.buy_order_id,
                    "卖单ID": trade.sell_order_id,
                    "成交价": trade.price,
                    "成交量": trade.quantity,
                }
            )
        return table_data

    def export_trades_csv(self, filename="trades.csv"):
        """导出成交记录到CSV文件."""
        import csv

        if not self.trades:
            self.logger.warning("没有成交记录可导出")
            return

        with Path(filename).open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["成交ID", "成交时间", "买单ID", "卖单ID", "成交价", "成交量"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for trade in self.trades:
                writer.writerow(
                    {
                        "成交ID": trade.trade_id,
                        "成交时间": self.format_trade_time_with_date(trade.trade_time),
                        "买单ID": trade.buy_order_id,
                        "卖单ID": trade.sell_order_id,
                        "成交价": trade.price,
                        "成交量": trade.quantity,
                    }
                )
        self.logger.info(f"成交记录已导出到 {filename}")

    def print_cancels(self):
        """打印取消记录."""
        lines = []
        lines.append("======================= 取消记录 =======================")
        if not self.cancels:
            lines.append("暂无取消记录。")
        else:
            lines.append(
                f"{'取消ID':>8} {'订单ID':>8} {'方向':>6} {'价格':>10} {'数量':>10} {'挂单时间':>29} {'取消时间':>29}"
            )
            lines.append("-" * 110)
            for cancel in self.cancels:
                lines.append(
                    f"{cancel.cancel_id:>8} {cancel.order_id:>8} {cancel.side.value:>6} "
                    f"{cancel.price:>10.2f} {cancel.quantity:>10} {cancel.order_time:>29} {cancel.cancel_time:>29}"
                )
        lines.append("======================================================")

        for line in lines:
            self.logger.info(line)

    def get_cancels_table(self):
        """获取取消记录的表格数据."""
        if not self.cancels:
            return []

        table_data = []
        for cancel in self.cancels:
            table_data.append(
                {
                    "取消ID": cancel.cancel_id,
                    "订单ID": cancel.order_id,
                    "方向": cancel.side.value,
                    "价格": cancel.price,
                    "数量": cancel.quantity,
                    "挂单时间": cancel.order_time,
                    "取消时间": cancel.cancel_time,
                }
            )
        return table_data

    def export_cancels_csv(self, filename="cancels.csv"):
        """导出取消记录到CSV文件."""
        import csv

        if not self.cancels:
            self.logger.warning("没有取消记录可导出")
            return

        with Path(filename).open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "取消ID",
                "订单ID",
                "方向",
                "价格",
                "数量",
                "挂单时间",
                "取消时间",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for cancel in self.cancels:
                writer.writerow(
                    {
                        "取消ID": cancel.cancel_id,
                        "订单ID": cancel.order_id,
                        "方向": cancel.side.value,
                        "价格": cancel.price,
                        "数量": cancel.quantity,
                        "挂单时间": self.format_trade_time_with_date(cancel.order_time),
                        "取消时间": self.format_trade_time_with_date(
                            cancel.cancel_time
                        ),
                    }
                )
        self.logger.info(f"取消记录已导出到 {filename}")

    def get_order_details(self, order_id: int):
        """
        根据订单ID获取订单详细信息.

        Args:
            order_id: 要查询的订单ID

        Returns
        -------
            dict: 包含以下信息的字典
            - 'trades': 该订单的成交记录列表
            - 'order_summary': 包含初始信息、当前状态、剩余数量等的综合信息
        """
        if order_id not in self.order_history:
            return {
                "error": f"订单ID {order_id} 不存在",
                "trades": [],
                "order_summary": {},
            }

        # 获取初始订单信息
        initial_info = self.order_history[order_id].copy()

        # 查找该订单的所有成交记录
        order_trades = []
        for trade in self.trades:
            if trade.buy_order_id == order_id or trade.sell_order_id == order_id:
                order_trades.append(
                    {
                        "成交ID": trade.trade_id,
                        "成交时间": trade.trade_time,
                        "买单ID": trade.buy_order_id,
                        "卖单ID": trade.sell_order_id,
                        "成交价": trade.price,
                        "成交量": trade.quantity,
                        "订单角色": "买方"
                        if trade.buy_order_id == order_id
                        else "卖方",
                    }
                )

        # 计算已成交数量
        traded_quantity = sum(trade_record["成交量"] for trade_record in order_trades)

        # 确定当前状态和剩余数量
        if order_id in self.orders:
            # 订单仍在系统中
            current_order = self.orders[order_id]
            remaining_quantity = current_order.quantity
            if traded_quantity > 0:
                status = "部分成交"
            else:
                status = "未成交"
        else:
            # 订单不在系统中，检查是否被取消
            is_cancelled = any(cancel.order_id == order_id for cancel in self.cancels)
            if is_cancelled:
                # 计算剩余数量（初始数量 - 已成交数量）
                remaining_quantity = initial_info["initial_quantity"] - traded_quantity
                status = "已取消"
            else:
                # 完全成交
                remaining_quantity = 0
                status = "完全成交"

        # 合并所有信息到order_summary中
        order_summary = {
            # 初始订单信息
            "order_id": order_id,
            "initial_side": initial_info["initial_side"],
            "initial_price": initial_info["initial_price"],
            "initial_quantity": initial_info["initial_quantity"],
            "initial_order_type": initial_info["initial_order_type"],
            "order_time": initial_info["order_time"],
            # 当前状态信息
            "status": status,
            "traded_quantity": traded_quantity,
            "remaining_quantity": remaining_quantity,
            # 成交概要
            "total_trades": len(order_trades),
            "avg_trade_price": sum(
                trade["成交价"] * trade["成交量"] for trade in order_trades
            )
            / traded_quantity
            if traded_quantity > 0
            else 0,
        }

        return {"trades": order_trades, "order_summary": order_summary}

    def get_order_details_dataframe(self, order_id: int):
        """获取订单详情并返回 DataFrame 格式的成交记录."""
        details = self.get_order_details(order_id)
        if "error" in details:
            self.logger.error(details["error"])
            return None, details

        if details["trades"]:
            trades_df = pd.DataFrame(details["trades"])
        else:
            trades_df = pd.DataFrame(
                columns=[
                    "成交ID",
                    "成交时间",
                    "买单ID",
                    "卖单ID",
                    "成交价",
                    "成交量",
                    "订单角色",
                ]
            )

        return trades_df, details

    def print_order_summary(self, order_id: int):
        """打印订单详细摘要信息."""
        details = self.get_order_details(order_id)

        if "error" in details:
            self.logger.error(details["error"])
            return

        summary = details["order_summary"]

        lines = []
        lines.append("=" * 60)
        lines.append(f"订单详细信息 - ID: {order_id}")
        lines.append("=" * 60)

        # 初始订单信息
        lines.append("初始订单信息:")
        price_str = f"{summary['initial_price']:.2f}"

        lines.append(f"  方向: {summary['initial_side']}")
        lines.append(f"  价格: {price_str}")
        lines.append(f"  数量: {summary['initial_quantity']}")
        lines.append(f"  挂单时间: {summary['order_time']}")

        # 当前状态
        lines.append(f"\n当前状态: {summary['status']}")
        lines.append(f"已成交数量: {summary['traded_quantity']}")
        lines.append(f"剩余数量: {summary['remaining_quantity']}")
        if summary["avg_trade_price"] > 0:
            lines.append(f"平均成交价: {summary['avg_trade_price']:.2f}")

        # 成交记录
        if details["trades"]:
            lines.append(f"\n成交记录 (共{summary['total_trades']}笔):")
            lines.append("-" * 80)
            lines.append(
                f"{'成交ID':>8} {'成交时间':>25} {'成交价':>10} {'成交量':>10} {'角色':>8}"
            )
            lines.append("-" * 80)
            for trade in details["trades"]:
                lines.append(
                    f"{trade['成交ID']:>8} {trade['成交时间']:>25} {trade['成交价']:>10.2f} "
                    f"{trade['成交量']:>10} {trade['订单角色']:>8}"
                )
        else:
            lines.append("\n暂无成交记录")

        lines.append("=" * 60)

        for line in lines:
            self.logger.info(line)

    def export_all_records_csv(
        self, trades_file="trades.csv", cancels_file="cancels.csv"
    ):
        """同时导出交易记录和取消记录."""
        self.export_trades_csv(trades_file)
        self.export_cancels_csv(cancels_file)
        self.logger.info("所有记录已导出完成")

    # 对比结果
    def compare_df(
        self,
        order_df,
        *,
        strict_mode: bool = False,
        price_atol: float = 1e-8,
        cut_time: int | None = None,
    ):
        """Compare simulated trades against reference rows in ``order_df``."""
        self.logger.debug("开始对比成交数据")
        tc = "int_time"
        trade_real = order_df[
            (order_df["type"] == "T") & (order_df["trade_type"] != "C")
        ].copy()
        trade_df = pd.DataFrame(self.get_trades_table())
        if cut_time is not None:
            trade_real[tc] = pd.to_numeric(trade_real[tc], errors="coerce")
            trade_real = trade_real[trade_real[tc] < int(cut_time)].copy()
        # 如果没有任何成交，都可以快速返回
        if trade_real.empty and trade_df.empty:
            self.logger.debug("[OK] 交易数据无误（双方都无成交）")
            return True

        # strict_mode：严格断言
        if strict_mode:
            left = trade_real.rename(
                columns={
                    "buy_id": "buy_id",
                    "sell_id": "sell_id",
                    "trade_price": "price",
                    "trade_volume": "qty",
                    tc: "time",
                }
            ).copy()
            left_cols = ["buy_id", "sell_id", "price", "qty", "time"]
            right = trade_df.rename(
                columns={
                    "买单ID": "buy_id",
                    "卖单ID": "sell_id",
                    "成交价": "price",
                    "成交量": "qty",
                    "成交时间": "time",
                }
            ).copy()
            right_cols = ["buy_id", "sell_id", "price", "qty", "time"]
            left = left[left_cols].copy()
            right = right[right_cols].copy()

            for c in ["buy_id", "sell_id", "qty"]:
                left[c] = pd.to_numeric(left[c], errors="coerce").astype("Int64")
                right[c] = pd.to_numeric(right[c], errors="coerce").astype("Int64")

            left["price"] = pd.to_numeric(left["price"], errors="coerce")

            right["price"] = pd.to_numeric(right["price"], errors="coerce")

            left["time"] = left["time"].astype(str)

            right["time"] = right["time"].astype(str)

            # 排序对齐：用 buy/sell + qty + price + time（如果有）做排序键
            sort_keys = ["buy_id", "sell_id", "qty", "price", "time"]
            left_sorted = left.sort_values(sort_keys).reset_index(drop=True)
            right_sorted = right.sort_values(sort_keys).reset_index(drop=True)
            self._strict_left = left_sorted
            self._strict_right = right_sorted
            # 严格断言：允许浮点误差（价格）
            assert_frame_equal(
                left_sorted,
                right_sorted,
                check_dtype=False,  # dtype 不同不算错（更稳）
                check_exact=False,  # 浮点不要求 bit-exact
                atol=price_atol,
                rtol=0.0,
            )

            self.logger.debug("[OK][STRICT] 交易数据严格一致")
            return True

        # ----------------------------
        # 非 strict：保持你原来的容错逻辑
        # ----------------------------
        compare_df = pd.merge(
            trade_real,
            trade_df,
            left_on=["buy_id", "sell_id"],
            right_on=["买单ID", "卖单ID"],
            how="outer",
        ).sort_values("serial")

        # 检查是否有差异
        missing_real = compare_df[compare_df["trade_volume"].isnull()]
        missing_my = compare_df[compare_df["成交量"].isnull()]

        if len(missing_real) == 0 and len(missing_my) == 0:
            self.logger.debug("[OK] 交易数据无误")
            return True

        # --------------------------------------------------
        # 有差异：输出详细诊断
        # --------------------------------------------------
        n_extra = len(missing_real)  # 模拟多出的
        n_missing = len(missing_my)  # 模拟缺失的
        n_real = len(trade_real)
        n_sim = len(trade_df)

        self.logger.error(
            f"[MISMATCH] 真实 {n_real} 笔 vs 模拟 {n_sim} 笔 | "
            f"模拟多出 {n_extra} 笔, 模拟缺失 {n_missing} 笔"
        )

        # --- 定位第一个出错点 ---
        first_serial = None
        first_time = None
        first_type = None
        first_detail = ""

        if n_missing > 0:
            row = missing_my.sort_values("serial").iloc[0]
            s = int(row["serial"]) if pd.notna(row.get("serial")) else None
            t = row.get("int_time")
            if first_serial is None or (s is not None and s < first_serial):
                first_serial = s
                first_time = t
                first_type = "模拟缺失"
                first_detail = (
                    f"buy_id={int(row['buy_id'])}, sell_id={int(row['sell_id'])}, "
                    f"price={row['trade_price']}, qty={int(row['trade_volume'])}"
                )

        if n_extra > 0:
            row = missing_real.sort_values("成交ID").iloc[0]
            if first_serial is None:
                first_type = "模拟多出"
                first_detail = (
                    f"buy_id={int(row['买单ID'])}, sell_id={int(row['卖单ID'])}, "
                    f"price={row['成交价']}, qty={int(row['成交量'])}"
                )

        # 计算第一个错误前连续匹配了多少笔
        if first_serial is not None:
            n_matched_before = len(
                compare_df[
                    (compare_df["serial"] < first_serial)
                    & compare_df["trade_volume"].notna()
                    & compare_df["成交量"].notna()
                ]
            )
            self.logger.error(
                f"[FIRST MISMATCH] serial={first_serial}, int_time={first_time} | "
                f"类型: {first_type}"
            )
            self.logger.error(f"  详情: {first_detail}")
            self.logger.error(f"  在此之前连续匹配了 {n_matched_before} 笔成交")
        elif first_type:
            self.logger.error(f"[FIRST MISMATCH] 类型: {first_type}")
            self.logger.error(f"  详情: {first_detail}")

        return False

    def mismatch_report(
        self,
        order_df,
        *,
        cut_time: int | None = None,
        time_bucket_seconds: int = 60,
    ) -> pd.DataFrame | None:
        """
        生成丢数据分布报告.

        Returns
        -------
        pd.DataFrame | None
            按时间桶统计的不匹配分布，列包含:
            time_bucket, n_real, n_sim, n_missing, n_extra
            如果完全匹配则返回 None。
        """
        tc = "int_time"
        trade_real = order_df[
            (order_df["type"] == "T") & (order_df["trade_type"] != "C")
        ].copy()
        trade_df = pd.DataFrame(self.get_trades_table())

        if cut_time is not None:
            trade_real[tc] = pd.to_numeric(trade_real[tc], errors="coerce")
            trade_real = trade_real[trade_real[tc] < int(cut_time)].copy()

        if trade_real.empty and trade_df.empty:
            return None

        merged = pd.merge(
            trade_real,
            trade_df,
            left_on=["buy_id", "sell_id"],
            right_on=["买单ID", "卖单ID"],
            how="outer",
        )

        missing_sim = merged["成交量"].isnull()  # 模拟缺失
        extra_sim = merged["trade_volume"].isnull()  # 模拟多出

        if not missing_sim.any() and not extra_sim.any():
            return None

        # 统一时间列：优先用真实数据的 int_time，模拟多出的用成交时间
        merged["_time"] = merged[tc]
        mask_no_time = merged["_time"].isna()
        if "成交时间" in merged.columns:
            merged.loc[mask_no_time, "_time"] = merged.loc[mask_no_time, "成交时间"]
        merged["_time"] = pd.to_numeric(merged["_time"], errors="coerce")

        # 把 int_time (如 93001000) 转为 HHMMSS 再分桶
        # int_time 格式: HHMMSSmmm，取前 6 位 = HHMMSS
        merged["_hhmmss"] = (merged["_time"] // 1000).astype("Int64")
        merged["_hour"] = merged["_hhmmss"] // 10000
        merged["_min"] = (merged["_hhmmss"] % 10000) // 100
        merged["_sec"] = merged["_hhmmss"] % 100
        merged["_total_sec"] = (
            merged["_hour"] * 3600 + merged["_min"] * 60 + merged["_sec"]
        )
        merged["_bucket"] = (
            merged["_total_sec"] // time_bucket_seconds
        ) * time_bucket_seconds

        # 转回 HH:MM:SS 格式做展示
        def sec_to_hms(s):
            if pd.isna(s):
                return "unknown"
            s = int(s)
            return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

        merged["_is_missing"] = missing_sim.astype(int)
        merged["_is_extra"] = extra_sim.astype(int)
        merged["_is_matched"] = ((~missing_sim) & (~extra_sim)).astype(int)

        report = (
            merged.groupby("_bucket", dropna=False)
            .agg(
                n_real=("_is_matched", "sum"),
                n_missing=("_is_missing", "sum"),
                n_extra=("_is_extra", "sum"),
            )
            .reset_index()
        )
        report["n_real"] = (
            report["n_real"] + report["n_missing"]
        )  # 真实总数 = 匹配 + 缺失
        report["time_bucket"] = report["_bucket"].apply(sec_to_hms)
        report = report[["time_bucket", "n_real", "n_missing", "n_extra"]]
        report = report.sort_values("time_bucket").reset_index(drop=True)

        # 输出摘要
        total_missing = report["n_missing"].sum()
        total_extra = report["n_extra"].sum()
        self.logger.info(
            f"[MISMATCH REPORT] 模拟缺失 {total_missing} 笔, 模拟多出 {total_extra} 笔"
        )
        # 只显示有问题的时间桶
        problem_rows = report[(report["n_missing"] > 0) | (report["n_extra"] > 0)]
        if not problem_rows.empty:
            self.logger.info("有问题的时间段:")
            for _, row in problem_rows.iterrows():
                self.logger.info(
                    f"  {row['time_bucket']}: "
                    f"真实={int(row['n_real'])}, "
                    f"缺失={int(row['n_missing'])}, "
                    f"多出={int(row['n_extra'])}"
                )

        return report

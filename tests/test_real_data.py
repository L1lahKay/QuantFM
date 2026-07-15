import os
import sys
from pathlib import Path

import polars as pl
import pytest

# 确保能导入 pylob 包
# 如果你的项目结构不同，请调整这里的路径
sys.path.append(str(Path.cwd()))

from pylob.orderbook_builder_sh import OrderBookSH
from pylob.orderbook_builder_sz import OrderBookSZ

# ==========================================
# 用户配置区域 (请修改这里)
# ==========================================

# 设置你的真实数据文件路径 (支持 .csv 或 .parquet)
# 示例: r"D:\data\600000_trade.csv"
REAL_DATA_CONFIG = {
    "symbol": "600000",  # 股票代码
    "market": "SH",  # 市场: "SH" 或 "SZ"
    # Override via PYLOB_REAL_TRADE_PATH / PYLOB_REAL_ORDER_PATH; skipped if missing.
    "trade_file": os.getenv(
        "PYLOB_REAL_TRADE_PATH",
        "tests/fixtures/SH600000_trade.parquet",
    ),
    "order_file": os.getenv(
        "PYLOB_REAL_ORDER_PATH",
        "tests/fixtures/SH600000_order.parquet",
    ),
    "cut_time": 150000000,  # 例如 935000000 表示 9:35
}

# ==========================================
# 辅助验证逻辑
# ==========================================


def verify_cancels_consistency(lob, processed_df, market_type):
    """对比真实数据中的撤单和撮合引擎生成的撤单."""
    print(f"\n正在验证 {market_type} 撤单数据一致性...")

    real_cancel_ids = set()

    # 1. 提取真实撤单 ID
    if market_type == "SH":
        # 上海: Order表中 order_type='D' 表示撤单
        # processed_df 中包含了合并后的数据
        cancel_df = processed_df[processed_df["order_type"] == "D"]
        if not cancel_df.empty:
            real_cancel_ids = set(cancel_df["orderorino"].astype(int))

    elif market_type == "SZ":
        # 深圳: Trade表中 trade_type='C' 表示撤单
        # processed_df 是合并后的，需要筛选 type='T' 且 trade_type='C'
        cancel_df = processed_df[
            (processed_df["type"] == "T") & (processed_df["trade_type"] == "C")
        ]
        if not cancel_df.empty:
            # SZ撤单记录中，buy_id 或 sell_id 中有一个是被撤的订单ID
            # 通常取两者中的最大值即为订单号
            ids = cancel_df[["buy_id", "sell_id"]].max(axis=1).astype(int)
            real_cancel_ids = set(ids)

    # 2. 提取撮合引擎生成的撤单 ID
    sim_cancel_ids = set()
    for c in lob.cancels:
        sim_cancel_ids.add(c.order_id)

    # 3. 对比
    # 注意：这里我们允许模拟器比真实数据多撤一些单（例如部分成交后的自动撤单），
    # 但真实数据中发生的撤单，模拟器必须也执行了。

    missing_cancels = real_cancel_ids - sim_cancel_ids

    print(f"真实撤单数: {len(real_cancel_ids)}")
    print(f"模拟撤单数: {len(sim_cancel_ids)}")

    if len(missing_cancels) > 0:
        print(f"[ERROR] 模拟器漏掉了 {len(missing_cancels)} 笔撤单!")
        # 打印前10个漏掉的ID方便调试
        print(f"漏单ID示例: {list(missing_cancels)[:10]}")
        return False

    print("[OK] 撤单验证通过 (所有真实撤单均已在模拟中执行)")
    return True


# ==========================================
# 测试用例
# ==========================================


class TestRealData:
    """Integration checks against local CSV/Parquet quote files."""

    @pytest.fixture
    def data_loader(self):
        """加载数据的 Fixture，只加载一次."""
        cfg = REAL_DATA_CONFIG

        def ensure_accessible(path: str) -> None:
            try:
                if not Path(path).exists():
                    pytest.skip(f"数据文件不存在，请检查路径: {path}")
            except PermissionError:
                pytest.skip(f"无权限访问真实数据文件: {path}")
            except OSError as exc:
                pytest.skip(f"无法访问真实数据文件: {path} ({exc})")

        print(f"\n加载数据中: {cfg['symbol']}...")
        ensure_accessible(cfg["trade_file"])
        ensure_accessible(cfg["order_file"])

        # 根据后缀读取 CSV 或 Parquet
        def read_file(path):
            try:
                if path.endswith(".csv"):
                    return pl.read_csv(path)
                elif path.endswith(".parquet"):
                    return pl.read_parquet(path)
                else:
                    msg = "不支持的文件格式，仅支持 csv 或 parquet"
                    raise ValueError(msg)
            except PermissionError:
                pytest.skip(f"无权限读取真实数据文件: {path}")
            except OSError as exc:
                pytest.skip(f"无法读取真实数据文件: {path} ({exc})")

        trade_pl = read_file(cfg["trade_file"])
        order_pl = read_file(cfg["order_file"])

        return trade_pl, order_pl, cfg

    def test_full_replay_consistency(self, data_loader):
        """主测试函数：回放并验证."""
        trade_pl, order_pl, cfg = data_loader
        symbol = cfg["symbol"]
        market = cfg["market"]
        cut_time = cfg.get("cut_time", 150000000)  # 默认直到收盘

        # 1. 初始化对应的 OrderBook
        if market == "SH":
            lob = OrderBookSH()
        elif market == "SZ":
            lob = OrderBookSZ()
        else:
            pytest.fail("配置中的 market 必须是 'SH' 或 'SZ'")

        print(f"初始化 {market} 订单簿完成，开始预处理数据...")

        # 2. 准备数据 (调用你的 prepare_market_data)
        # 注意：这里会返回 pandas DataFrame，是合并排序后的流
        processed_df = lob.prepare_market_data(
            trade_pl, order_pl, symbol, cut_time=cut_time if cut_time else 150000000
        )

        print(f"数据预处理完成，共 {len(processed_df)} 条记录。开始撮合回放...")

        # 3. 执行撮合
        lob.process_workflow(processed_df)

        print("\n回放结束，开始校验结果...")
        # 这会在当前目录下生成 trades.csv 和 cancels.csv
        lob.export_all_records_csv("trades_result.csv", "cancels_result.csv")
        # 4. 验证成交 (Trades)
        # 使用你代码自带的 compare_df
        trades_match = lob.compare_df(processed_df)

        # 5. 验证撤单 (Cancels)
        cancels_match = verify_cancels_consistency(lob, processed_df, market)

        # 6. 断言结果
        assert trades_match is True, "成交记录对比不一致！请查看上方日志中的差异详情。"
        assert cancels_match is True, "撤单记录对比不一致！请查看上方日志中的漏单详情。"

        print(f"\nSUCCESS: {symbol} 在 {market} 市场的回测与真实数据完全一致！")


if __name__ == "__main__":
    # 允许直接运行此脚本
    # 也可以在命令行使用: pytest tests/test_real_data.py -v -s
    sys.exit(pytest.main(["-v", "-s", __file__]))

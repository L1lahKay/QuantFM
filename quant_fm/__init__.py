"""
quant_fm：A 股订单流基础模型端到端预训练。

本包在现有 PyLOB 沪深限价订单簿重建引擎之上，构建可复现、可验证的训练流水线：

    L2 委托/成交/快照
      -> pylob LOB 重建（复用）
      -> cn_l2_v1 规范事件流        (quant_fm.schema)
      -> 全局字段级分词器 / 词表    (quant_fm.tokenizer)
      -> 分片清单 + 时间切分         (quant_fm.manifest)
      -> 解码器多任务下一事件 FM     (quant_fm.pretrain)
      -> 冻结的股日嵌入              (quant_fm.embedding)
      -> 横截面排序器 + 回测门控     (quant_fm.downstream)

重量级模块（``pretrain``、``embedding``、``downstream``）惰性导入 torch，
因此 ``schema``、``tokenizer`` 和 ``manifest`` 可在无 GPU 环境下使用。
"""

from __future__ import annotations

__version__ = "0.1.0"

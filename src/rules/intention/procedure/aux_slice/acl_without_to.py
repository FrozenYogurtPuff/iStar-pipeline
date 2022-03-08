import logging
from typing import List

from src.utils.spacy import token_not_start
from src.utils.typing import HybridToken, SpacySpan

logger = logging.getLogger(__name__)


def acl_without_to(s: SpacySpan) -> List[HybridToken]:

    pool: List[HybridToken] = list()
    for token in s:
        if (
            token.dep_ == "acl"
            and token_not_start(token.head)
            and token.head.nbor(1).lower_ != "to"
        ):
            if token_not_start(token):
                pool.append(token.nbor(-1))

    return pool


# TODO: 冒烟测试通过后，检查jsonl里，这些分割的准确率？
# 最好加一个指示：哪半句是 Core，哪半句是 Aux
# TODO: 继续完成 relcl，ccomp 等其它 dep 规则 to slice aux
# TODO: 与 BERT 融合，能够提升 Aux 分割的准确率吗？

# TODO: 推进 Entity 深入规则
# 添加一个 inspector，逐 item in sent 检查，在开启该规则后，正确错误（不同颜色显示）
# 打印同个规则错误的文本，调用 displacy 进行可视化，观察有无异同

# TODO: 使用 TypeVar 和包装过一次的 Generic[FixEntityLabel, ...] 重构目前的类型系统

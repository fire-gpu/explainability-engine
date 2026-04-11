"""
轻量级状态机模块

管理分析引擎的生命周期状态转换，支持跳过逻辑和历史记录查询。
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from core.models import StateLog


# ============================================================
# 状态枚举
# ============================================================


class AnalysisState(enum.Enum):
    """分析引擎状态枚举

    定义了分析引擎从空闲到完成的所有可能状态。
    """

    IDLE = "IDLE"  # 空闲，等待输入
    PARSING = "PARSING"  # 数据解析阶段
    DESCRIPTIVE = "DESCRIPTIVE"  # 描述性分析阶段
    CAUSAL = "CAUSAL"  # 因果分析阶段
    PREDICTIVE = "PREDICTIVE"  # 预测模拟阶段
    REPORTING = "REPORTING"  # 报告生成阶段
    DONE = "DONE"  # 分析完成
    ERROR = "ERROR"  # 发生错误


# ============================================================
# 状态转换表
# ============================================================

# 格式: (当前状态, 事件) -> 下一状态
# 特殊事件 "skip_causal" 和 "skip_predictive" 用于跳过可选阶段
_TRANSITIONS: dict[tuple[AnalysisState, str], AnalysisState] = {
    # 正常流程
    (AnalysisState.IDLE, "start"): AnalysisState.PARSING,
    (AnalysisState.PARSING, "parse_done"): AnalysisState.DESCRIPTIVE,
    (AnalysisState.DESCRIPTIVE, "descriptive_done"): AnalysisState.CAUSAL,
    (AnalysisState.CAUSAL, "causal_done"): AnalysisState.PREDICTIVE,
    (AnalysisState.PREDICTIVE, "predictive_done"): AnalysisState.REPORTING,
    (AnalysisState.REPORTING, "report_done"): AnalysisState.DONE,
    # 跳过因果分析
    (AnalysisState.DESCRIPTIVE, "skip_causal"): AnalysisState.PREDICTIVE,
    # 跳过预测模拟
    (AnalysisState.CAUSAL, "skip_predictive"): AnalysisState.REPORTING,
    # 同时跳过因果和预测（从描述性直接到报告）
    (AnalysisState.DESCRIPTIVE, "skip_to_report"): AnalysisState.REPORTING,
    # 跳过因果且跳过预测（从描述性直接到报告的另一种触发方式）
    (AnalysisState.PREDICTIVE, "skip_predictive"): AnalysisState.REPORTING,
    # 错误处理：任何非终态都可以进入 ERROR
    (AnalysisState.PARSING, "error"): AnalysisState.ERROR,
    (AnalysisState.DESCRIPTIVE, "error"): AnalysisState.ERROR,
    (AnalysisState.CAUSAL, "error"): AnalysisState.ERROR,
    (AnalysisState.PREDICTIVE, "error"): AnalysisState.ERROR,
    (AnalysisState.REPORTING, "error"): AnalysisState.ERROR,
}


# ============================================================
# 状态机
# ============================================================


class StateMachine:
    """轻量级状态机

    管理分析引擎的状态转换，记录每次转换的历史日志。

    Attributes:
        current_state: 当前状态，初始为 IDLE
        history: 状态转换历史日志列表
    """

    def __init__(self) -> None:
        """初始化状态机，将当前状态设为 IDLE。"""
        self.current_state: AnalysisState = AnalysisState.IDLE
        self.history: list[StateLog] = []

    def transition(self, event: str, **kwargs: Any) -> AnalysisState:
        """执行状态转换

        根据当前状态和触发事件，查找转换表并更新状态。
        如果转换不合法，抛出 ValueError。

        Args:
            event: 触发事件名称
            **kwargs: 附加信息，用于记录到 StateLog 中

        Returns:
            转换后的新状态

        Raises:
            ValueError: 当转换不合法时
        """
        key = (self.current_state, event)
        if key not in _TRANSITIONS:
            raise ValueError(
                f"非法状态转换: 当前状态={self.current_state.value}, "
                f"事件={event!r}"
            )

        next_state = _TRANSITIONS[key]
        self.current_state = next_state

        # 记录日志
        log = StateLog(
            state=next_state.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=kwargs.get("input_summary", ""),
            output_summary=kwargs.get("output_summary", ""),
            duration_ms=kwargs.get("duration_ms", 0.0),
            error=kwargs.get("error"),
        )
        self.history.append(log)

        return next_state

    def can_transition_to(self, state: AnalysisState) -> bool:
        """检查是否可以转换到指定状态

        遍历转换表，判断从当前状态是否存在任何事件可以到达目标状态。

        Args:
            state: 目标状态

        Returns:
            是否可以转换到目标状态
        """
        for (current, _event), next_state in _TRANSITIONS.items():
            if current == self.current_state and next_state == state:
                return True
        return False

    def get_history(self) -> list[StateLog]:
        """获取完整的状态转换历史

        Returns:
            状态日志列表（按时间顺序）
        """
        return list(self.history)

    def reset(self) -> None:
        """重置状态机到初始状态"""
        self.current_state = AnalysisState.IDLE
        self.history.clear()

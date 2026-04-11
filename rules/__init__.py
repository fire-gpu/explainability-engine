"""
规则引擎模块 —— 领域规则与约束管理
"""

from rules.engine import Rule, RulesEngine
from rules.explanation_rules import create_explanation_rules, register_explanation_rules
from rules.flow_rules import create_flow_rules, register_flow_rules
from rules.validation_rules import create_validation_rules, register_validation_rules

__all__ = [
    # 核心引擎
    "Rule",
    "RulesEngine",
    # 流程控制规则
    "create_flow_rules",
    "register_flow_rules",
    # 校验规则
    "create_validation_rules",
    "register_validation_rules",
    # 解释模板规则
    "create_explanation_rules",
    "register_explanation_rules",
]

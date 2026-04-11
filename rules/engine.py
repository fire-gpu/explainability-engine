"""
规则引擎模块

提供轻量级的规则引擎，支持条件评估和动作触发，
用于分析流程控制、结果校验和解释模板管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Rule:
    """单条规则

    定义一条由条件判断和触发动作组成的规则。

    Args:
        name: 规则名称，用于标识和日志
        rule_type: 规则类型，如 ``flow`` / ``validation`` / ``explanation``
        condition: 条件判断函数，接收 context 字典，返回 bool
        action: 触发动作函数，接收 context 字典，返回任意结果
        description: 规则描述
    """

    name: str
    rule_type: str
    condition: Callable[[dict[str, Any]], bool]
    action: Callable[[dict[str, Any]], Any]
    description: str = ""


class RulesEngine:
    """轻量规则引擎

    管理一组 :class:`Rule` 对象，支持批量评估和动作触发。

    使用示例::

        engine = RulesEngine()
        engine.add_rule(Rule(
            name="变量数过多",
            rule_type="flow",
            condition=lambda ctx: ctx.get("variable_count", 0) > 50,
            action=lambda ctx: {"skip_causal_graph": True},
            description="变量数超过 50 时跳过因果图可视化",
        ))
        results = engine.evaluate({"variable_count": 60})
    """

    def __init__(self) -> None:
        self.rules: list[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """添加一条规则

        Args:
            rule: 要添加的规则对象
        """
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """按名称移除规则

        Args:
            name: 规则名称

        Returns:
            bool: 是否成功移除
        """
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False

    def evaluate(self, context: dict[str, Any]) -> list[tuple[Rule, bool]]:
        """评估所有规则，返回 (规则, 是否触发) 列表

        遍历所有已注册的规则，对每条规则执行条件判断。

        Args:
            context: 评估上下文字典

        Returns:
            list[tuple[Rule, bool]]: 每条规则及其是否被触发的结果
        """
        results: list[tuple[Rule, bool]] = []
        for rule in self.rules:
            try:
                triggered = rule.condition(context)
            except Exception:
                # 条件评估异常时视为未触发
                triggered = False
            results.append((rule, triggered))
        return results

    def get_triggered_actions(self, context: dict[str, Any]) -> list[tuple[str, Any]]:
        """返回所有被触发规则的动作

        对评估为 True 的规则执行其动作函数，并返回结果。

        Args:
            context: 评估上下文字典

        Returns:
            list[tuple[str, Any]]: (规则名称, 动作结果) 列表
        """
        actions: list[tuple[str, Any]] = []
        for rule in self.rules:
            try:
                if rule.condition(context):
                    result = rule.action(context)
                    actions.append((rule.name, result))
            except Exception:
                # 跳过执行异常的规则
                continue
        return actions

    def get_rules_by_type(self, rule_type: str) -> list[Rule]:
        """按类型获取规则列表

        Args:
            rule_type: 规则类型

        Returns:
            list[Rule]: 匹配类型的规则列表
        """
        return [rule for rule in self.rules if rule.rule_type == rule_type]

    def clear(self) -> None:
        """清空所有规则"""
        self.rules.clear()

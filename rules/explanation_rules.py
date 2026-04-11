"""
解释模板规则模块

根据不同业务场景预定义解释模板规则，
确保生成的叙述重点关注各领域的核心要素。
"""

from __future__ import annotations

from typing import Any

from rules.engine import Rule, RulesEngine


def create_explanation_rules() -> list[Rule]:
    """创建预定义的解释模板规则列表

    包含以下规则：
    - 定价场景重点关注价格弹性、竞品因素
    - 风险场景重点关注风险因子、尾部风险
    - 营销场景重点关注转化率、ROI、渠道效果
    - 供应链场景重点关注库存周转、交付延迟
    - 通用场景使用默认解释模板

    Returns:
        list[Rule]: 解释模板规则列表
    """
    rules: list[Rule] = []

    # 定价场景：重点关注价格弹性、竞品因素
    rules.append(
        Rule(
            name="pricing_focus",
            rule_type="explanation",
            condition=lambda ctx: ctx.get("domain", "") in ("pricing", "定价", "价格"),
            action=lambda ctx: {
                "action": "set_focus",
                "focus_areas": [
                    "价格弹性分析",
                    "竞品价格对比",
                    "价格敏感度",
                    "最优定价策略",
                    "价格变动对需求的影响",
                ],
                "highlight_variables": [
                    "price", "价格", "competitor_price", "竞品价格",
                    "demand", "需求", "revenue", "收入",
                ],
                "reason": "定价场景下重点关注价格弹性和竞品因素",
            },
            description="定价场景重点关注价格弹性、竞品因素",
        )
    )

    # 风险场景：重点关注风险因子、尾部风险
    rules.append(
        Rule(
            name="risk_focus",
            rule_type="explanation",
            condition=lambda ctx: ctx.get("domain", "") in ("risk", "风险", "风控"),
            action=lambda ctx: {
                "action": "set_focus",
                "focus_areas": [
                    "风险因子识别",
                    "尾部风险分析",
                    "风险集中度",
                    "压力测试结果",
                    "风险缓释措施",
                ],
                "highlight_variables": [
                    "risk_score", "风险评分", "loss", "损失",
                    "default_rate", "违约率", "exposure", "敞口",
                ],
                "reason": "风险场景下重点关注风险因子和尾部风险",
            },
            description="风险场景重点关注风险因子、尾部风险",
        )
    )

    # 营销场景：重点关注转化率、ROI、渠道效果
    rules.append(
        Rule(
            name="marketing_focus",
            rule_type="explanation",
            condition=lambda ctx: ctx.get("domain", "") in ("marketing", "营销", "广告"),
            action=lambda ctx: {
                "action": "set_focus",
                "focus_areas": [
                    "转化率分析",
                    "ROI 评估",
                    "渠道效果对比",
                    "用户画像洞察",
                    "营销活动效果归因",
                ],
                "highlight_variables": [
                    "conversion_rate", "转化率", "roi", "ROI",
                    "ctr", "点击率", "spend", "花费", "channel", "渠道",
                ],
                "reason": "营销场景下重点关注转化率、ROI 和渠道效果",
            },
            description="营销场景重点关注转化率、ROI、渠道效果",
        )
    )

    # 供应链场景：重点关注库存周转、交付延迟
    rules.append(
        Rule(
            name="supply_chain_focus",
            rule_type="explanation",
            condition=lambda ctx: ctx.get("domain", "") in (
                "supply_chain", "供应链", "物流",
            ),
            action=lambda ctx: {
                "action": "set_focus",
                "focus_areas": [
                    "库存周转率分析",
                    "交付延迟原因",
                    "供应商绩效评估",
                    "需求预测准确性",
                    "成本优化建议",
                ],
                "highlight_variables": [
                    "inventory", "库存", "lead_time", "交期",
                    "delivery_delay", "交付延迟", "cost", "成本",
                ],
                "reason": "供应链场景下重点关注库存周转和交付延迟",
            },
            description="供应链场景重点关注库存周转、交付延迟",
        )
    )

    # 通用场景：使用默认解释模板
    rules.append(
        Rule(
            name="generic_focus",
            rule_type="explanation",
            condition=lambda ctx: ctx.get("domain", "") == "generic",
            action=lambda ctx: {
                "action": "set_focus",
                "focus_areas": [
                    "数据整体概况",
                    "关键变量分布",
                    "重要趋势和模式",
                    "异常值和异常模式",
                    "综合建议",
                ],
                "highlight_variables": [],
                "reason": "通用场景使用默认解释模板",
            },
            description="通用场景使用默认解释模板",
        )
    )

    return rules


def register_explanation_rules(engine: RulesEngine) -> None:
    """将解释模板规则注册到规则引擎

    Args:
        engine: 目标规则引擎实例
    """
    for rule in create_explanation_rules():
        engine.add_rule(rule)

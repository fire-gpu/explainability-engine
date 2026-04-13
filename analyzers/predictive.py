"""
预测模拟器模块

回答"如果改变 X，未来会怎样"——基于因果分析结果，
生成情景模拟、敏感性分析和 What-If 分析，帮助决策者
理解不同策略的潜在影响。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.models import (
    AnalysisInput,
    CausalEffect,
    CausalResult,
    DescriptionResult,
    PredictionResult,
    Scenario,
    SensitivityAnalysis,
    SensitivityEntry,
    WhatIf,
)

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger(__name__)


class PredictiveSimulator:
    """预测模拟器 - 回答"如果改变 X，未来会怎样"

    基于因果分析结果，通过情景模拟、敏感性分析和 What-If 分析，
    量化不同变量变化对目标变量的影响，辅助决策制定。

    Args:
        llm_client: LLM 客户端实例，为 None 时使用模板回退生成叙述
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """初始化预测模拟器

        Args:
            llm_client: 可选的 LLM 客户端，用于生成自然语言叙述
        """
        self.llm = llm_client

    def simulate(
        self,
        analysis_input: AnalysisInput,
        desc_result: DescriptionResult,
        causal_result: CausalResult | None,
    ) -> PredictionResult:
        """执行预测模拟

        分析流程：
        1. 基于因果分析设置情景（乐观/中性/悲观）
        2. 执行敏感性分析
        3. 执行 What-If 分析
        4. LLM 生成预测解释叙述

        Args:
            analysis_input: 分析输入
            desc_result: 描述性分析结果
            causal_result: 因果分析结果（可选）

        Returns:
            PredictionResult: 预测模拟结果
        """
        df = analysis_input.data
        target = analysis_input.config.target_variable
        domain = analysis_input.metadata.domain
        audience = analysis_input.config.audience
        domain_config = analysis_input.metadata.domain_config

        logger.info("开始预测模拟")

        # 从因果分析中提取因果效应
        causal_effects = (
            causal_result.causal_effects if causal_result else []
        )

        # 如果没有因果效应，基于描述性分析中的变量重要性构建伪效应
        if not causal_effects:
            causal_effects = self._derive_effects_from_descriptive(
                df, desc_result, target
            )
            logger.info("从描述性分析推导了 %d 个伪因果效应", len(causal_effects))

        # 1. 生成情景模拟（优先对领域关键变量做模拟）
        scenarios = self._generate_scenarios(df, causal_effects, target, domain_config)
        logger.info("情景模拟完成，共 %d 个场景", len(scenarios))

        # 2. 敏感性分析
        sensitivity = self._sensitivity_analysis(df, target, causal_effects)
        logger.info(
            "敏感性分析完成，共 %d 个变量",
            len(sensitivity.entries) if sensitivity else 0,
        )

        # 3. What-If 分析
        what_ifs = self._what_if_analysis(df, target, causal_effects)
        logger.info("What-If 分析完成，共 %d 条", len(what_ifs))

        # 4. 生成叙述
        narrative = self._generate_narrative(
            scenarios=scenarios,
            sensitivity=sensitivity,
            what_ifs=what_ifs,
            domain=domain,
            audience=audience,
            business_understanding=analysis_input.business_understanding,
            domain_config=domain_config,
        )
        logger.info("预测叙述生成完成")

        return PredictionResult(
            scenarios=scenarios,
            sensitivity=sensitivity,
            what_ifs=what_ifs,
            narrative=narrative,
        )

    def _derive_effects_from_descriptive(
        self,
        df: pd.DataFrame,
        desc_result: DescriptionResult,
        target: str | None,
    ) -> list[CausalEffect]:
        """从描述性分析结果推导伪因果效应

        当没有因果分析结果时，基于变量重要性和相关性构建
        伪因果效应用于模拟。

        Args:
            df: 数据框
            desc_result: 描述性分析结果
            target: 目标变量名

        Returns:
            list[CausalEffect]: 推导的因果效应列表
        """
        effects: list[CausalEffect] = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 如果有目标变量，基于相关性构建效应
        if target and target in numeric_cols:
            for corr in desc_result.correlations:
                if corr.var2 == target and corr.var1 in numeric_cols:
                    # 用相关系数近似效应大小
                    std_x = df[corr.var1].std()
                    std_y = df[target].std()
                    effect_size = corr.coefficient * (std_y / (std_x + 1e-10))

                    effects.append(
                        CausalEffect(
                            treatment=corr.var1,
                            outcome=target,
                            effect_size=round(float(effect_size), 4),
                            p_value=None,
                            method="相关性推导（近似）",
                        )
                    )
                elif corr.var1 == target and corr.var2 in numeric_cols:
                    std_x = df[corr.var2].std()
                    std_y = df[target].std()
                    effect_size = corr.coefficient * (std_y / (std_x + 1e-10))

                    effects.append(
                        CausalEffect(
                            treatment=corr.var2,
                            outcome=target,
                            effect_size=round(float(effect_size), 4),
                            p_value=None,
                            method="相关性推导（近似）",
                        )
                    )

        # 如果仍然没有效应，基于变量重要性构建
        if not effects and desc_result.variable_importance:
            for var_imp in desc_result.variable_importance[:5]:
                if var_imp.name in numeric_cols and var_imp.name != target:
                    effects.append(
                        CausalEffect(
                            treatment=var_imp.name,
                            outcome=target or "目标",
                            effect_size=round(float(var_imp.score), 4),
                            p_value=None,
                            method="重要性推导（近似）",
                        )
                    )

        return effects

    def _generate_scenarios(
        self,
        df: pd.DataFrame,
        causal_effects: list[CausalEffect],
        target: str | None,
        domain_config: object | None = None,
    ) -> list[Scenario]:
        """生成情景模拟

        基于因果效应生成三种场景：
        - 乐观：所有正面因素 +10%
        - 中性：维持现状
        - 悲观：所有负面因素 -10%

        如果提供了领域配置且包含 key_variables，会优先对
        领域关键变量做情景模拟，确保业务最关注的变量被覆盖。

        Args:
            df: 数据框
            causal_effects: 因果效应列表
            target: 目标变量名
            domain_config: 领域配置对象（可选）

        Returns:
            list[Scenario]: 模拟场景列表
        """
        scenarios: list[Scenario] = []

        if not causal_effects or not target or target not in df.columns:
            # 无因果效应或无目标变量时，返回中性场景
            baseline = float(df[target].mean()) if target and target in df.columns else 0.0
            scenarios.append(
                Scenario(
                    name="中性（维持现状）",
                    description="无因果分析数据，仅展示当前基线值",
                    predicted_outcomes={target: round(baseline, 4)} if target else {},
                )
            )
            return scenarios

        # 如果有领域配置，优先筛选关键变量的因果效应
        effective_effects = causal_effects
        if domain_config is not None and hasattr(domain_config, "key_variables"):
            key_vars = set(domain_config.key_variables) if domain_config.key_variables else set()
            if key_vars:
                # 将关键变量的效应排在前面
                key_effects = [e for e in causal_effects if e.treatment in key_vars]
                other_effects = [e for e in causal_effects if e.treatment not in key_vars]
                if key_effects:
                    effective_effects = key_effects + other_effects
                    logger.info(
                        "情景模拟优先处理 %d 个领域关键变量",
                        len(key_effects),
                    )

        baseline = float(df[target].mean())

        # 区分正面和负面因素
        positive_effects = [e for e in effective_effects if e.effect_size > 0]
        negative_effects = [e for e in effective_effects if e.effect_size < 0]

        # 乐观场景：正面因素 +10%，负面因素 -10%
        optimistic_change = 0.0
        optimistic_params: dict[str, float] = {}

        for eff in positive_effects:
            if eff.treatment in df.columns:
                mean_val = float(df[eff.treatment].mean())
                delta = mean_val * 0.1 if mean_val != 0 else 0.0
                optimistic_params[eff.treatment] = round(mean_val + delta, 4)
                optimistic_change += eff.effect_size * delta

        for eff in negative_effects:
            if eff.treatment in df.columns:
                mean_val = float(df[eff.treatment].mean())
                delta = mean_val * 0.1 if mean_val != 0 else 0.0
                optimistic_params[eff.treatment] = round(mean_val - delta, 4)
                optimistic_change -= eff.effect_size * delta

        scenarios.append(
            Scenario(
                name="乐观",
                description="所有正面驱动因素提升 10%，负面因素降低 10%",
                parameters=optimistic_params,
                predicted_outcomes={
                    target: round(baseline + optimistic_change, 4)
                },
            )
        )

        # 中性场景
        neutral_params: dict[str, float] = {}
        for eff in causal_effects:
            if eff.treatment in df.columns:
                neutral_params[eff.treatment] = round(float(df[eff.treatment].mean()), 4)

        scenarios.append(
            Scenario(
                name="中性（维持现状）",
                description="所有变量维持当前均值水平",
                parameters=neutral_params,
                predicted_outcomes={target: round(baseline, 4)},
            )
        )

        # 悲观场景：正面因素 -10%，负面因素 +10%
        pessimistic_change = 0.0
        pessimistic_params: dict[str, float] = {}

        for eff in positive_effects:
            if eff.treatment in df.columns:
                mean_val = float(df[eff.treatment].mean())
                delta = mean_val * 0.1 if mean_val != 0 else 0.0
                pessimistic_params[eff.treatment] = round(mean_val - delta, 4)
                pessimistic_change -= eff.effect_size * delta

        for eff in negative_effects:
            if eff.treatment in df.columns:
                mean_val = float(df[eff.treatment].mean())
                delta = mean_val * 0.1 if mean_val != 0 else 0.0
                pessimistic_params[eff.treatment] = round(mean_val + delta, 4)
                pessimistic_change += eff.effect_size * delta

        scenarios.append(
            Scenario(
                name="悲观",
                description="所有正面驱动因素降低 10%，负面因素增加 10%",
                parameters=pessimistic_params,
                predicted_outcomes={
                    target: round(baseline + pessimistic_change, 4)
                },
            )
        )

        return scenarios

    def _sensitivity_analysis(
        self,
        df: pd.DataFrame,
        target: str | None,
        causal_effects: list[CausalEffect],
    ) -> SensitivityAnalysis:
        """敏感性分析

        对每个变量，计算其变化对目标变量的影响程度。
        使用基于回归系数的敏感性分数：sensitivity = |effect_size * std(treatment)|。

        Args:
            df: 数据框
            target: 目标变量名
            causal_effects: 因果效应列表

        Returns:
            SensitivityAnalysis: 敏感性分析结果
        """
        entries: list[SensitivityEntry] = []

        if not causal_effects:
            return SensitivityAnalysis(entries=[], method="基于回归系数")

        # 目标变量的标准差（用于归一化）
        target_std = float(df[target].std()) if target and target in df.columns else 1.0

        for eff in causal_effects:
            if eff.treatment not in df.columns:
                continue

            treatment_std = float(df[eff.treatment].std())

            # 敏感性分数：效应大小 * 处理变量标准差
            sensitivity_score = abs(eff.effect_size * treatment_std)

            # 归一化（相对于目标变量标准差）
            normalized_score = sensitivity_score / (target_std + 1e-10)

            # 影响方向
            if eff.effect_size > 0:
                direction = "positive"
            elif eff.effect_size < 0:
                direction = "negative"
            else:
                direction = "neutral"

            # 影响范围：+/- 1 个标准差的变化
            impact_low = round(eff.effect_size * treatment_std, 4)
            impact_high = round(-eff.effect_size * treatment_std, 4)
            impact_range = (min(impact_low, impact_high), max(impact_low, impact_high))

            entries.append(
                SensitivityEntry(
                    variable=eff.treatment,
                    sensitivity_score=round(normalized_score, 4),
                    direction=direction,
                    impact_range=impact_range,
                )
            )

        # 按敏感性分数降序排列
        entries.sort(key=lambda e: e.sensitivity_score, reverse=True)

        return SensitivityAnalysis(entries=entries, method="基于回归系数")

    def _what_if_analysis(
        self,
        df: pd.DataFrame,
        target: str | None,
        causal_effects: list[CausalEffect],
    ) -> list[WhatIf]:
        """What-If 分析

        对 top 5 重要的因果变量，模拟其 +-20% 变化对目标变量的影响。

        Args:
            df: 数据框
            target: 目标变量名
            causal_effects: 因果效应列表

        Returns:
            list[WhatIf]: What-If 分析结果列表
        """
        what_ifs: list[WhatIf] = []

        if not causal_effects or not target or target not in df.columns:
            return what_ifs

        # 按效应大小排序，取 top 5
        sorted_effects = sorted(
            causal_effects, key=lambda e: abs(e.effect_size), reverse=True
        )[:5]

        baseline = float(df[target].mean())

        for eff in sorted_effects:
            if eff.treatment not in df.columns:
                continue

            original_value = float(df[eff.treatment].mean())

            if original_value == 0:
                # 使用标准差代替
                original_value = float(df[eff.treatment].std())
                if original_value == 0:
                    continue

            delta = original_value * 0.2

            # +20% 变化
            outcome_change_plus = eff.effect_size * delta
            what_ifs.append(
                WhatIf(
                    variable=eff.treatment,
                    original_value=round(original_value, 4),
                    new_value=round(original_value + delta, 4),
                    outcome_change=round(outcome_change_plus, 4),
                    # 置信度基于 p 值
                    confidence=round(1.0 - (eff.p_value or 0.5), 4),
                )
            )

            # -20% 变化
            outcome_change_minus = -eff.effect_size * delta
            what_ifs.append(
                WhatIf(
                    variable=eff.treatment,
                    original_value=round(original_value, 4),
                    new_value=round(original_value - delta, 4),
                    outcome_change=round(outcome_change_minus, 4),
                    confidence=round(1.0 - (eff.p_value or 0.5), 4),
                )
            )

        return what_ifs

    def _generate_narrative(
        self,
        scenarios: list[Scenario],
        sensitivity: SensitivityAnalysis | None,
        what_ifs: list[WhatIf],
        domain: str,
        audience: str,
        business_understanding: object | None = None,
        domain_config: object | None = None,
    ) -> str:
        """生成预测解释叙述

        如果配置了 LLM 客户端，调用 LLM 生成叙述；
        否则使用模板拼接生成结构化文字描述。

        如果提供了领域配置且包含 explanation_focus，
        会将其传入 prompt，引导 LLM 重点关注这些方向。

        Args:
            scenarios: 模拟场景列表
            sensitivity: 敏感性分析结果
            what_ifs: What-If 分析列表
            domain: 业务领域
            audience: 目标受众
            business_understanding: L1 数据预扫描的业务理解结果（可选）
            domain_config: 领域配置对象（可选）

        Returns:
            str: 自然语言预测解释文本
        """
        # 构建场景模拟结果文本
        scenarios_text = self._build_scenarios_text(scenarios, sensitivity, what_ifs)

        # 数据概要
        data_summary = f"共模拟了 {len(scenarios)} 个场景"
        if sensitivity:
            data_summary += f"，分析了 {len(sensitivity.entries)} 个变量的敏感性"
        data_summary += f"，执行了 {len(what_ifs)} 个 What-If 分析。"

        # 提取领域配置中的解释重点方向
        explanation_focus: list[str] = []
        if domain_config is not None and hasattr(domain_config, "explanation_focus"):
            explanation_focus = domain_config.explanation_focus or []

        if self.llm is not None:
            try:
                from llm.prompts import PromptTemplates

                prompt = PromptTemplates.predictive_prompt(
                    data_summary=data_summary,
                    scenarios=scenarios_text,
                    domain=domain or "通用",
                    audience=audience,
                    business_understanding=business_understanding,  # type: ignore[arg-type]
                    explanation_focus=explanation_focus if explanation_focus else None,
                )
                narrative = self.llm.generate(
                    prompt=prompt,
                    system_prompt=PromptTemplates.PREDICTIVE_SYSTEM,
                )
                return narrative.strip()
            except Exception as e:
                logger.warning("LLM 预测叙述生成失败，回退到模板: %s", e)

        # 模板回退
        return self._template_narrative(scenarios, sensitivity, what_ifs, domain)

    def _build_scenarios_text(
        self,
        scenarios: list[Scenario],
        sensitivity: SensitivityAnalysis | None,
        what_ifs: list[WhatIf],
    ) -> str:
        """构建供 LLM 使用的场景模拟结果文本

        Args:
            scenarios: 模拟场景列表
            sensitivity: 敏感性分析结果
            what_ifs: What-If 分析列表

        Returns:
            str: 格式化的场景模拟结果文本
        """
        lines: list[str] = []

        # 场景模拟
        lines.append("### 场景模拟结果")
        for scenario in scenarios:
            lines.append(f"#### {scenario.name}")
            lines.append(f"描述: {scenario.description}")
            if scenario.parameters:
                param_str = ", ".join(
                    f"{k}={v}" for k, v in list(scenario.parameters.items())[:5]
                )
                lines.append(f"参数: {param_str}")
            if scenario.predicted_outcomes:
                outcome_str = ", ".join(
                    f"{k}={v}" for k, v in scenario.predicted_outcomes.items()
                )
                lines.append(f"预测结果: {outcome_str}")
            lines.append("")

        # 敏感性分析
        if sensitivity and sensitivity.entries:
            lines.append("### 敏感性分析")
            for entry in sensitivity.entries[:10]:
                direction_map = {"positive": "正向", "negative": "负向", "neutral": "中性"}
                direction_str = direction_map.get(entry.direction, entry.direction)
                range_str = (
                    f", 影响范围=[{entry.impact_range[0]}, {entry.impact_range[1]}]"
                    if entry.impact_range
                    else ""
                )
                lines.append(
                    f"- {entry.variable}: 敏感性得分={entry.sensitivity_score}, "
                    f"方向={direction_str}{range_str}"
                )

        # What-If 分析
        if what_ifs:
            lines.append("\n### What-If 分析")
            for wi in what_ifs[:10]:
                lines.append(
                    f"- 「{wi.variable}」从 {wi.original_value} 变为 {wi.new_value}，"
                    f"结果变化={wi.outcome_change}, 置信度={wi.confidence}"
                )

        return "\n".join(lines)

    def _template_narrative(
        self,
        scenarios: list[Scenario],
        sensitivity: SensitivityAnalysis | None,
        what_ifs: list[WhatIf],
        domain: str,
    ) -> str:
        """使用模板生成预测叙述（LLM 不可用时的回退方案）

        Args:
            scenarios: 模拟场景列表
            sensitivity: 敏感性分析结果
            what_ifs: What-If 分析列表
            domain: 业务领域

        Returns:
            str: 模板化的预测叙述文本
        """
        parts: list[str] = []

        domain_prefix = f"在「{domain}」领域" if domain else "在本次分析中"

        # 场景概述
        if len(scenarios) >= 3:
            optimistic = scenarios[0]
            neutral = scenarios[1]
            pessimistic = scenarios[2]

            # 提取目标变量预测值
            target_key = next(iter(optimistic.predicted_outcomes), None)
            if target_key:
                opt_val = optimistic.predicted_outcomes.get(target_key, 0)
                neu_val = neutral.predicted_outcomes.get(target_key, 0)
                pes_val = pessimistic.predicted_outcomes.get(target_key, 0)

                parts.append(
                    f"{domain_prefix}，基于因果分析进行了情景模拟：\n"
                    f"- 乐观场景下，「{target_key}」预测值为 {opt_val}\n"
                    f"- 中性场景下，「{target_key}」预测值为 {neu_val}\n"
                    f"- 悲观场景下，「{target_key}」预测值为 {pes_val}"
                )
        elif scenarios:
            parts.append(f"{domain_prefix}，生成了 {len(scenarios)} 个模拟场景。")

        # 敏感性分析
        if sensitivity and sensitivity.entries:
            top_sensitive = sensitivity.entries[:3]
            sens_names = "、".join(
                f"「{e.variable}」（得分 {e.sensitivity_score}）" for e in top_sensitive
            )
            parts.append(f"敏感性最高的变量为：{sens_names}。")

        # What-If 洞察
        if what_ifs:
            # 找出影响最大的 What-If
            max_wi = max(what_ifs, key=lambda w: abs(w.outcome_change or 0))
            parts.append(
                f"What-If 分析显示，「{max_wi.variable}」的变化对结果影响最大"
                f"（变化 {max_wi.original_value} -> {max_wi.new_value}，"
                f"结果变化 {max_wi.outcome_change}）。"
            )

        # 注意事项
        parts.append(
            "注意：以上预测基于线性因果效应假设，实际结果可能受到非线性因素、"
            "外部环境和交互效应的影响，建议结合业务判断综合决策。"
        )

        return "\n\n".join(parts)

"""
报告生成器模块

负责将描述性分析、因果分析和预测模拟的结果汇总，
生成分层的可解释性报告，包含执行摘要、详细分析、技术附录和图表数据。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.models import (
    AnalysisInput,
    CausalResult,
    Chart,
    CorrelationPair,
    DescriptionResult,
    ExplainabilityReport,
    PredictionResult,
    ReportMetadata,
    VarImportance,
)

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器 - 生成分层可解释性报告

    将各分析阶段的结果汇总为结构化的报告，包含三个层次：
    - 执行摘要：面向业务决策者，突出关键发现和行动建议
    - 详细分析：面向业务+技术受众，包含完整的分析过程
    - 技术附录：面向数据科学家，包含方法论和统计细节

    Args:
        llm_client: LLM 客户端实例，为 None 时使用模板回退生成报告内容
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """初始化报告生成器

        Args:
            llm_client: 可选的 LLM 客户端，用于生成更自然的报告文本
        """
        self.llm = llm_client

    def generate(
        self,
        analysis_input: AnalysisInput,
        desc_result: DescriptionResult,
        causal_result: CausalResult | None,
        pred_result: PredictionResult | None,
    ) -> ExplainabilityReport:
        """生成完整报告

        按照分层结构依次生成执行摘要、详细分析、技术附录，
        并收集图表数据，最终组装为完整的可解释性报告。

        Args:
            analysis_input: 分析输入，包含数据和元信息
            desc_result: 描述性分析结果
            causal_result: 因果分析结果（可选）
            pred_result: 预测模拟结果（可选）

        Returns:
            ExplainabilityReport: 完整的可解释性分析报告
        """
        domain = analysis_input.metadata.domain
        audience = analysis_input.config.audience

        logger.info("开始生成报告，领域: %s，受众: %s", domain, audience)

        # 1. 生成执行摘要（面向业务决策者）
        executive_summary = self._generate_executive_summary(
            desc=desc_result,
            causal=causal_result,
            pred=pred_result,
            domain=domain,
            audience=audience,
        )
        logger.info("执行摘要生成完成，长度: %d 字", len(executive_summary))

        # 2. 生成详细分析（面向业务+技术）
        detailed_analysis = self._generate_detailed_analysis(
            desc=desc_result,
            causal=causal_result,
            pred=pred_result,
            domain=domain,
            audience=audience,
        )
        logger.info("详细分析生成完成，长度: %d 字", len(detailed_analysis))

        # 3. 生成技术附录（面向数据科学家）
        technical_appendix = self._generate_technical_appendix(
            desc=desc_result,
            causal=causal_result,
            pred=pred_result,
            domain=domain,
        )
        logger.info("技术附录生成完成，长度: %d 字", len(technical_appendix))

        # 4. 收集图表数据
        charts = self._collect_charts(desc_result, causal_result, pred_result)
        logger.info("图表数据收集完成，共 %d 个图表", len(charts))

        # 5. 构建报告元信息
        metadata = ReportMetadata(
            title=f"可解释性分析报告 - {domain}" if domain else "可解释性分析报告",
            generated_at=datetime.now(timezone.utc).isoformat(),
            domain=domain,
            data_source=f"数据集（{analysis_input.metadata.row_count} 行，"
                        f"{len(analysis_input.metadata.columns)} 列）",
            analysis_config=analysis_input.config,
        )

        return ExplainabilityReport(
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            technical_appendix=technical_appendix,
            charts=charts,
            metadata=metadata,
        )

    def _generate_executive_summary(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        domain: str,
        audience: str,
    ) -> str:
        """生成执行摘要

        提取 top 5 关键发现，生成 3-5 条行动建议，
        控制在 300 字以内。如果有 LLM 客户端，调用 LLM 生成更自然的摘要。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            domain: 业务领域
            audience: 目标受众

        Returns:
            str: 执行摘要文本
        """
        # 收集关键发现
        findings = self._extract_key_findings(desc, causal, pred)
        # 生成行动建议
        recommendations = self._generate_recommendations(desc, causal, pred)

        # 构建素材文本
        material = self._build_executive_material(findings, recommendations, domain)

        # 尝试使用 LLM 生成更自然的摘要
        if self.llm is not None:
            try:
                from llm.prompts import PromptTemplates

                prompt = (
                    f"## 任务\n"
                    f"请基于以下分析素材，生成一份面向业务决策者的执行摘要。\n\n"
                    f"## 业务领域\n{domain or '通用'}\n\n"
                    f"## 分析素材\n{material}\n\n"
                    f"## 输出要求\n"
                    f"1. 提取最重要的 3-5 条关键发现\n"
                    f"2. 给出 3-5 条可操作的行动建议\n"
                    f"3. 使用简洁的非技术语言\n"
                    f"4. 控制在 300 字以内\n"
                    f"5. 直接输出摘要内容，不要加标题"
                )
                summary = self.llm.generate(
                    prompt=prompt,
                    system_prompt=PromptTemplates.REPORT_SYSTEM,
                )
                # 截断到 300 字
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                return summary.strip()
            except Exception as e:
                logger.warning("LLM 执行摘要生成失败，回退到模板: %s", e)

        # 模板回退
        return self._template_executive_summary(findings, recommendations, domain)

    def _extract_key_findings(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
    ) -> list[str]:
        """提取 top 5 关键发现

        从描述性、因果和预测分析结果中提取最重要的发现。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）

        Returns:
            list[str]: 关键发现列表（最多 5 条）
        """
        findings: list[str] = []

        # 从描述性分析提取
        if desc.variable_importance:
            top_var = desc.variable_importance[0]
            findings.append(
                f"最重要的变量为「{top_var.name}」"
                f"（重要性得分: {top_var.score}）"
            )

        # 强相关性
        strong_corrs = [
            c for c in desc.correlations if abs(c.coefficient) > 0.5
        ]
        if strong_corrs:
            c = strong_corrs[0]
            direction = "正" if c.coefficient > 0 else "负"
            findings.append(
                f"发现强{direction}相关："
                f"「{c.var1}」与「{c.var2}」（r={c.coefficient}）"
            )

        # 异常值
        if desc.anomalies:
            from collections import Counter

            col_counts = Counter(a.column for a in desc.anomalies)
            top_anomaly_col, count = col_counts.most_common(1)[0]
            findings.append(
                f"「{top_anomaly_col}」列存在 {count} 个异常值，建议关注"
            )

        # 因果分析
        if causal and causal.causal_effects:
            significant = [
                e for e in causal.causal_effects
                if e.p_value is not None and e.p_value < 0.05
            ]
            if significant:
                eff = significant[0]
                direction = "正向" if eff.effect_size > 0 else "负向"
                findings.append(
                    f"因果分析发现「{eff.treatment}」对「{eff.outcome}」"
                    f"有{direction}因果效应（效应大小: {eff.effect_size}）"
                )

        # 预测模拟
        if pred and len(pred.scenarios) >= 3:
            target_key = next(iter(pred.scenarios[0].predicted_outcomes), None)
            if target_key:
                opt_val = pred.scenarios[0].predicted_outcomes.get(target_key, 0)
                pes_val = pred.scenarios[2].predicted_outcomes.get(target_key, 0)
                findings.append(
                    f"情景模拟显示「{target_key}」在乐观/悲观场景下"
                    f"分别为 {opt_val} / {pes_val}"
                )

        return findings[:5]

    def _generate_recommendations(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
    ) -> list[str]:
        """生成 3-5 条行动建议

        基于分析结果生成可操作的业务建议。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）

        Returns:
            list[str]: 行动建议列表
        """
        recommendations: list[str] = []

        # 基于异常值的建议
        if desc.anomalies:
            recommendations.append(
                "建议对检测到的异常值进行进一步审查，"
                "确认是否为数据质量问题或真实的业务异常"
            )

        # 基于因果分析的建议
        if causal and causal.causal_effects:
            significant = [
                e for e in causal.causal_effects
                if e.p_value is not None and e.p_value < 0.05
            ]
            if significant:
                top_effect = significant[0]
                if top_effect.effect_size > 0:
                    recommendations.append(
                        f"建议关注「{top_effect.treatment}」的提升，"
                        f"其对「{top_effect.outcome}」有显著正向影响"
                    )
                else:
                    recommendations.append(
                        f"建议控制「{top_effect.treatment}」，"
                        f"其对「{top_effect.outcome}」有显著负向影响"
                    )

        # 基于预测模拟的建议
        if pred and pred.sensitivity and pred.sensitivity.entries:
            top_sensitive = pred.sensitivity.entries[0]
            recommendations.append(
                f"建议重点监控「{top_sensitive.variable}」的变化，"
                f"该变量对目标结果敏感性最高"
            )

        # 基于分布偏态的建议
        high_skew = [
            d for d in desc.distributions
            if d.skewness is not None and abs(d.skewness) > 1
        ]
        if high_skew:
            recommendations.append(
                f"建议对偏态分布的变量（如「{high_skew[0].column}」）"
                f"考虑数据变换以改善分析效果"
            )

        # 通用建议
        recommendations.append(
            "建议定期更新分析以捕捉数据趋势变化，"
            "并结合领域专家知识验证分析结论"
        )

        return recommendations[:5]

    def _build_executive_material(
        self,
        findings: list[str],
        recommendations: list[str],
        domain: str,
    ) -> str:
        """构建执行摘要的素材文本

        Args:
            findings: 关键发现列表
            recommendations: 行动建议列表
            domain: 业务领域

        Returns:
            str: 格式化的素材文本
        """
        lines: list[str] = []

        lines.append(f"### 关键发现")
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. {finding}")

        lines.append(f"\n### 行动建议")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def _template_executive_summary(
        self,
        findings: list[str],
        recommendations: list[str],
        domain: str,
    ) -> str:
        """使用模板生成执行摘要（LLM 不可用时的回退方案）

        Args:
            findings: 关键发现列表
            recommendations: 行动建议列表
            domain: 业务领域

        Returns:
            str: 模板化的执行摘要文本
        """
        parts: list[str] = []

        domain_prefix = f"「{domain}」领域" if domain else "本次"

        # 关键发现
        if findings:
            finding_text = "；".join(findings[:3])
            parts.append(f"在{domain_prefix}的数据分析中，主要发现：{finding_text}。")
        else:
            parts.append(f"在{domain_prefix}的数据分析中，未发现显著的模式或异常。")

        # 行动建议
        if recommendations:
            rec_text = "；".join(recommendations[:3])
            parts.append(f"建议：{rec_text}")

        return "\n\n".join(parts)

    def _generate_detailed_analysis(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        domain: str,
        audience: str,
    ) -> str:
        """生成详细分析

        包含变量分析章节、因果链分析章节和预测情景章节。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            domain: 业务领域
            audience: 目标受众

        Returns:
            str: 详细分析文本
        """
        sections: list[str] = []

        # 变量分析章节
        sections.append(self._build_variable_section(desc))

        # 因果链分析章节
        if causal:
            sections.append(self._build_causal_section(causal))

        # 预测情景章节
        if pred:
            sections.append(self._build_predictive_section(pred))

        # 尝试使用 LLM 润色
        if self.llm is not None:
            try:
                from llm.prompts import PromptTemplates

                raw_content = "\n\n".join(sections)
                prompt = (
                    f"## 任务\n"
                    f"请基于以下分析素材，生成一份面向分析师的详细分析报告。\n\n"
                    f"## 业务领域\n{domain or '通用'}\n\n"
                    f"## 受众要求\n"
                    f"请提供详细的分析过程和关键统计指标。\n\n"
                    f"## 分析素材\n{raw_content}\n\n"
                    f"## 输出要求\n"
                    f"1. 保持结构化，使用章节标题\n"
                    f"2. 包含关键统计指标\n"
                    f"3. 使用清晰的分析语言\n"
                    f"4. 突出重要发现"
                )
                detailed = self.llm.generate(
                    prompt=prompt,
                    system_prompt=PromptTemplates.REPORT_SYSTEM,
                )
                return detailed.strip()
            except Exception as e:
                logger.warning("LLM 详细分析生成失败，回退到模板: %s", e)

        return "\n\n".join(sections)

    def _build_variable_section(self, desc: DescriptionResult) -> str:
        """构建变量分析章节

        Args:
            desc: 描述性分析结果

        Returns:
            str: 变量分析章节文本
        """
        lines: list[str] = ["## 变量分析"]

        # 变量重要性
        if desc.variable_importance:
            lines.append("\n### 变量重要性排名")
            for item in desc.variable_importance[:10]:
                lines.append(
                    f"- #{item.rank} {item.name}: 重要性得分 {item.score}"
                )

        # 分布特征
        if desc.distributions:
            lines.append("\n### 变量分布特征")
            for dist in desc.distributions[:10]:
                lines.append(
                    f"- **{dist.column}**: "
                    f"均值={dist.mean}, 中位数={dist.median}, "
                    f"标准差={dist.std}, 偏度={dist.skewness}, "
                    f"范围=[{dist.min}, {dist.max}], "
                    f"异常值={dist.outlier_count}个"
                )

        # 相关性
        if desc.correlations:
            lines.append("\n### 重要相关性")
            for corr in desc.correlations[:10]:
                direction = "正相关" if corr.coefficient > 0 else "负相关"
                lines.append(
                    f"- {corr.var1} 与 {corr.var2}: "
                    f"{direction}（{corr.method}, r={corr.coefficient}）"
                )

        return "\n".join(lines)

    def _build_causal_section(self, causal: CausalResult) -> str:
        """构建因果链分析章节

        Args:
            causal: 因果分析结果

        Returns:
            str: 因果链分析章节文本
        """
        lines: list[str] = ["## 因果链分析"]

        # 因果图
        if causal.causal_graph and causal.causal_graph.edges:
            lines.append(f"\n### 因果假设图（方法: {causal.causal_graph.method}）")
            for edge in causal.causal_graph.edges:
                lines.append(
                    f"- {edge.from_node} -> {edge.to_node}: "
                    f"效应={edge.effect_size}, p值={edge.p_value}"
                )

        # 因果效应
        if causal.causal_effects:
            lines.append("\n### 因果效应估计")
            for eff in causal.causal_effects:
                ci_str = ""
                if eff.confidence_interval:
                    ci_str = f", 95%CI=[{eff.confidence_interval[0]}, {eff.confidence_interval[1]}]"
                lines.append(
                    f"- {eff.treatment} -> {eff.outcome}: "
                    f"ATE={eff.effect_size}, p={eff.p_value}{ci_str}"
                )

        # 反事实分析
        if causal.counterfactuals:
            lines.append("\n### 反事实推理")
            for cf in causal.counterfactuals[:5]:
                change = (
                    cf.predicted_outcome - cf.original_outcome
                    if cf.predicted_outcome is not None and cf.original_outcome is not None
                    else 0
                )
                lines.append(
                    f"- 若「{cf.treatment}」从 {cf.original_value} "
                    f"变为 {cf.counterfactual_value}，"
                    f"结果变化 {change:+.4f}"
                )

        return "\n".join(lines)

    def _build_predictive_section(self, pred: PredictionResult) -> str:
        """构建预测情景章节

        Args:
            pred: 预测模拟结果

        Returns:
            str: 预测情景章节文本
        """
        lines: list[str] = ["## 预测情景分析"]

        # 场景模拟
        if pred.scenarios:
            lines.append("\n### 场景模拟结果")
            for scenario in pred.scenarios:
                if scenario.predicted_outcomes:
                    outcome_str = ", ".join(
                        f"{k}={v}" for k, v in scenario.predicted_outcomes.items()
                    )
                    lines.append(
                        f"- **{scenario.name}**: {scenario.description}，"
                        f"预测: {outcome_str}"
                    )

        # 敏感性分析
        if pred.sensitivity and pred.sensitivity.entries:
            lines.append(f"\n### 敏感性分析（方法: {pred.sensitivity.method}）")
            for entry in pred.sensitivity.entries[:10]:
                direction_map = {"positive": "正向", "negative": "负向", "neutral": "中性"}
                direction_str = direction_map.get(entry.direction, entry.direction)
                lines.append(
                    f"- {entry.variable}: 敏感性={entry.sensitivity_score}, "
                    f"方向={direction_str}"
                )

        # What-If 分析
        if pred.what_ifs:
            lines.append("\n### What-If 分析")
            for wi in pred.what_ifs[:10]:
                lines.append(
                    f"- 「{wi.variable}」{wi.original_value} -> {wi.new_value}，"
                    f"结果变化={wi.outcome_change}, 置信度={wi.confidence}"
                )

        return "\n".join(lines)

    def _generate_technical_appendix(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        domain: str,
    ) -> str:
        """生成技术附录

        包含方法说明、假设条件、置信区间和局限性说明。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            domain: 业务领域

        Returns:
            str: 技术附录文本
        """
        sections: list[str] = []

        # 方法说明
        sections.append(self._build_methods_section(desc, causal, pred))

        # 假设条件
        sections.append(self._build_assumptions_section())

        # 置信区间汇总
        sections.append(self._build_confidence_section(causal, pred))

        # 局限性说明
        sections.append(self._build_limitations_section())

        return "\n\n".join(sections)

    def _build_methods_section(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
    ) -> str:
        """构建方法说明章节

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）

        Returns:
            str: 方法说明文本
        """
        lines: list[str] = ["## 方法说明"]

        lines.append("\n### 描述性分析")
        lines.append("- 变量重要性：基于互信息（有目标变量时）或方差贡献比例")
        lines.append("- 分布统计：均值、中位数、标准差、偏度、峰度")
        lines.append("- 相关性分析：数值列使用 Pearson 相关系数，涉及分类列使用 Spearman 秩相关")
        lines.append("- 异常值检测：IQR 方法（1.5 倍四分位距）")

        if causal:
            lines.append("\n### 因果分析")
            if causal.causal_graph:
                lines.append(f"- 因果图发现：{causal.causal_graph.method}")
            lines.append("- 因果效应估计：OLS 线性回归 + 控制变量")
            lines.append("- 反事实推理：基于因果效应的线性模拟")

        if pred:
            lines.append("\n### 预测模拟")
            lines.append("- 场景模拟：乐观/中性/悲观三种场景（变量变化 +-10%）")
            if pred.sensitivity:
                lines.append(f"- 敏感性分析：{pred.sensitivity.method}")
            lines.append("- What-If 分析：关键变量 +-20% 变化模拟")

        return "\n".join(lines)

    def _build_assumptions_section(self) -> str:
        """构建假设条件章节

        Returns:
            str: 假设条件文本
        """
        lines: list[str] = ["## 假设条件"]

        assumptions = [
            "数据样本能够代表总体分布",
            "变量间的线性关系假设（因果效应估计基于线性回归）",
            "无未观测的混杂因子（因果分析的识别假设）",
            "异常值检测基于 IQR 方法，假设数据近似正态分布",
            "预测模拟基于线性因果效应，不考虑非线性交互",
            "相关性不等于因果性，因果方向基于统计推断而非实验验证",
        ]

        for i, assumption in enumerate(assumptions, 1):
            lines.append(f"{i}. {assumption}")

        return "\n".join(lines)

    def _build_confidence_section(
        self,
        causal: CausalResult | None,
        pred: PredictionResult | None,
    ) -> str:
        """构建置信区间汇总章节

        Args:
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）

        Returns:
            str: 置信区间汇总文本
        """
        lines: list[str] = ["## 置信区间汇总"]

        has_content = False

        if causal and causal.causal_effects:
            lines.append("\n### 因果效应置信区间")
            for eff in causal.causal_effects:
                if eff.confidence_interval:
                    has_content = True
                    lines.append(
                        f"- {eff.treatment} -> {eff.outcome}: "
                        f"效应={eff.effect_size}, "
                        f"95%CI=[{eff.confidence_interval[0]}, {eff.confidence_interval[1]}], "
                        f"p={eff.p_value}"
                    )

        if pred and pred.what_ifs:
            lines.append("\n### What-If 置信度")
            for wi in pred.what_ifs[:5]:
                has_content = True
                lines.append(
                    f"- {wi.variable}: "
                    f"变化={wi.outcome_change}, 置信度={wi.confidence}"
                )

        if not has_content:
            lines.append("\n当前分析未产生置信区间数据。")

        return "\n".join(lines)

    def _build_limitations_section(self) -> str:
        """构建局限性说明章节

        Returns:
            str: 局限性说明文本
        """
        lines: list[str] = ["## 局限性说明"]

        limitations = [
            "因果分析基于观察数据，因果关系的确认需要随机对照实验或准实验设计",
            "线性回归假设可能无法捕捉变量间的非线性关系",
            "因果图发现基于相关性推导，可能遗漏真实的因果关系或引入虚假因果",
            "预测模拟基于历史数据的线性外推，对未来结构性变化的预测能力有限",
            "异常值检测的 IQR 方法对非正态分布数据可能不够准确",
            "敏感性分析假设各变量独立变化，未考虑变量间的交互效应",
        ]

        for i, limitation in enumerate(limitations, 1):
            lines.append(f"{i}. {limitation}")

        return "\n".join(lines)

    def _collect_charts(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
    ) -> list[Chart]:
        """收集图表数据

        从各分析结果中提取图表定义，包括：
        - 变量重要性柱状图数据
        - 相关性热力图数据
        - 因果图数据
        - 情景对比图数据

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）

        Returns:
            list[Chart]: 图表定义列表
        """
        charts: list[Chart] = []

        # 变量重要性柱状图
        if desc.variable_importance:
            top_importance = desc.variable_importance[:15]
            charts.append(Chart(
                type="bar",
                title="变量重要性排名",
                data={
                    "labels": [item.name for item in top_importance],
                    "values": [item.score for item in top_importance],
                    "orientation": "horizontal",
                },
                description="基于互信息或方差贡献的变量重要性排名",
            ))

        # 相关性热力图
        if desc.correlations:
            # 提取所有涉及的变量
            vars_set: set[str] = set()
            for corr in desc.correlations[:20]:
                vars_set.add(corr.var1)
                vars_set.add(corr.var2)
            vars_list = sorted(vars_set)

            # 构建矩阵
            matrix = []
            for v1 in vars_list:
                row = []
                for v2 in vars_list:
                    if v1 == v2:
                        row.append(1.0)
                    else:
                        # 查找相关系数
                        coeff = 0.0
                        for corr in desc.correlations:
                            if (corr.var1 == v1 and corr.var2 == v2) or \
                               (corr.var1 == v2 and corr.var2 == v1):
                                coeff = corr.coefficient
                                break
                        row.append(round(coeff, 4))
                matrix.append(row)

            charts.append(Chart(
                type="heatmap",
                title="变量相关性热力图",
                data={
                    "variables": vars_list,
                    "matrix": matrix,
                },
                description="变量间 Pearson/Spearman 相关系数矩阵",
            ))

        # 因果图
        if causal and causal.causal_graph and causal.causal_graph.edges:
            nodes = causal.causal_graph.nodes
            edges_data = [
                {
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "effect_size": edge.effect_size,
                    "p_value": edge.p_value,
                }
                for edge in causal.causal_graph.edges
            ]
            charts.append(Chart(
                type="graph",
                title="因果假设图",
                data={
                    "nodes": nodes,
                    "edges": edges_data,
                    "method": causal.causal_graph.method,
                },
                description="基于统计推断的因果假设图",
            ))

        # 情景对比图
        if pred and len(pred.scenarios) >= 3:
            target_key = next(iter(pred.scenarios[0].predicted_outcomes), None)
            if target_key:
                charts.append(Chart(
                    type="bar",
                    title="情景对比",
                    data={
                        "labels": [s.name for s in pred.scenarios],
                        "values": [
                            s.predicted_outcomes.get(target_key, 0)
                            for s in pred.scenarios
                        ],
                        "target_variable": target_key,
                    },
                    description="乐观/中性/悲观三种场景下的目标变量预测值对比",
                ))

        # 敏感性分析图
        if pred and pred.sensitivity and pred.sensitivity.entries:
            top_sens = pred.sensitivity.entries[:10]
            charts.append(Chart(
                type="bar",
                title="敏感性分析",
                data={
                    "labels": [e.variable for e in top_sens],
                    "values": [e.sensitivity_score for e in top_sens],
                    "directions": [e.direction for e in top_sens],
                    "orientation": "horizontal",
                },
                description="各变量对目标结果的敏感性得分",
            ))

        return charts

"""
报告生成器模块

负责将描述性分析、因果分析和预测模拟的结果汇总，
生成分层的可解释性报告，包含执行摘要、详细分析、技术附录和图表数据。

支持按受众（executive / business / analyst / both / technical）生成不同风格的报告：
- executive/business：业务语言，无技术术语，聚焦决策含义
- analyst/both：适中深度，关键指标带数值
- technical：完整统计细节，包含方法说明和假设条件
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from core.models import (
    AnalysisInput,
    BusinessUnderstanding,
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

    根据受众类型自动调整语言风格和内容深度：
    - executive/business：咨询顾问风格，无技术术语
    - analyst/both：适中深度，关键指标带数值但不啰嗦
    - technical：完整统计细节

    Args:
        llm_client: LLM 客户端实例，为 None 时使用模板回退生成报告内容
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """初始化报告生成器

        Args:
            llm_client: 可选的 LLM 客户端，用于生成更自然的报告文本
        """
        self.llm = llm_client

    # ================================================================
    # 公共方法
    # ================================================================

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
        business_understanding = analysis_input.business_understanding

        logger.info("开始生成报告，领域: %s，受众: %s", domain, audience)

        # 1. 生成执行摘要（面向业务决策者）
        executive_summary = self._generate_executive_summary(
            desc=desc_result,
            causal=causal_result,
            pred=pred_result,
            domain=domain,
            audience=audience,
            business_understanding=business_understanding,
        )
        logger.info("执行摘要生成完成，长度: %d 字", len(executive_summary))

        # 2. 生成详细分析（面向业务+技术）
        detailed_analysis = self._generate_detailed_analysis(
            desc=desc_result,
            causal=causal_result,
            pred=pred_result,
            domain=domain,
            audience=audience,
            business_understanding=business_understanding,
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

    # ================================================================
    # 执行摘要生成
    # ================================================================

    def _generate_executive_summary(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        domain: str,
        audience: str,
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """生成执行摘要

        根据受众类型采用不同风格：
        - executive/business：咨询顾问风格，无技术术语，聚焦"所以呢？"
        - technical：保留所有统计细节
        - analyst/both：适中深度

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            domain: 业务领域
            audience: 目标受众
            business_understanding: 业务理解信息（可选）

        Returns:
            str: 执行摘要文本
        """
        is_business = audience in ("executive", "business")

        # 收集关键发现和行动建议
        findings = self._extract_key_findings(desc, causal, pred)
        recommendations = self._extract_recommendations(desc, causal, pred)

        # 尝试使用 LLM 生成更自然的摘要
        if self.llm is not None:
            try:
                return self._llm_executive_summary(
                    desc=desc,
                    causal=causal,
                    pred=pred,
                    domain=domain,
                    audience=audience,
                    findings=findings,
                    recommendations=recommendations,
                    business_understanding=business_understanding,
                )
            except Exception as e:
                logger.warning("LLM 执行摘要生成失败，回退到模板: %s", e)

        # 模板回退
        if is_business:
            return self._template_executive_summary_business(
                desc=desc,
                causal=causal,
                pred=pred,
                findings=findings,
                recommendations=recommendations,
                domain=domain,
                business_understanding=business_understanding,
            )
        elif audience == "technical":
            return self._template_executive_summary_technical(
                findings=findings,
                recommendations=recommendations,
                domain=domain,
            )
        else:
            # analyst / both
            return self._template_executive_summary_analyst(
                findings=findings,
                recommendations=recommendations,
                domain=domain,
            )

    def _llm_executive_summary(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        domain: str,
        audience: str,
        findings: list[str],
        recommendations: list[str],
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """使用 LLM 生成执行摘要

        根据受众类型构建不同的 prompt，确保输出风格匹配。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            domain: 业务领域
            audience: 目标受众
            findings: 关键发现列表
            recommendations: 行动建议列表
            business_understanding: 业务理解信息（可选）

        Returns:
            str: LLM 生成的执行摘要文本
        """
        from llm.prompts import PromptTemplates

        # 构建素材
        material = self._build_executive_material(findings, recommendations, domain)

        # 业务理解上下文
        bu_context = ""
        if business_understanding:
            bu_parts = []
            if business_understanding.inferred_scenario:
                bu_parts.append(f"业务场景: {business_understanding.inferred_scenario}")
            if business_understanding.key_metrics:
                bu_parts.append(f"关键指标: {business_understanding.key_metrics}")
            if business_understanding.causal_hypotheses:
                bu_parts.append(f"因果假设: {business_understanding.causal_hypotheses}")
            if business_understanding.business_question_answer:
                bu_parts.append(
                    f"业务问题回答: {business_understanding.business_question_answer}"
                )
            if bu_parts:
                bu_context = "\n\n## 业务理解\n" + "\n".join(bu_parts)

        if audience in ("executive", "business"):
            prompt = (
                f"## 任务\n"
                f"你是一位资深咨询顾问，请基于以下分析素材，为高层决策者撰写一份简明的执行摘要。\n\n"
                f"## 业务领域\n{domain or '通用'}\n"
                f"{bu_context}\n\n"
                f"## 分析素材\n{material}\n\n"
                f"## 输出要求\n"
                f"1. **绝对禁止**出现任何技术术语：p值、置信区间、IQR、偏度、标准差、"
                f"相关系数、ATE、回归系数等统计学术语一律不得出现\n"
                f"2. 用业务语言表达：如\u201c影响显著\u201d而非\u201cp<0.05\u201d，\u201c波动较大\u201d而非\u201c标准差=XX\u201d\n"
                f"3. 每个发现后面必须跟业务含义（\u201c所以呢？\u201d）\n"
            )
            if business_understanding and business_understanding.business_question_answer:
                prompt += (
                    f"4. 先直接回答用户的业务问题，然后列出核心发现\n"
                )
            else:
                prompt += (
                    f"4. 先给出总体结论，然后列出核心发现\n"
                )
            prompt += (
                f"5. 格式：先回答问题（如有），然后 3 条核心发现，然后 3 条行动建议\n"
                f"6. 语言风格：像咨询顾问给老板的 brief，不是数据报告\n"
                f"7. 控制在 400 字以内\n"
                f"8. 直接输出摘要内容，不要加标题"
            )
        elif audience == "technical":
            prompt = (
                f"## 任务\n"
                f"请基于以下分析素材，生成一份面向技术团队的执行摘要。\n\n"
                f"## 业务领域\n{domain or '通用'}\n"
                f"{bu_context}\n\n"
                f"## 分析素材\n{material}\n\n"
                f"## 输出要求\n"
                f"1. 提取最重要的 3-5 条关键发现，保留完整统计细节\n"
                f"2. 给出 3-5 条可操作的行动建议\n"
                f"3. 包含方法说明和假设条件\n"
                f"4. 控制在 500 字以内\n"
                f"5. 直接输出摘要内容，不要加标题"
            )
        else:
            # analyst / both
            prompt = (
                f"## 任务\n"
                f"请基于以下分析素材，生成一份面向数据分析师的执行摘要。\n\n"
                f"## 业务领域\n{domain or '通用'}\n"
                f"{bu_context}\n\n"
                f"## 分析素材\n{material}\n\n"
                f"## 输出要求\n"
                f"1. 提取最重要的 3-5 条关键发现，关键指标带数值\n"
                f"2. 给出 3-5 条可操作的行动建议\n"
                f"3. 使用清晰的分析语言，避免过于啰嗦\n"
                f"4. 控制在 400 字以内\n"
                f"5. 直接输出摘要内容，不要加标题"
            )

        summary = self.llm.generate(
            prompt=prompt,
            system_prompt=PromptTemplates.REPORT_SYSTEM,
        )
        return summary.strip()

    # ================================================================
    # 模板回退：执行摘要
    # ================================================================

    def _template_executive_summary_business(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        findings: list[str],
        recommendations: list[str],
        domain: str,
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """业务版执行摘要模板（无 LLM 时的回退方案）

        咨询顾问风格，完全不出现技术术语，聚焦业务含义和行动建议。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            findings: 关键发现列表
            recommendations: 行动建议列表
            domain: 业务领域
            business_understanding: 业务理解信息（可选）

        Returns:
            str: 业务版执行摘要文本
        """
        parts: list[str] = []

        # --- 开头：回答业务问题或总体结论 ---
        if business_understanding and business_understanding.business_question_answer:
            parts.append(business_understanding.business_question_answer)
        else:
            domain_prefix = f"「{domain}」" if domain else "本次"
            parts.append(
                f"本次分析围绕{domain_prefix}场景下的数据展开，"
                f"共分析了 {len(desc.distributions)} 个关键变量。"
            )

        # --- 数据概况（业务语言） ---
        if desc.distributions:
            var_count = len(desc.distributions)
            parts.append(
                f"数据共涵盖 {var_count} 个分析变量。"
            )

        # --- 核心发现（3 条，用业务语言） ---
        business_findings = self._translate_findings_to_business(
            desc, causal, pred, findings
        )
        if business_findings:
            finding_lines = []
            for i, f in enumerate(business_findings[:3], 1):
                finding_lines.append(f"{i}. {f}")
            parts.append("\n核心发现：\n" + "\n".join(finding_lines))

        # --- 行动建议（3 条） ---
        business_recs = self._translate_recommendations_to_business(
            desc, causal, pred, recommendations
        )
        if business_recs:
            rec_lines = []
            for i, r in enumerate(business_recs[:3], 1):
                rec_lines.append(f"{i}. {r}")
            parts.append("\n建议关注：\n" + "\n".join(rec_lines))

        return "\n\n".join(parts)

    def _template_executive_summary_technical(
        self,
        findings: list[str],
        recommendations: list[str],
        domain: str,
    ) -> str:
        """技术版执行摘要模板（无 LLM 时的回退方案）

        保留完整统计细节，包含方法说明和假设条件。

        Args:
            findings: 关键发现列表
            recommendations: 行动建议列表
            domain: 业务领域

        Returns:
            str: 技术版执行摘要文本
        """
        parts: list[str] = []

        domain_prefix = f"「{domain}」" if domain else "本次"

        # 关键发现（保留技术细节）
        if findings:
            finding_text = "\n".join(f"- {f}" for f in findings[:5])
            parts.append(
                f"在{domain_prefix}的数据分析中，主要发现：\n\n{finding_text}"
            )
        else:
            parts.append(f"在{domain_prefix}的数据分析中，未发现显著的模式或异常。")

        # 行动建议
        if recommendations:
            rec_text = "\n".join(f"- {r}" for r in recommendations[:5])
            parts.append(f"\n行动建议：\n\n{rec_text}")

        # 方法说明
        parts.append(
            "\n分析方法：描述性统计（均值、中位数、标准差、偏度）、"
            "相关性分析（Pearson/Spearman）、异常值检测（IQR 方法）。"
            "因果分析基于 OLS 线性回归，预测模拟基于线性因果效应外推。"
        )

        return "\n\n".join(parts)

    def _template_executive_summary_analyst(
        self,
        findings: list[str],
        recommendations: list[str],
        domain: str,
    ) -> str:
        """分析师版执行摘要模板（无 LLM 时的回退方案）

        适中深度，关键指标带数值但不啰嗦。

        Args:
            findings: 关键发现列表
            recommendations: 行动建议列表
            domain: 业务领域

        Returns:
            str: 分析师版执行摘要文本
        """
        parts: list[str] = []

        domain_prefix = f"「{domain}」" if domain else "本次"

        # 关键发现
        if findings:
            finding_text = "；".join(findings[:3])
            parts.append(
                f"在{domain_prefix}的数据分析中，主要发现：{finding_text}。"
            )
        else:
            parts.append(f"在{domain_prefix}的数据分析中，未发现显著的模式或异常。")

        # 行动建议
        if recommendations:
            rec_text = "；".join(recommendations[:3])
            parts.append(f"建议：{rec_text}")

        return "\n\n".join(parts)

    # ================================================================
    # 业务语言翻译工具
    # ================================================================

    @staticmethod
    def _translate_findings_to_business(
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        raw_findings: list[str],
    ) -> list[str]:
        """将技术发现翻译为业务语言

        去除统计学术语，聚焦业务含义和影响。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            raw_findings: 原始技术发现列表

        Returns:
            list[str]: 业务语言版本的发现列表
        """
        business_findings: list[str] = []

        # 最重要的变量
        if desc.variable_importance:
            top_var = desc.variable_importance[0]
            business_findings.append(
                f"「{top_var.name}」是影响整体表现的最关键因素，"
                f"其变化直接驱动了核心业务结果，应作为首要关注点。"
            )

        # 强相关性
        strong_corrs = [c for c in desc.correlations if abs(c.coefficient) > 0.5]
        if strong_corrs:
            c = strong_corrs[0]
            strength = "密切" if abs(c.coefficient) > 0.7 else "明显"
            direction = "同步增长" if c.coefficient > 0 else "此消彼长"
            business_findings.append(
                f"「{c.var1}」与「{c.var2}」之间存在{strength}的{direction}关系，"
                f"提示两者可能受到共同因素驱动或存在直接的业务关联。"
            )

        # 异常值
        if desc.anomalies:
            col_counts = Counter(a.column for a in desc.anomalies)
            top_anomaly_col, count = col_counts.most_common(1)[0]
            business_findings.append(
                f"数据中检测到 {count} 个异常记录，主要集中在「{top_anomaly_col}」，"
                f"可能反映了数据录入错误或真实的业务异常情况，建议核实。"
            )

        # 因果分析
        if causal and causal.causal_effects:
            significant = [
                e for e in causal.causal_effects
                if e.p_value is not None and e.p_value < 0.05
            ]
            if significant:
                eff = significant[0]
                if eff.effect_size > 0:
                    business_findings.append(
                        f"「{eff.treatment}」对「{eff.outcome}」有显著的正向推动作用，"
                        f"提升该指标将直接带动业务结果改善。"
                    )
                else:
                    business_findings.append(
                        f"「{eff.treatment}」对「{eff.outcome}」存在明显的抑制作用，"
                        f"需要关注并考虑优化该指标。"
                    )

        # 预测模拟
        if pred and len(pred.scenarios) >= 3:
            target_key = next(iter(pred.scenarios[0].predicted_outcomes), None)
            if target_key:
                opt_val = pred.scenarios[0].predicted_outcomes.get(target_key, 0)
                pes_val = pred.scenarios[2].predicted_outcomes.get(target_key, 0)
                business_findings.append(
                    f"在不同假设场景下，「{target_key}」的波动范围较大"
                    f"（乐观 {opt_val} 至悲观 {pes_val}），"
                    f"建议制定应对不同情况的预案。"
                )

        return business_findings[:3]

    @staticmethod
    def _translate_recommendations_to_business(
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        raw_recommendations: list[str],
    ) -> list[str]:
        """将技术建议翻译为业务行动建议

        用决策导向的语言替代技术性建议。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            raw_recommendations: 原始技术建议列表

        Returns:
            list[str]: 业务语言版本的建议列表
        """
        business_recs: list[str] = []

        # 基于最重要变量的建议
        if desc.variable_importance:
            top_var = desc.variable_importance[0]
            business_recs.append(
                f"优先关注「{top_var.name}」的优化空间，其对最终结果的影响最大，"
                f"投入资源改善该指标将获得最高的回报。"
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
                    business_recs.append(
                        f"加大「{top_effect.treatment}」的投入力度，"
                        f"数据表明这能有效提升「{top_effect.outcome}」。"
                    )
                else:
                    business_recs.append(
                        f"控制「{top_effect.treatment}」的水平，"
                        f"数据表明该指标过高会拖累「{top_effect.outcome}」的表现。"
                    )

        # 基于预测模拟的建议
        if pred and pred.sensitivity and pred.sensitivity.entries:
            top_sensitive = pred.sensitivity.entries[0]
            direction = "提升" if top_sensitive.direction == "positive" else "控制"
            business_recs.append(
                f"重点{direction}「{top_sensitive.variable}」，"
                f"该因素对目标结果的影响最为敏感，小幅调整即可带来显著变化。"
            )

        # 基于异常值的建议
        if desc.anomalies:
            business_recs.append(
                "对检测到的异常数据进行排查，确认是否为录入错误或真实的业务波动，"
                "以确保后续分析结论的可靠性。"
            )

        return business_recs[:3]

    # ================================================================
    # 关键发现与行动建议提取
    # ================================================================

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

    def _extract_recommendations(
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

        lines.append("### 关键发现")
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. {finding}")

        lines.append("\n### 行动建议")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    # ================================================================
    # 详细分析生成
    # ================================================================

    def _generate_detailed_analysis(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
        domain: str,
        audience: str,
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """生成详细分析

        根据受众类型调整详细分析的深度和风格：
        - executive/business：简化版，只有核心发现，用叙述性语言
        - analyst/both/technical：完整分析内容

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）
            domain: 业务领域
            audience: 目标受众
            business_understanding: 业务理解信息（可选）

        Returns:
            str: 详细分析文本
        """
        is_business = audience in ("executive", "business")

        # 业务背景理解（analyst/both/technical 受众，有 business_understanding 时）
        bu_intro = ""
        if business_understanding and not is_business:
            bu_parts = []
            if business_understanding.inferred_scenario:
                bu_parts.append(f"**业务场景**：{business_understanding.inferred_scenario}")
            if business_understanding.key_metrics:
                bu_parts.append(f"**关键指标**：{business_understanding.key_metrics}")
            if business_understanding.causal_hypotheses:
                bu_parts.append(f"**因果假设**：{business_understanding.causal_hypotheses}")
            if business_understanding.data_characteristics:
                bu_parts.append(f"**数据特征**：{business_understanding.data_characteristics}")
            if business_understanding.analysis_suggestions:
                bu_parts.append(f"**分析建议**：{business_understanding.analysis_suggestions}")
            if bu_parts:
                bu_intro = "## 业务背景理解\n\n" + "\n\n".join(bu_parts) + "\n\n"

        if is_business:
            # 业务版：简化版，叙述性语言
            sections = self._build_business_detailed_sections(desc, causal, pred)
            raw_content = bu_intro + "\n\n".join(sections)
        else:
            # analyst/both/technical：完整分析
            sections: list[str] = []

            # 变量分析章节
            sections.append(self._build_variable_section(desc, audience))

            # 因果链分析章节
            if causal:
                sections.append(self._build_causal_section(causal, audience))

            # 预测情景章节
            if pred:
                sections.append(self._build_predictive_section(pred, audience))

            raw_content = bu_intro + "\n\n".join(sections)

        # 尝试使用 LLM 润色
        if self.llm is not None:
            try:
                from llm.prompts import PromptTemplates

                if is_business:
                    prompt = (
                        f"## 任务\n"
                        f"请基于以下分析素材，生成一段面向业务管理者的叙述性分析。\n\n"
                        f"## 业务领域\n{domain or '通用'}\n\n"
                        f"## 受众要求\n"
                        f"请用流畅的叙述性语言，不要使用列表罗列。\n"
                        f"绝对禁止出现技术术语（p值、置信区间、相关系数等）。\n"
                        f"聚焦每个发现的业务含义和决策启示。\n\n"
                        f"## 分析素材\n{raw_content}\n\n"
                        f"## 输出要求\n"
                        f"1. 使用段落式叙述，而非列表\n"
                        f"2. 每段聚焦一个主题\n"
                        f"3. 突出业务含义和行动建议"
                    )
                elif audience == "technical":
                    prompt = (
                        f"## 任务\n"
                        f"请基于以下分析素材，生成一份面向技术团队的详细分析报告。\n\n"
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
                else:
                    prompt = (
                        f"## 任务\n"
                        f"请基于以下分析素材，生成一份面向数据分析师的详细分析报告。\n\n"
                        f"## 业务领域\n{domain or '通用'}\n\n"
                        f"## 受众要求\n"
                        f"请提供适中的分析深度，关键指标带数值但不啰嗦。\n\n"
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

        return raw_content

    def _build_business_detailed_sections(
        self,
        desc: DescriptionResult,
        causal: CausalResult | None,
        pred: PredictionResult | None,
    ) -> list[str]:
        """构建业务版详细分析的各个章节

        使用叙述性语言，不出现技术术语，聚焦核心发现和业务含义。

        Args:
            desc: 描述性分析结果
            causal: 因果分析结果（可选）
            pred: 预测模拟结果（可选）

        Returns:
            list[str]: 章节文本列表
        """
        sections: list[str] = []

        # --- 核心发现叙述 ---
        narrative_parts: list[str] = []

        if desc.variable_importance:
            top_vars = desc.variable_importance[:3]
            var_names = "、".join(f"「{v.name}」" for v in top_vars)
            narrative_parts.append(
                f"在本次分析的所有变量中，{var_names} 是最值得关注的几个指标，"
                f"它们对整体业务表现有着不同程度的影响。"
            )

        strong_corrs = [c for c in desc.correlations if abs(c.coefficient) > 0.5]
        if strong_corrs:
            c = strong_corrs[0]
            relationship = "正相关（同涨同跌）" if c.coefficient > 0 else "负相关（此消彼长）"
            narrative_parts.append(
                f"值得注意的是，「{c.var1}」与「{c.var2}」之间存在{relationship}，"
                f"这意味着调整其中一个指标时，需要同时考虑对另一个的影响。"
            )

        if desc.anomalies:
            col_counts = Counter(a.column for a in desc.anomalies)
            top_col, count = col_counts.most_common(1)[0]
            narrative_parts.append(
                f"此外，在「{top_col}」中发现了 {count} 条异常记录，"
                f"这些异常可能对应着特定的业务事件或数据问题，值得进一步排查。"
            )

        if narrative_parts:
            sections.append("## 核心发现\n\n" + "\n\n".join(narrative_parts))

        # --- 因果洞察（业务语言） ---
        if causal and causal.causal_effects:
            significant = [
                e for e in causal.causal_effects
                if e.p_value is not None and e.p_value < 0.05
            ]
            if significant:
                causal_parts: list[str] = []
                for eff in significant[:3]:
                    if eff.effect_size > 0:
                        causal_parts.append(
                            f"「{eff.treatment}」的提升会带动「{eff.outcome}」的增长，"
                            f"是值得投入资源优化的方向。"
                        )
                    else:
                        causal_parts.append(
                            f"「{eff.treatment}」的升高会拖累「{eff.outcome}」的表现，"
                            f"需要考虑适当控制该指标。"
                        )
                sections.append("## 因果洞察\n\n" + "\n\n".join(causal_parts))

        # --- 情景展望（业务语言） ---
        if pred and len(pred.scenarios) >= 3:
            scenario_parts: list[str] = []
            target_key = next(iter(pred.scenarios[0].predicted_outcomes), None)
            if target_key:
                opt_val = pred.scenarios[0].predicted_outcomes.get(target_key, 0)
                base_val = pred.scenarios[1].predicted_outcomes.get(target_key, 0) if len(pred.scenarios) > 1 else opt_val
                pes_val = pred.scenarios[2].predicted_outcomes.get(target_key, 0)
                scenario_parts.append(
                    f"如果各项指标保持当前水平，「{target_key}」预计为 {base_val}。"
                    f"在较为乐观的假设下可达到 {opt_val}，"
                    f"而如果出现不利因素则可能降至 {pes_val}。"
                )
            if pred.sensitivity and pred.sensitivity.entries:
                top_sens = pred.sensitivity.entries[0]
                scenario_parts.append(
                    f"在所有影响因素中，「{top_sens.variable}」的变化对结果最为敏感，"
                    f"建议在决策时优先考虑该指标的不确定性。"
                )
            if scenario_parts:
                sections.append("## 情景展望\n\n" + "\n\n".join(scenario_parts))

        return sections

    # ================================================================
    # 各章节构建（区分受众）
    # ================================================================

    def _build_variable_section(
        self,
        desc: DescriptionResult,
        audience: str = "analyst",
    ) -> str:
        """构建变量分析章节

        根据受众类型调整展示风格：
        - business/executive：自然语言描述趋势和含义
        - analyst/both：关键指标带数值
        - technical：完整数值表格

        Args:
            desc: 描述性分析结果
            audience: 目标受众

        Returns:
            str: 变量分析章节文本
        """
        lines: list[str] = ["## 变量分析"]

        is_business = audience in ("executive", "business")

        # 变量重要性
        if desc.variable_importance:
            lines.append("\n### 变量重要性排名")
            if is_business:
                top_vars = desc.variable_importance[:5]
                var_desc = "、".join(f"「{v.name}」" for v in top_vars)
                lines.append(
                    f"以下变量对整体业务表现影响最大：{var_desc}。"
                    f"其中「{top_vars[0].name}」是最关键的因素。"
                )
            else:
                for item in desc.variable_importance[:10]:
                    lines.append(
                        f"- #{item.rank} {item.name}: 重要性得分 {item.score}"
                    )

        # 分布特征
        if desc.distributions:
            lines.append("\n### 变量分布特征")
            if is_business:
                # 业务版：用自然语言描述
                for dist in desc.distributions[:5]:
                    desc_parts = [f"「{dist.column}」"]
                    if dist.mean is not None and dist.median is not None:
                        if abs(dist.mean - dist.median) / max(abs(dist.mean), 0.01) > 0.1:
                            desc_parts.append("分布不太对称")
                        else:
                            desc_parts.append("分布相对均匀")
                    if dist.outlier_count > 0:
                        desc_parts.append(f"存在 {dist.outlier_count} 条异常记录")
                    if dist.min is not None and dist.max is not None:
                        desc_parts.append(f"取值范围从 {dist.min} 到 {dist.max}")
                    lines.append("- " + "，".join(desc_parts) + "。")
            else:
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
            if is_business:
                for corr in desc.correlations[:5]:
                    if abs(corr.coefficient) > 0.7:
                        strength = "非常密切"
                    elif abs(corr.coefficient) > 0.5:
                        strength = "较为明显"
                    else:
                        strength = "较弱"
                    direction = "同向变化" if corr.coefficient > 0 else "反向变化"
                    lines.append(
                        f"- 「{corr.var1}」与「{corr.var2}」存在{strength}的{direction}关系"
                    )
            else:
                for corr in desc.correlations[:10]:
                    direction = "正相关" if corr.coefficient > 0 else "负相关"
                    lines.append(
                        f"- {corr.var1} 与 {corr.var2}: "
                        f"{direction}（{corr.method}, r={corr.coefficient}）"
                    )

        return "\n".join(lines)

    def _build_causal_section(
        self,
        causal: CausalResult,
        audience: str = "analyst",
    ) -> str:
        """构建因果链分析章节

        根据受众类型调整展示风格：
        - business/executive：用"X 影响 Y"的自然语言
        - analyst/both：适中细节
        - technical：保留完整统计（ATE、p值、置信区间）

        Args:
            causal: 因果分析结果
            audience: 目标受众

        Returns:
            str: 因果链分析章节文本
        """
        lines: list[str] = ["## 因果链分析"]
        is_business = audience in ("executive", "business")

        # 因果图
        if causal.causal_graph and causal.causal_graph.edges:
            lines.append(f"\n### 因果假设图（方法: {causal.causal_graph.method}）")
            for edge in causal.causal_graph.edges:
                if is_business:
                    direction = "推动" if edge.effect_size > 0 else "抑制"
                    lines.append(
                        f"- 「{edge.from_node}」{direction}了「{edge.to_node}」"
                    )
                else:
                    lines.append(
                        f"- {edge.from_node} -> {edge.to_node}: "
                        f"效应={edge.effect_size}, p值={edge.p_value}"
                    )

        # 因果效应
        if causal.causal_effects:
            lines.append("\n### 因果效应估计")
            for eff in causal.causal_effects:
                if is_business:
                    if eff.effect_size > 0:
                        lines.append(
                            f"- 「{eff.treatment}」的提升会带动「{eff.outcome}」的增长"
                        )
                    else:
                        lines.append(
                            f"- 「{eff.treatment}」的升高会拖累「{eff.outcome}」的表现"
                        )
                else:
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
                if is_business:
                    direction = "提升" if change > 0 else "下降"
                    lines.append(
                        f"- 若「{cf.treatment}」从 {cf.original_value} "
                        f"调整为 {cf.counterfactual_value}，"
                        f"结果将{direction}约 {abs(change):.2f}"
                    )
                else:
                    lines.append(
                        f"- 若「{cf.treatment}」从 {cf.original_value} "
                        f"变为 {cf.counterfactual_value}，"
                        f"结果变化 {change:+.4f}"
                    )

        return "\n".join(lines)

    def _build_predictive_section(
        self,
        pred: PredictionResult,
        audience: str = "analyst",
    ) -> str:
        """构建预测情景章节

        根据受众类型调整展示风格：
        - business/executive：聚焦"如果...会怎样"的决策含义
        - analyst/both：适中细节
        - technical：保留完整数值

        Args:
            pred: 预测模拟结果
            audience: 目标受众

        Returns:
            str: 预测情景章节文本
        """
        lines: list[str] = ["## 预测情景分析"]
        is_business = audience in ("executive", "business")

        # 场景模拟
        if pred.scenarios:
            lines.append("\n### 场景模拟结果")
            for scenario in pred.scenarios:
                if scenario.predicted_outcomes:
                    if is_business:
                        outcome_items = []
                        for k, v in scenario.predicted_outcomes.items():
                            outcome_items.append(f"「{k}」预计为 {v}")
                        lines.append(
                            f"- **{scenario.name}**：{scenario.description}，"
                            + "，".join(outcome_items)
                        )
                    else:
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
                if is_business:
                    direction_map = {
                        "positive": "正向推动",
                        "negative": "反向拖累",
                        "neutral": "影响不大",
                    }
                    direction_str = direction_map.get(entry.direction, entry.direction)
                    lines.append(
                        f"- 「{entry.variable}」对结果有{direction_str}作用"
                    )
                else:
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
                if is_business:
                    direction = "提升" if (wi.outcome_change and wi.outcome_change > 0) else "下降"
                    change_val = abs(wi.outcome_change) if wi.outcome_change else 0
                    lines.append(
                        f"- 如果将「{wi.variable}」从 {wi.original_value} "
                        f"调整为 {wi.new_value}，"
                        f"结果将{direction}约 {change_val:.2f}"
                    )
                else:
                    lines.append(
                        f"- 「{wi.variable}」{wi.original_value} -> {wi.new_value}，"
                        f"结果变化={wi.outcome_change}, 置信度={wi.confidence}"
                    )

        return "\n".join(lines)

    # ================================================================
    # 技术附录生成
    # ================================================================

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

    # ================================================================
    # 图表数据收集
    # ================================================================

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

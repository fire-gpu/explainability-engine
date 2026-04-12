"""
Prompt 模板管理模块

集中管理各分析阶段的系统提示词和用户提示词模板，
确保 LLM 输出风格一致且符合业务场景需求。
支持 L1 数据预扫描，将业务理解注入后续分析流程。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import BusinessUnderstanding


class PromptTemplates:
    """Prompt 模板管理器

    提供数据预扫描、描述性分析、因果解释、预测解释和报告生成五个阶段的
    系统提示词常量及用户提示词生成方法。
    支持根据业务理解（BusinessUnderstanding）动态调整 prompt 风格。
    """

    # ================================================================
    # 系统提示词
    # ================================================================

    SCAN_SYSTEM: str = (
        "你是一个资深业务分析师和数据科学家，擅长快速理解数据的业务含义。"
        "当你看到一组数据时，你能：\n"
        "1. 快速判断这可能是什么业务场景\n"
        "2. 识别出哪些是关键业务指标\n"
        "3. 提出合理的因果假设\n"
        "4. 给出有针对性的分析建议\n"
        "请基于数据事实进行分析，不要编造信息。"
    )

    DESCRIPTIVE_SYSTEM: str = (
        "你是一个数据分析专家，擅长从数据中提取关键洞察并生成清晰易懂的分析叙述。"
        "你的分析应当：\n"
        "1. 基于数据事实，不编造信息\n"
        "2. 使用简洁明了的语言\n"
        "3. 突出最重要的发现\n"
        "4. 适当使用数据支撑论点\n"
        "5. 按照逻辑顺序组织内容"
    )

    CAUSAL_SYSTEM: str = (
        "你是一个因果推断专家，擅长识别变量之间的因果关系并解释因果机制。"
        "你的分析应当：\n"
        "1. 明确区分相关性和因果性\n"
        "2. 基于因果图和效应估计进行解释\n"
        "3. 说明因果假设和局限性\n"
        "4. 使用通俗易懂的语言解释专业概念\n"
        "5. 提供可操作的建议"
    )

    PREDICTIVE_SYSTEM: str = (
        "你是一个预测分析专家，擅长解释预测模型的行为和模拟结果。"
        "你的分析应当：\n"
        "1. 清晰解释预测结果的含义\n"
        "2. 说明预测的置信度和不确定性\n"
        "3. 对比不同场景的差异\n"
        "4. 解释关键驱动因素\n"
        "5. 提供合理的决策建议"
    )

    REPORT_SYSTEM: str = (
        "你是一个报告撰写专家，擅长将技术分析结果转化为不同受众可理解的报告。"
        "你的报告应当：\n"
        "1. 结构清晰、层次分明\n"
        "2. 根据受众调整语言风格和专业深度\n"
        "3. 突出关键发现和建议\n"
        "4. 使用图表和数据进行支撑\n"
        "5. 保持客观、专业的语气"
    )

    # ================================================================
    # 用户提示词生成方法
    # ================================================================

    @staticmethod
    def scan_prompt(
        data_summary: str,
        business_question: str = "",
        business_context: str = "",
    ) -> str:
        """生成数据预扫描的 prompt

        让 LLM 快速理解数据并推断业务语义，返回结构化的 JSON 结果。

        Args:
            data_summary: 数据摘要文本（列名、类型、统计量、采样数据）
            business_question: 用户输入的关键业务问题
            business_context: 用户描述的业务场景上下文

        Returns:
            str: 完整的用户提示词
        """
        parts: list[str] = []

        parts.append("## 任务")
        parts.append(
            "请仔细阅读以下数据摘要，推断数据的业务含义，并以 JSON 格式返回你的分析结果。"
        )

        # 注入用户提供的业务上下文
        if business_context:
            parts.append("\n## 用户提供的业务背景")
            parts.append(f"用户描述的业务场景：{business_context}")

        if business_question:
            parts.append("\n## 用户的关键业务问题")
            parts.append(f"用户希望回答的问题：{business_question}")
            parts.append("请在分析中特别关注与该问题相关的数据特征和指标。")

        parts.append("\n## 数据摘要")
        parts.append(data_summary)

        parts.append("\n## 输出要求")
        parts.append("请严格以 JSON 格式返回，包含以下字段：")
        parts.append("{")
        parts.append('  "inferred_scenario": "推断的业务场景描述（1-2句话）",')
        parts.append('  "key_metrics": ["关键业务指标1", "关键业务指标2", ...],')
        parts.append('  "causal_hypotheses": ["因果假设1", "因果假设2", ...],')
        parts.append('  "data_characteristics": "数据特征描述（质量、规模、特点等）",')
        parts.append('  "analysis_suggestions": ["分析建议1", "分析建议2", ...],')
        parts.append('  "business_question_answer": "对用户业务问题的初步回答（如用户未提供问题则为空字符串）"')
        parts.append("}")
        parts.append("")
        parts.append("注意事项：")
        parts.append("- inferred_scenario 应基于列名和数据内容推断，不要编造")
        parts.append("- key_metrics 列出 3-5 个最重要的业务指标")
        parts.append("- causal_hypotheses 提出 2-4 个合理的因果假设")
        parts.append("- analysis_suggestions 给出 2-3 条有针对性的分析建议")
        parts.append("- 如果用户提供了业务问题，business_question_answer 应给出基于数据的初步回答")

        return "\n".join(parts)

    @staticmethod
    def descriptive_prompt(
        data_summary: str,
        domain: str,
        audience: str,
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """生成描述性分析的 prompt

        根据受众类型和业务理解动态调整 prompt 风格：
        - 业务受众：不出现 p 值、置信区间等技术术语，用"影响很大/较小"、"关系密切"等业务语言
        - 技术受众：保留所有统计细节
        - 有 business_understanding 时，注入业务场景和关键指标信息

        Args:
            data_summary: 数据概要信息（统计量、分布等）
            domain: 业务领域描述
            audience: 目标受众（executive / analyst / technical）
            business_understanding: L1 数据预扫描的业务理解结果（可选）

        Returns:
            str: 完整的用户提示词
        """
        # 根据是否有业务理解，使用不同的受众引导
        if business_understanding and business_understanding.inferred_scenario:
            # 有业务理解时，使用更贴合业务场景的引导
            audience_guide = {
                "executive": (
                    "请用简洁的非技术语言，聚焦核心发现和商业影响。"
                    "不要使用 p 值、置信区间、标准差等统计术语。"
                    "用'影响很大/较小'、'关系密切/较弱'、'显著偏高/偏低'等业务语言替代。"
                    "始终围绕业务场景展开分析。"
                ),
                "analyst": (
                    "请提供详细的分析过程和关键统计指标。"
                    "在统计发现的基础上，结合业务场景给出洞察。"
                ),
                "technical": (
                    "请包含详细的技术指标、分布特征和异常检测细节。"
                    "同时说明这些技术发现对业务场景的意义。"
                ),
            }
        else:
            audience_guide = {
                "executive": "请用简洁的非技术语言，聚焦核心发现和商业影响。",
                "analyst": "请提供详细的分析过程和关键统计指标。",
                "technical": "请包含详细的技术指标、分布特征和异常检测细节。",
            }
        guide = audience_guide.get(audience, audience_guide["analyst"])

        parts: list[str] = []

        parts.append("## 任务")
        parts.append("请对以下数据进行描述性分析，生成一段自然语言叙述。")

        # 注入业务理解信息
        if business_understanding:
            parts.append("\n## 业务背景理解")
            if business_understanding.inferred_scenario:
                parts.append(f"推断的业务场景：{business_understanding.inferred_scenario}")
            if business_understanding.key_metrics:
                parts.append(f"关键业务指标：{', '.join(business_understanding.key_metrics)}")
            if business_understanding.data_characteristics:
                parts.append(f"数据特征：{business_understanding.data_characteristics}")
            if business_understanding.analysis_suggestions:
                parts.append(f"分析建议：{'; '.join(business_understanding.analysis_suggestions)}")
            if business_understanding.business_question_answer:
                parts.append(f"用户业务问题的初步回答：{business_understanding.business_question_answer}")

        parts.append(f"\n## 业务领域\n{domain}")
        parts.append(f"\n## 受众要求\n{guide}")
        parts.append(f"\n## 数据概要\n{data_summary}")

        parts.append("\n## 输出要求")
        if business_understanding and business_understanding.business_question_answer:
            # 有业务问题时，围绕回答问题组织分析
            parts.append(
                "请生成一段结构化的分析叙述，围绕回答用户的业务问题来组织：\n"
                "1. 数据整体概况\n"
                "2. 与业务问题直接相关的关键发现\n"
                "3. 关键变量的分布特征\n"
                "4. 重要的相关性和趋势\n"
                "5. 异常值和值得关注的发现\n"
                "6. 对业务问题的综合回答"
            )
        else:
            parts.append(
                "请生成一段结构化的分析叙述，涵盖以下方面：\n"
                "1. 数据整体概况\n"
                "2. 关键变量的分布特征\n"
                "3. 重要的相关性和趋势\n"
                "4. 异常值和值得关注的发现\n"
                "5. 初步的业务洞察"
            )

        return "\n".join(parts)

    @staticmethod
    def causal_prompt(
        data_summary: str,
        causal_findings: str,
        domain: str,
        audience: str,
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """生成因果解释的 prompt

        根据受众类型和业务理解动态调整 prompt 风格：
        - 业务受众：用"影响很大/较小"替代效应大小和 p 值
        - 技术受众：保留完整的统计检验细节
        - 有 business_understanding 时，注入因果假设和业务场景

        Args:
            data_summary: 数据概要信息
            causal_findings: 因果分析发现（因果图、效应估计等）
            domain: 业务领域描述
            audience: 目标受众
            business_understanding: L1 数据预扫描的业务理解结果（可选）

        Returns:
            str: 完整的用户提示词
        """
        if business_understanding and business_understanding.inferred_scenario:
            audience_guide = {
                "executive": (
                    "请用简洁的语言解释因果关系，聚焦业务影响和行动建议。"
                    "不要使用 p 值、置信区间、效应大小等技术术语。"
                    "用'A 对 B 影响很大/较小/几乎无影响'、'关系密切/较弱'等业务语言。"
                    "重点说明'这意味着什么'以及'应该怎么做'。"
                ),
                "analyst": (
                    "请详细解释因果机制，包括效应大小和置信度。"
                    "结合业务场景解释因果关系的实际意义。"
                ),
                "technical": (
                    "请包含详细的因果识别方法、假设条件和统计检验结果。"
                    "同时说明这些技术发现对业务场景的影响。"
                ),
            }
        else:
            audience_guide = {
                "executive": "请用简洁的语言解释因果关系，聚焦业务影响和行动建议。",
                "analyst": "请详细解释因果机制，包括效应大小和置信度。",
                "technical": "请包含详细的因果识别方法、假设条件和统计检验结果。",
            }
        guide = audience_guide.get(audience, audience_guide["analyst"])

        parts: list[str] = []

        parts.append("## 任务")
        parts.append("请基于以下因果分析结果，生成一段自然语言的因果解释。")

        # 注入业务理解信息
        if business_understanding:
            parts.append("\n## 业务背景理解")
            if business_understanding.inferred_scenario:
                parts.append(f"推断的业务场景：{business_understanding.inferred_scenario}")
            if business_understanding.causal_hypotheses:
                parts.append(
                    "预扫描阶段的因果假设：\n"
                    + "\n".join(f"- {h}" for h in business_understanding.causal_hypotheses)
                )
            if business_understanding.key_metrics:
                parts.append(f"关键业务指标：{', '.join(business_understanding.key_metrics)}")
            if business_understanding.business_question_answer:
                parts.append(f"用户业务问题的初步回答：{business_understanding.business_question_answer}")

        parts.append(f"\n## 业务领域\n{domain}")
        parts.append(f"\n## 受众要求\n{guide}")
        parts.append(f"\n## 数据概要\n{data_summary}")
        parts.append(f"\n## 因果分析结果\n{causal_findings}")

        parts.append("\n## 输出要求")
        if business_understanding and business_understanding.business_question_answer:
            parts.append(
                "请生成因果解释叙述，围绕回答用户的业务问题来组织：\n"
                "1. 与业务问题直接相关的关键因果关系\n"
                "2. 各因果效应的大小和方向（用业务语言描述）\n"
                "3. 因果假设和局限性\n"
                "4. 反事实推理的洞察\n"
                "5. 基于因果分析对业务问题的回答和建议"
            )
        else:
            parts.append(
                "请生成因果解释叙述，涵盖以下方面：\n"
                "1. 识别出的关键因果关系\n"
                "2. 各因果效应的大小和方向\n"
                "3. 因果假设和局限性\n"
                "4. 反事实推理的洞察\n"
                "5. 可操作的业务建议"
            )

        return "\n".join(parts)

    @staticmethod
    def predictive_prompt(
        data_summary: str,
        scenarios: str,
        domain: str,
        audience: str,
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """生成预测解释的 prompt

        根据受众类型和业务理解动态调整 prompt 风格：
        - 业务受众：用"可能增长/下降"替代具体数值和置信区间
        - 技术受众：保留完整的模型细节
        - 有 business_understanding 时，注入关键指标和业务场景

        Args:
            data_summary: 数据概要信息
            scenarios: 场景模拟结果描述
            domain: 业务领域描述
            audience: 目标受众
            business_understanding: L1 数据预扫描的业务理解结果（可选）

        Returns:
            str: 完整的用户提示词
        """
        if business_understanding and business_understanding.inferred_scenario:
            audience_guide = {
                "executive": (
                    "请用简洁的语言解释预测结果和场景差异，聚焦决策建议。"
                    "不要使用置信区间、敏感性评分等技术术语。"
                    "用'可能增长/下降/持平'、'影响较大/较小'等业务语言。"
                    "重点说明'在不同情况下会怎样'以及'建议怎么做'。"
                ),
                "analyst": (
                    "请详细解释各场景的预测结果、敏感性分析和关键驱动因素。"
                    "结合业务场景解释预测的实际意义。"
                ),
                "technical": (
                    "请包含模型细节、置信区间、敏感性评分和 What-If 分析结果。"
                    "同时说明预测模型对业务场景的适用性。"
                ),
            }
        else:
            audience_guide = {
                "executive": "请用简洁的语言解释预测结果和场景差异，聚焦决策建议。",
                "analyst": "请详细解释各场景的预测结果、敏感性分析和关键驱动因素。",
                "technical": "请包含模型细节、置信区间、敏感性评分和 What-If 分析结果。",
            }
        guide = audience_guide.get(audience, audience_guide["analyst"])

        parts: list[str] = []

        parts.append("## 任务")
        parts.append("请基于以下预测模拟结果，生成一段自然语言的预测解释。")

        # 注入业务理解信息
        if business_understanding:
            parts.append("\n## 业务背景理解")
            if business_understanding.inferred_scenario:
                parts.append(f"推断的业务场景：{business_understanding.inferred_scenario}")
            if business_understanding.key_metrics:
                parts.append(f"关键业务指标：{', '.join(business_understanding.key_metrics)}")
            if business_understanding.causal_hypotheses:
                parts.append(
                    "已知的因果假设：\n"
                    + "\n".join(f"- {h}" for h in business_understanding.causal_hypotheses)
                )
            if business_understanding.business_question_answer:
                parts.append(f"用户业务问题的初步回答：{business_understanding.business_question_answer}")

        parts.append(f"\n## 业务领域\n{domain}")
        parts.append(f"\n## 受众要求\n{guide}")
        parts.append(f"\n## 数据概要\n{data_summary}")
        parts.append(f"\n## 场景模拟结果\n{scenarios}")

        parts.append("\n## 输出要求")
        if business_understanding and business_understanding.business_question_answer:
            parts.append(
                "请生成预测解释叙述，围绕回答用户的业务问题来组织：\n"
                "1. 与业务问题直接相关的预测结果概述\n"
                "2. 不同场景下业务问题的可能走向\n"
                "3. 敏感性分析和关键驱动因素\n"
                "4. 预测的置信度和不确定性\n"
                "5. 基于预测对业务问题的回答和决策建议"
            )
        else:
            parts.append(
                "请生成预测解释叙述，涵盖以下方面：\n"
                "1. 各场景的预测结果概述\n"
                "2. 场景之间的关键差异\n"
                "3. 敏感性分析和关键驱动因素\n"
                "4. 预测的置信度和不确定性\n"
                "5. 基于预测的决策建议"
            )

        return "\n".join(parts)

    @staticmethod
    def report_prompt(
        executive_data: str,
        detailed_data: str,
        technical_data: str,
        audience: str,
        domain: str,
        business_understanding: BusinessUnderstanding | None = None,
    ) -> str:
        """生成报告的 prompt

        如果有 business_understanding，在报告中注入业务背景信息，
        使报告更有针对性。

        Args:
            executive_data: 高管摘要数据
            detailed_data: 详细分析数据
            technical_data: 技术附录数据
            audience: 目标受众
            domain: 业务领域描述
            business_understanding: L1 数据预扫描的业务理解结果（可选）

        Returns:
            str: 完整的用户提示词
        """
        parts: list[str] = []

        parts.append("## 任务")
        parts.append("请将以下分析结果整合为一份完整的可解释性分析报告。")

        # 注入业务理解信息
        if business_understanding:
            parts.append("\n## 业务背景")
            if business_understanding.inferred_scenario:
                parts.append(f"业务场景：{business_understanding.inferred_scenario}")
            if business_understanding.key_metrics:
                parts.append(f"关键指标：{', '.join(business_understanding.key_metrics)}")
            if business_understanding.business_question_answer:
                parts.append(
                    f"用户关注的问题及初步回答：{business_understanding.business_question_answer}"
                )

        parts.append(f"\n## 业务领域\n{domain}")
        parts.append(f"\n## 目标受众\n{audience}")
        parts.append(f"\n## 高管摘要素材\n{executive_data}")
        parts.append(f"\n## 详细分析素材\n{detailed_data}")
        parts.append(f"\n## 技术附录素材\n{technical_data}")

        parts.append("\n## 输出要求")
        if business_understanding and business_understanding.business_question_answer:
            parts.append(
                "请生成一份结构完整的报告，围绕用户的业务问题组织内容：\n"
                "1. **执行摘要**：面向管理层的核心发现和建议（1-2 段），直接回答用户问题\n"
                "2. **详细分析**：面向分析师的完整分析过程和结论\n"
                "3. **技术附录**：面向技术人员的方法论、假设和统计细节\n"
                "4. 确保各部分内容风格与受众匹配，避免信息冗余\n"
                "5. 报告应始终围绕用户的业务问题展开，给出清晰的回答"
            )
        else:
            parts.append(
                "请生成一份结构完整的报告，包含以下部分：\n"
                "1. **执行摘要**：面向管理层的核心发现和建议（1-2 段）\n"
                "2. **详细分析**：面向分析师的完整分析过程和结论\n"
                "3. **技术附录**：面向技术人员的方法论、假设和统计细节\n"
                "4. 确保各部分内容风格与受众匹配，避免信息冗余"
            )

        return "\n".join(parts)

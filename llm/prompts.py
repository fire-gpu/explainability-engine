"""
Prompt 模板管理模块

集中管理各分析阶段的系统提示词和用户提示词模板，
确保 LLM 输出风格一致且符合业务场景需求。
"""

from __future__ import annotations


class PromptTemplates:
    """Prompt 模板管理器

    提供描述性分析、因果解释、预测解释和报告生成四个阶段的
    系统提示词常量及用户提示词生成方法。
    """

    # ================================================================
    # 系统提示词
    # ================================================================

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
    def descriptive_prompt(data_summary: str, domain: str, audience: str) -> str:
        """生成描述性分析的 prompt

        Args:
            data_summary: 数据概要信息（统计量、分布等）
            domain: 业务领域描述
            audience: 目标受众（executive / analyst / technical）

        Returns:
            str: 完整的用户提示词
        """
        audience_guide = {
            "executive": "请用简洁的非技术语言，聚焦核心发现和商业影响。",
            "analyst": "请提供详细的分析过程和关键统计指标。",
            "technical": "请包含详细的技术指标、分布特征和异常检测细节。",
        }
        guide = audience_guide.get(audience, audience_guide["analyst"])

        return (
            f"## 任务\n"
            f"请对以下数据进行描述性分析，生成一段自然语言叙述。\n\n"
            f"## 业务领域\n"
            f"{domain}\n\n"
            f"## 受众要求\n"
            f"{guide}\n\n"
            f"## 数据概要\n"
            f"{data_summary}\n\n"
            f"## 输出要求\n"
            f"请生成一段结构化的分析叙述，涵盖以下方面：\n"
            f"1. 数据整体概况\n"
            f"2. 关键变量的分布特征\n"
            f"3. 重要的相关性和趋势\n"
            f"4. 异常值和值得关注的发现\n"
            f"5. 初步的业务洞察"
        )

    @staticmethod
    def causal_prompt(
        data_summary: str, causal_findings: str, domain: str, audience: str
    ) -> str:
        """生成因果解释的 prompt

        Args:
            data_summary: 数据概要信息
            causal_findings: 因果分析发现（因果图、效应估计等）
            domain: 业务领域描述
            audience: 目标受众

        Returns:
            str: 完整的用户提示词
        """
        audience_guide = {
            "executive": "请用简洁的语言解释因果关系，聚焦业务影响和行动建议。",
            "analyst": "请详细解释因果机制，包括效应大小和置信度。",
            "technical": "请包含详细的因果识别方法、假设条件和统计检验结果。",
        }
        guide = audience_guide.get(audience, audience_guide["analyst"])

        return (
            f"## 任务\n"
            f"请基于以下因果分析结果，生成一段自然语言的因果解释。\n\n"
            f"## 业务领域\n"
            f"{domain}\n\n"
            f"## 受众要求\n"
            f"{guide}\n\n"
            f"## 数据概要\n"
            f"{data_summary}\n\n"
            f"## 因果分析结果\n"
            f"{causal_findings}\n\n"
            f"## 输出要求\n"
            f"请生成因果解释叙述，涵盖以下方面：\n"
            f"1. 识别出的关键因果关系\n"
            f"2. 各因果效应的大小和方向\n"
            f"3. 因果假设和局限性\n"
            f"4. 反事实推理的洞察\n"
            f"5. 可操作的业务建议"
        )

    @staticmethod
    def predictive_prompt(
        data_summary: str, scenarios: str, domain: str, audience: str
    ) -> str:
        """生成预测解释的 prompt

        Args:
            data_summary: 数据概要信息
            scenarios: 场景模拟结果描述
            domain: 业务领域描述
            audience: 目标受众

        Returns:
            str: 完整的用户提示词
        """
        audience_guide = {
            "executive": "请用简洁的语言解释预测结果和场景差异，聚焦决策建议。",
            "analyst": "请详细解释各场景的预测结果、敏感性分析和关键驱动因素。",
            "technical": "请包含模型细节、置信区间、敏感性评分和 What-If 分析结果。",
        }
        guide = audience_guide.get(audience, audience_guide["analyst"])

        return (
            f"## 任务\n"
            f"请基于以下预测模拟结果，生成一段自然语言的预测解释。\n\n"
            f"## 业务领域\n"
            f"{domain}\n\n"
            f"## 受众要求\n"
            f"{guide}\n\n"
            f"## 数据概要\n"
            f"{data_summary}\n\n"
            f"## 场景模拟结果\n"
            f"{scenarios}\n\n"
            f"## 输出要求\n"
            f"请生成预测解释叙述，涵盖以下方面：\n"
            f"1. 各场景的预测结果概述\n"
            f"2. 场景之间的关键差异\n"
            f"3. 敏感性分析和关键驱动因素\n"
            f"4. 预测的置信度和不确定性\n"
            f"5. 基于预测的决策建议"
        )

    @staticmethod
    def report_prompt(
        executive_data: str,
        detailed_data: str,
        technical_data: str,
        audience: str,
        domain: str,
    ) -> str:
        """生成报告的 prompt

        Args:
            executive_data: 高管摘要数据
            detailed_data: 详细分析数据
            technical_data: 技术附录数据
            audience: 目标受众
            domain: 业务领域描述

        Returns:
            str: 完整的用户提示词
        """
        return (
            f"## 任务\n"
            f"请将以下分析结果整合为一份完整的可解释性分析报告。\n\n"
            f"## 业务领域\n"
            f"{domain}\n\n"
            f"## 目标受众\n"
            f"{audience}\n\n"
            f"## 高管摘要素材\n"
            f"{executive_data}\n\n"
            f"## 详细分析素材\n"
            f"{detailed_data}\n\n"
            f"## 技术附录素材\n"
            f"{technical_data}\n\n"
            f"## 输出要求\n"
            f"请生成一份结构完整的报告，包含以下部分：\n"
            f"1. **执行摘要**：面向管理层的核心发现和建议（1-2 段）\n"
            f"2. **详细分析**：面向分析师的完整分析过程和结论\n"
            f"3. **技术附录**：面向技术人员的方法论、假设和统计细节\n"
            f"4. 确保各部分内容风格与受众匹配，避免信息冗余"
        )

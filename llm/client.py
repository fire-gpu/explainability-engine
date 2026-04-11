"""
LLM 客户端封装模块

提供对 OpenAI 兼容接口的统一封装，支持重试机制和结构化输出。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用异常"""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class LLMClient:
    """LLM 客户端封装，支持 OpenAI 兼容接口

    支持通过构造参数或环境变量 ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
    进行配置，内置指数退避重试机制。

    Args:
        api_key: OpenAI API 密钥，默认从环境变量 ``OPENAI_API_KEY`` 读取
        base_url: API 基础 URL，默认从环境变量 ``OPENAI_BASE_URL`` 读取
        model: 使用的模型名称，默认 ``gpt-4o-mini``
        temperature: 默认采样温度，默认 0.2
        max_retries: 最大重试次数，默认 3
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # 优先使用传入参数，否则从环境变量读取
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        if not resolved_api_key:
            raise LLMError(
                "未提供 API 密钥。请通过 api_key 参数或环境变量 OPENAI_API_KEY 配置。"
            )

        client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self._client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """调用 LLM 生成文本，带重试机制

        使用指数退避策略进行重试，重试间隔为 1s、2s、4s ...

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            temperature: 采样温度，为 None 时使用实例默认值

        Returns:
            str: LLM 生成的文本内容

        Raises:
            LLMError: 所有重试均失败后抛出
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resolved_temp = temperature if temperature is not None else self.temperature

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=resolved_temp,
                )
                content = response.choices[0].message.content
                return content if content else ""

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次）: %s",
                    attempt + 1,
                    self.max_retries,
                    str(e),
                )
                if attempt < self.max_retries - 1:
                    # 指数退避：1s, 2s, 4s ...
                    import time

                    time.sleep(2**attempt)

        raise LLMError(
            f"LLM 调用在 {self.max_retries} 次重试后仍然失败",
            original_error=last_error,
        )

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """调用 LLM 生成结构化输出（JSON）

        将 LLM 返回的文本解析为 JSON 字典。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            response_format: OpenAI 响应格式配置（可选），
                如 ``{"type": "json_object"}``

        Returns:
            dict: 解析后的 JSON 字典

        Raises:
            LLMError: LLM 调用失败或返回内容无法解析为 JSON
        """
        # 如果指定了 response_format，通过 generate 传递
        # 否则在 system_prompt 中要求 JSON 输出
        effective_system = system_prompt or ""
        if response_format is None:
            effective_system += "\n\n请以 JSON 格式返回结果。"

        raw_text = self.generate(prompt=prompt, system_prompt=effective_system)

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            # 尝试从文本中提取 JSON 块
            extracted = self._extract_json(raw_text)
            if extracted is not None:
                return extracted
            raise LLMError(
                f"LLM 返回的内容无法解析为 JSON: {e}",
                original_error=e,
            )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """从文本中提取 JSON 块

        尝试匹配 ```json ... ``` 代码块或直接解析整个文本。

        Args:
            text: 包含 JSON 的文本

        Returns:
            dict | None: 解析成功返回字典，否则返回 None
        """
        import re

        # 尝试匹配 ```json ... ``` 代码块
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试匹配第一个 { ... } 块
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        brace_match = re.search(brace_pattern, text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

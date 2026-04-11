"""
全局配置管理模块

提供 Settings 类，支持从 YAML 文件或环境变量加载配置，
管理 LLM 连接、分析默认参数和输出设置。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Settings:
    """全局配置管理

    支持从 YAML 配置文件或环境变量加载配置项。
    环境变量的优先级高于配置文件。

    Attributes:
        llm_api_key: LLM API 密钥
        llm_base_url: LLM API 基础 URL
        llm_model: 使用的 LLM 模型名称
        llm_temperature: LLM 采样温度
        llm_max_retries: LLM 最大重试次数
        default_domain: 默认业务领域
        default_audience: 默认目标受众
        default_depth: 默认分析深度
        output_dir: 报告输出目录
        cache_enabled: 是否启用缓存
        log_level: 日志级别
    """

    def __init__(self, config_path: str | None = None) -> None:
        """初始化配置

        加载顺序（后者覆盖前者）：
        1. 内置默认值
        2. YAML 配置文件（如果提供且存在）
        3. 环境变量

        Args:
            config_path: YAML 配置文件路径（可选）
        """
        # 内置默认值
        self.llm_api_key: str = ""
        self.llm_base_url: str = ""
        self.llm_model: str = "gpt-4o-mini"
        self.llm_temperature: float = 0.2
        self.llm_max_retries: int = 3

        self.default_domain: str = ""
        self.default_audience: str = "analyst"
        self.default_depth: str = "standard"

        self.output_dir: str = "./output"
        self.cache_enabled: bool = True

        self.log_level: str = "INFO"

        # 从 YAML 文件加载（如果提供）
        if config_path:
            self._load_from_yaml(config_path)

        # 从环境变量覆盖
        self._load_from_env()

    def _load_from_yaml(self, config_path: str) -> None:
        """从 YAML 配置文件加载配置

        Args:
            config_path: YAML 文件路径
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning("配置文件不存在: %s", config_path)
            return

        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning("配置文件格式不正确，期望字典类型: %s", config_path)
                return

            self._apply_dict(data)
            logger.info("已从配置文件加载配置: %s", config_path)

        except ImportError:
            logger.warning("未安装 PyYAML，跳过配置文件加载: %s", config_path)
        except Exception as e:
            logger.warning("配置文件加载失败: %s，错误: %s", config_path, e)

    def _apply_dict(self, data: dict[str, Any]) -> None:
        """从字典中应用配置项

        Args:
            data: 配置字典
        """
        # LLM 配置
        llm = data.get("llm", {})
        if isinstance(llm, dict):
            self.llm_api_key = llm.get("api_key", self.llm_api_key)
            self.llm_base_url = llm.get("base_url", self.llm_base_url)
            self.llm_model = llm.get("model", self.llm_model)
            self.llm_temperature = float(llm.get("temperature", self.llm_temperature))
            self.llm_max_retries = int(llm.get("max_retries", self.llm_max_retries))

        # 分析默认配置
        analysis = data.get("analysis", {})
        if isinstance(analysis, dict):
            self.default_domain = analysis.get("domain", self.default_domain)
            self.default_audience = analysis.get("audience", self.default_audience)
            self.default_depth = analysis.get("depth", self.default_depth)

        # 输出配置
        output = data.get("output", {})
        if isinstance(output, dict):
            self.output_dir = output.get("dir", self.output_dir)
            self.cache_enabled = bool(output.get("cache_enabled", self.cache_enabled))

        # 日志配置
        log = data.get("log", {})
        if isinstance(log, dict):
            self.log_level = log.get("level", self.log_level)

    def _load_from_env(self) -> None:
        """从环境变量加载配置

        支持的环境变量：
        - OPENAI_API_KEY: LLM API 密钥
        - OPENAI_BASE_URL: LLM API 基础 URL
        - LLM_MODEL: LLM 模型名称
        - LLM_TEMPERATURE: LLM 采样温度
        - LLM_MAX_RETRIES: LLM 最大重试次数
        - DEFAULT_DOMAIN: 默认业务领域
        - DEFAULT_AUDIENCE: 默认目标受众
        - DEFAULT_DEPTH: 默认分析深度
        - OUTPUT_DIR: 报告输出目录
        - CACHE_ENABLED: 是否启用缓存
        - LOG_LEVEL: 日志级别
        """
        # LLM 配置
        if os.environ.get("OPENAI_API_KEY"):
            self.llm_api_key = os.environ["OPENAI_API_KEY"]
        if os.environ.get("OPENAI_BASE_URL"):
            self.llm_base_url = os.environ["OPENAI_BASE_URL"]
        if os.environ.get("LLM_MODEL"):
            self.llm_model = os.environ["LLM_MODEL"]
        if os.environ.get("LLM_TEMPERATURE"):
            try:
                self.llm_temperature = float(os.environ["LLM_TEMPERATURE"])
            except ValueError:
                pass
        if os.environ.get("LLM_MAX_RETRIES"):
            try:
                self.llm_max_retries = int(os.environ["LLM_MAX_RETRIES"])
            except ValueError:
                pass

        # 分析默认配置
        if os.environ.get("DEFAULT_DOMAIN"):
            self.default_domain = os.environ["DEFAULT_DOMAIN"]
        if os.environ.get("DEFAULT_AUDIENCE"):
            self.default_audience = os.environ["DEFAULT_AUDIENCE"]
        if os.environ.get("DEFAULT_DEPTH"):
            self.default_depth = os.environ["DEFAULT_DEPTH"]

        # 输出配置
        if os.environ.get("OUTPUT_DIR"):
            self.output_dir = os.environ["OUTPUT_DIR"]
        if os.environ.get("CACHE_ENABLED"):
            self.cache_enabled = os.environ["CACHE_ENABLED"].lower() in (
                "true", "1", "yes", "on",
            )

        # 日志配置
        if os.environ.get("LOG_LEVEL"):
            self.log_level = os.environ["LOG_LEVEL"].upper()

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量创建配置

        便捷方法，直接从环境变量加载所有配置项。

        Returns:
            Settings: 配置实例
        """
        return cls(config_path=None)

    def to_dict(self) -> dict[str, Any]:
        """将配置导出为字典

        Returns:
            dict: 配置字典（排除敏感信息）
        """
        return {
            "llm": {
                "base_url": self.llm_base_url,
                "model": self.llm_model,
                "temperature": self.llm_temperature,
                "max_retries": self.llm_max_retries,
                # 不导出 api_key
            },
            "analysis": {
                "domain": self.default_domain,
                "audience": self.default_audience,
                "depth": self.default_depth,
            },
            "output": {
                "dir": self.output_dir,
                "cache_enabled": self.cache_enabled,
            },
            "log": {
                "level": self.log_level,
            },
        }

    def ensure_output_dir(self) -> str:
        """确保输出目录存在，不存在则创建

        Returns:
            str: 输出目录的绝对路径
        """
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

    def __repr__(self) -> str:
        """配置的字符串表示（排除敏感信息）"""
        return (
            f"Settings("
            f"llm_model={self.llm_model!r}, "
            f"default_domain={self.default_domain!r}, "
            f"default_audience={self.default_audience!r}, "
            f"default_depth={self.default_depth!r}, "
            f"output_dir={self.output_dir!r}, "
            f"cache_enabled={self.cache_enabled})"
        )

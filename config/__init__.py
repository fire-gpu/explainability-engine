"""
全局配置模块

提供 Settings 类，支持从 YAML 文件或环境变量加载配置。
提供 DomainLoader 类，用于加载领域特定的分析配置。
"""

from config.domain_loader import DomainLoader
from config.settings import Settings

__all__ = [
    "Settings",
    "DomainLoader",
]

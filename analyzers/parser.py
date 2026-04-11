"""
文件解析器模块

支持 CSV / JSON / Excel 格式的文件解析，自动推断列类型并提取数据元信息。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from core.models import AnalysisConfig, AnalysisInput, ColumnMeta, DataMetadata


class FileParser:
    """文件解析器，支持 CSV/JSON/Excel 格式

    通过文件扩展名自动选择解析方式，推断列类型并提取元信息，
    最终返回 :class:`AnalysisInput` 供下游分析模块使用。
    """

    # 支持的文件扩展名映射
    _SUPPORTED_EXTENSIONS: dict[str, str] = {
        ".csv": "csv",
        ".json": "json",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    def parse(self, file_path: str, config: AnalysisConfig | None = None) -> AnalysisInput:
        """解析文件，返回 AnalysisInput

        Args:
            file_path: 文件路径，支持 CSV / JSON / Excel 格式
            config: 分析配置，为 None 时使用默认配置

        Returns:
            AnalysisInput: 包含数据、元信息和配置的分析输入对象

        Raises:
            ValueError: 不支持的文件类型或文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件类型: {ext}，"
                f"支持的类型: {list(self._SUPPORTED_EXTENSIONS.keys())}"
            )

        file_type = self._SUPPORTED_EXTENSIONS[ext]
        df = self._read_file(path, file_type)

        # 推断列类型
        columns = self._detect_column_types(df)

        # 提取元信息
        domain = "generic"
        metadata = self._extract_metadata(df, columns, domain)

        # 使用传入配置或默认配置
        if config is None:
            config = AnalysisConfig()

        return AnalysisInput(data=df, metadata=metadata, config=config)

    def _read_file(self, path: Path, file_type: str) -> pd.DataFrame:
        """根据文件类型读取数据

        Args:
            path: 文件路径
            file_type: 文件类型标识（csv / json / excel）

        Returns:
            pd.DataFrame: 解析后的数据框
        """
        if file_type == "csv":
            return pd.read_csv(path)
        elif file_type == "json":
            return pd.read_json(path, orient="records")
        elif file_type == "excel":
            return pd.read_excel(path)
        else:
            raise ValueError(f"未知的文件类型: {file_type}")

    def _detect_column_types(self, df: pd.DataFrame) -> list[ColumnMeta]:
        """推断每列的数据类型

        按以下优先级判断：
        1. **numeric**: int / float 类型的列
        2. **datetime**: 可被 ``pd.to_datetime`` 解析的列
        3. **categorical**: 唯一值数量 < 20 的非数值列
        4. **text**: 其余列

        Args:
            df: 待推断的数据框

        Returns:
            list[ColumnMeta]: 各列的元信息列表
        """
        columns: list[ColumnMeta] = []

        for col in df.columns:
            series = df[col]

            # 跳过全为 NaN 的列
            if series.isna().all():
                columns.append(ColumnMeta(name=col, type="text", description=""))
                continue

            # 判断 numeric
            if pd.api.types.is_numeric_dtype(series):
                columns.append(ColumnMeta(name=col, type="numeric", description=""))
                continue

            # 判断 datetime
            if self._is_datetime_column(series):
                columns.append(ColumnMeta(name=col, type="datetime", description=""))
                continue

            # 判断 categorical（唯一值 < 20 的非数值列）
            unique_count = series.dropna().nunique()
            if unique_count < 20:
                columns.append(ColumnMeta(name=col, type="categorical", description=""))
            else:
                columns.append(ColumnMeta(name=col, type="text", description=""))

        return columns

    def _is_datetime_column(self, series: pd.Series) -> bool:
        """判断列是否为日期时间类型

        尝试将列解析为 datetime，若成功且非空值超过 50% 则认为是日期时间列。

        Args:
            series: 待判断的序列

        Returns:
            bool: 是否为日期时间列
        """
        # 已经是 datetime 类型
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # 尝试解析
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            non_null_ratio = parsed.notna().mean()
            return non_null_ratio > 0.5
        except (ValueError, TypeError):
            return False

    def _extract_metadata(
        self, df: pd.DataFrame, columns: list[ColumnMeta], domain: str
    ) -> DataMetadata:
        """提取数据元信息

        包括行数、缺失值比例和时间范围。

        Args:
            df: 数据框
            columns: 列元信息列表
            domain: 业务领域描述

        Returns:
            DataMetadata: 数据集元信息
        """
        row_count = len(df)

        # 计算整体缺失比例
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0.0

        # 检测时间范围
        time_range = self._detect_time_range(df, columns)

        return DataMetadata(
            columns=columns,
            row_count=row_count,
            time_range=time_range,
            missing_ratio=round(missing_ratio, 4),
            domain=domain,
        )

    def _detect_time_range(
        self, df: pd.DataFrame, columns: list[ColumnMeta]
    ) -> tuple[str, str] | None:
        """检测数据集中的时间范围

        查找第一个 datetime 类型的列，提取其最小值和最大值作为时间范围。

        Args:
            df: 数据框
            columns: 列元信息列表

        Returns:
            时间范围元组 (最小时间, 最大时间)，若无日期列则返回 None
        """
        datetime_cols = [col.name for col in columns if col.type == "datetime"]

        if not datetime_cols:
            return None

        # 取第一个 datetime 列
        col_name = datetime_cols[0]
        try:
            parsed = pd.to_datetime(df[col_name], errors="coerce")
            valid = parsed.dropna()
            if len(valid) == 0:
                return None
            return (
                valid.min().strftime("%Y-%m-%d"),
                valid.max().strftime("%Y-%m-%d"),
            )
        except (ValueError, TypeError):
            return None

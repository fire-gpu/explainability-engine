"""
文件解析器测试模块

测试 CSV / JSON 文件解析、列类型推断、元数据提取和不支持格式的异常处理。
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from analyzers.parser import FileParser
from core.models import AnalysisConfig, ColumnMeta


class TestParser:
    """文件解析器测试类"""

    def test_parse_csv(self, tmp_path):
        """测试解析 CSV 文件"""
        # 创建临时 CSV 文件
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "salary": [5000, 6000, 7000, 8000, 9000],
            "department": ["工程", "市场", "工程", "财务", "市场"],
        })
        df.to_csv(csv_file, index=False)

        # 解析
        parser = FileParser()
        result = parser.parse(str(csv_file))

        # 验证
        assert result.data.shape == (5, 3)
        assert result.metadata.row_count == 5
        assert len(result.metadata.columns) == 3

    def test_parse_json(self, tmp_path):
        """测试解析 JSON 文件"""
        # 创建临时 JSON 文件
        json_file = tmp_path / "test_data.json"
        data = [
            {"name": "Alice", "score": 85, "passed": True},
            {"name": "Bob", "score": 72, "passed": False},
            {"name": "Charlie", "score": 93, "passed": True},
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # 解析
        parser = FileParser()
        result = parser.parse(str(json_file))

        # 验证
        assert result.data.shape == (3, 3)
        assert result.metadata.row_count == 3
        assert "name" in result.data.columns

    def test_column_type_detection(self, tmp_path):
        """测试列类型推断"""
        # 创建包含多种类型列的 CSV
        csv_file = tmp_path / "types_test.csv"
        df = pd.DataFrame({
            "numeric_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical_col": ["A", "B", "A", "C", "B"],
            "int_col": [10, 20, 30, 40, 50],
        })
        df.to_csv(csv_file, index=False)

        parser = FileParser()
        result = parser.parse(str(csv_file))

        # 验证列类型推断
        col_types = {col.name: col.type for col in result.metadata.columns}
        assert col_types["numeric_col"] == "numeric"
        assert col_types["int_col"] == "numeric"
        assert col_types["categorical_col"] == "categorical"

    def test_metadata_extraction(self, tmp_path):
        """测试元数据提取"""
        # 创建包含缺失值的 CSV
        csv_file = tmp_path / "meta_test.csv"
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        df.to_csv(csv_file, index=False)

        parser = FileParser()
        result = parser.parse(str(csv_file))

        # 验证元数据
        assert result.metadata.row_count == 10
        assert result.metadata.missing_ratio > 0  # 有一个缺失值
        assert len(result.metadata.columns) == 2

    def test_unsupported_format(self, tmp_path):
        """测试不支持的文件格式应抛出 ValueError"""
        # 创建一个不支持的格式文件
        unsupported_file = tmp_path / "data.parquet"
        unsupported_file.write_text("some content")

        parser = FileParser()
        with pytest.raises(ValueError, match="不支持的文件类型"):
            parser.parse(str(unsupported_file))

    def test_file_not_found(self):
        """测试文件不存在时应抛出 ValueError"""
        parser = FileParser()
        with pytest.raises(ValueError, match="文件不存在"):
            parser.parse("/nonexistent/path/to/file.csv")

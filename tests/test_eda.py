"""
Tests for EDA module.

This module contains unit tests for the exploratory data analysis functionality,
including auto-profiling, visualization generation, and summary statistics.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.eda import (
    generate_auto_profile,
    generate_category_analysis,
    generate_correlation_heatmap,
    generate_eda_report,
    generate_rating_distribution,
    generate_size_analysis,
)


class TestEDAFunctions:
    """Test cases for EDA functions."""

    def test_generate_rating_distribution(self, sample_app_data: pd.DataFrame) -> None:
        """Test rating distribution plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_rating_dist.png"
            
            # Should not raise any exceptions
            generate_rating_distribution(sample_app_data, str(output_path))
            
            # Check file was created
            assert output_path.exists()

    def test_generate_category_analysis(self, sample_app_data: pd.DataFrame) -> None:
        """Test category analysis plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_category.png"
            
            # Should not raise any exceptions
            generate_category_analysis(sample_app_data, str(output_path))
            
            # Check file was created
            assert output_path.exists()

    def test_generate_size_analysis(self, sample_app_data: pd.DataFrame) -> None:
        """Test size analysis plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_size.png"
            
            # Should not raise any exceptions
            generate_size_analysis(sample_app_data, str(output_path))
            
            # Check file was created
            assert output_path.exists()

    def test_generate_correlation_heatmap(self, sample_app_data: pd.DataFrame) -> None:
        """Test correlation heatmap generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_correlation.png"
            
            # Should not raise any exceptions
            generate_correlation_heatmap(sample_app_data, str(output_path))
            
            # Check file was created
            assert output_path.exists()

    @patch('src.eda.ProfileReport')
    def test_generate_auto_profile(
        self, 
        mock_profile: Mock, 
        sample_app_data: pd.DataFrame
    ) -> None:
        """Test auto profile generation."""
        mock_report = Mock()
        mock_profile.return_value = mock_report
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_profile.html"
            
            generate_auto_profile(sample_app_data, str(output_path))
            
            # Check ProfileReport was called correctly
            mock_profile.assert_called_once_with(
                sample_app_data,
                title="Google Play Store Apps - Data Profile",
                explorative=True
            )
            mock_report.to_file.assert_called_once_with(str(output_path))

    def test_generate_eda_report(self, sample_app_data: pd.DataFrame) -> None:
        """Test complete EDA report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should not raise any exceptions
            stats = generate_eda_report(sample_app_data, str(output_dir))
            
            # Check that stats are returned
            assert isinstance(stats, dict)
            assert "total_apps" in stats
            assert "unique_categories" in stats
            assert stats["total_apps"] == len(sample_app_data)

    def test_empty_dataframe_handling(self) -> None:
        """Test EDA functions handle empty dataframes gracefully."""
        empty_df = pd.DataFrame()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_empty.png"
            
            # Should handle empty dataframe without crashing
            with pytest.raises((ValueError, IndexError)):
                generate_rating_distribution(empty_df, str(output_path))

    def test_missing_columns_handling(self) -> None:
        """Test EDA functions handle missing columns gracefully."""
        # DataFrame with missing expected columns
        incomplete_df = pd.DataFrame({
            "App": ["Test App"],
            "Category": ["GAME"]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_incomplete.png"
            
            # Should handle missing columns gracefully
            with pytest.raises(KeyError):
                generate_rating_distribution(incomplete_df, str(output_path))

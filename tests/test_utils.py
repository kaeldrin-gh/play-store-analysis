# MIT License
#
# Copyright (c) 2025 Play Store Analysis Project

"""
Tests for utility functions.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils import (
    setup_logging,
    load_config,
    save_config,
    validate_dataframe,
    hash_string,
    safe_divide,
    format_number,
    chunk_dataframe,
    get_system_info,
    timer,
    ProgressTracker
)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging("INFO")
        
        assert logger is not None
        assert logger.name == "playstore_analysis"

    def test_load_save_config(self, temp_directory):
        """Test configuration loading and saving."""
        config_data = {
            'test_key': 'test_value',
            'nested': {
                'inner_key': 123
            }
        }
        
        config_path = temp_directory / "test_config.json"
        
        # Test saving
        save_config(config_data, config_path)
        assert config_path.exists()
        
        # Test loading
        loaded_config = load_config(config_path)
        assert loaded_config == config_data

    def test_load_config_missing_file(self, temp_directory):
        """Test loading non-existent config file."""
        missing_path = temp_directory / "missing.json"
        
        with pytest.raises(FileNotFoundError):
            load_config(missing_path)

    def test_load_config_invalid_json(self, temp_directory):
        """Test loading invalid JSON config."""
        invalid_path = temp_directory / "invalid.json"
        invalid_path.write_text("{ invalid json")
        
        with pytest.raises(json.JSONDecodeError):
            load_config(invalid_path)

    def test_validate_dataframe_success(self, sample_dataframe):
        """Test successful DataFrame validation."""
        required_columns = ['app_id', 'name', 'category']
        
        result = validate_dataframe(sample_dataframe, required_columns, min_rows=10)
        assert result == True

    def test_validate_dataframe_missing_columns(self, sample_dataframe):
        """Test DataFrame validation with missing columns."""
        required_columns = ['app_id', 'missing_column']
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(sample_dataframe, required_columns)

    def test_validate_dataframe_insufficient_rows(self, sample_dataframe):
        """Test DataFrame validation with insufficient rows."""
        with pytest.raises(ValueError, match="minimum"):
            validate_dataframe(sample_dataframe, ['app_id'], min_rows=1000)

    def test_validate_dataframe_none(self):
        """Test DataFrame validation with None input."""
        with pytest.raises(ValueError, match="DataFrame is None"):
            validate_dataframe(None, ['column'])

    def test_hash_string(self):
        """Test string hashing."""
        test_string = "test_string"
        
        # Test MD5
        md5_hash = hash_string(test_string, "md5")
        assert len(md5_hash) == 32
        assert md5_hash == hash_string(test_string, "md5")  # Reproducible
        
        # Test SHA256
        sha256_hash = hash_string(test_string, "sha256")
        assert len(sha256_hash) == 64
        
        # Test unsupported algorithm
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            hash_string(test_string, "unsupported")

    def test_safe_divide(self):
        """Test safe division function."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0  # Default
        assert safe_divide(10, 0, default=999) == 999  # Custom default
        assert safe_divide("invalid", 2) == 0.0  # Type error handling

    def test_format_number(self):
        """Test number formatting."""
        assert format_number(123) == "123.00"
        assert format_number(1234) == "1.23K"
        assert format_number(1234567) == "1.23M"
        assert format_number(1234567890) == "1.23B"
        
        # Test precision
        assert format_number(1234, precision=1) == "1.2K"
        assert format_number(1234, precision=0) == "1K"

    def test_chunk_dataframe(self, sample_dataframe):
        """Test DataFrame chunking."""
        chunks = chunk_dataframe(sample_dataframe, chunk_size=25)
        
        assert len(chunks) == 4  # 100 rows / 25 = 4 chunks
        assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)
        assert sum(len(chunk) for chunk in chunks) == len(sample_dataframe)
        
        # Test with chunk size larger than DataFrame
        single_chunk = chunk_dataframe(sample_dataframe, chunk_size=200)
        assert len(single_chunk) == 1
        assert len(single_chunk[0]) == len(sample_dataframe)

    def test_get_system_info(self):
        """Test system information retrieval."""
        info = get_system_info()
        
        assert isinstance(info, dict)
        assert 'python_version' in info
        assert 'platform' in info
        assert 'pandas_version' in info
        assert 'numpy_version' in info
        assert 'memory_usage_mb' in info

    def test_timer_decorator(self):
        """Test timer decorator."""
        import time
        
        @timer
        def test_function():
            time.sleep(0.01)  # Short sleep
            return "completed"
        
        result = test_function()
        assert result == "completed"

    def test_progress_tracker(self):
        """Test progress tracking."""
        tracker = ProgressTracker(100, "Test task")
        
        assert tracker.total == 100
        assert tracker.current == 0
        assert tracker.description == "Test task"
        
        # Test updates
        tracker.update(10)
        assert tracker.current == 10
        
        tracker.update(90)
        assert tracker.current == 100
        
        # Test finish
        tracker.finish()  # Should not raise exception


class TestFileOperations:
    """Test file operation utilities."""

    def test_backup_file(self, temp_directory):
        """Test file backup functionality."""
        from src.utils import backup_file
        
        # Create test file
        test_file = temp_directory / "test.txt"
        test_file.write_text("test content")
        
        # Create backup
        backup_path = backup_file(test_file)
        
        assert backup_path.exists()
        assert backup_path.read_text() == "test content"
        assert "test_" in backup_path.name  # Should contain timestamp

    def test_backup_file_missing(self, temp_directory):
        """Test backup of non-existent file."""
        from src.utils import backup_file
        
        missing_file = temp_directory / "missing.txt"
        
        with pytest.raises(FileNotFoundError):
            backup_file(missing_file)

    def test_pickle_operations(self, temp_directory):
        """Test pickle save and load operations."""
        from src.utils import save_pickle, load_pickle
        
        test_data = {'key': 'value', 'numbers': [1, 2, 3]}
        pickle_path = temp_directory / "test.pkl"
        
        # Test saving
        save_pickle(test_data, pickle_path)
        assert pickle_path.exists()
        
        # Test loading
        loaded_data = load_pickle(pickle_path)
        assert loaded_data == test_data

    def test_load_pickle_missing(self, temp_directory):
        """Test loading non-existent pickle file."""
        from src.utils import load_pickle
        
        missing_path = temp_directory / "missing.pkl"
        
        with pytest.raises(FileNotFoundError):
            load_pickle(missing_path)

    def test_calculate_file_hash(self, temp_directory):
        """Test file hash calculation."""
        from src.utils import calculate_file_hash
        
        # Create test file
        test_file = temp_directory / "test.txt"
        test_file.write_text("test content for hashing")
        
        # Calculate hash
        file_hash = calculate_file_hash(test_file)
        
        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5 hash length
        
        # Test reproducibility
        hash2 = calculate_file_hash(test_file)
        assert file_hash == hash2

    def test_calculate_file_hash_missing(self, temp_directory):
        """Test hash calculation for missing file."""
        from src.utils import calculate_file_hash
        
        missing_file = temp_directory / "missing.txt"
        
        with pytest.raises(FileNotFoundError):
            calculate_file_hash(missing_file)

    def test_create_directory_structure(self, temp_directory):
        """Test directory structure creation."""
        from src.utils import create_directory_structure
        
        structure = {
            'data': {
                'raw': {},
                'processed': {}
            },
            'reports': {
                'figures': {}
            },
            'models': {}
        }
        
        create_directory_structure(temp_directory, structure)
        
        # Check that directories were created
        assert (temp_directory / 'data').exists()
        assert (temp_directory / 'data' / 'raw').exists()
        assert (temp_directory / 'data' / 'processed').exists()
        assert (temp_directory / 'reports' / 'figures').exists()
        assert (temp_directory / 'models').exists()

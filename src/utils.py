# MIT License
#
# Copyright (c) 2025 Play Store Analysis Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""
Utility functions for Google Play Store analysis.

This module provides common utility functions used across the project,
including file I/O, logging setup, data validation, and helper functions.
"""

import hashlib
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("playstore_analysis")
    
    # Add file handler if log_file specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logging.info(f"Loaded configuration from {config_path}")
        return config
    
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {e}")


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logging.info(f"Configuration saved to {config_path}")


def validate_dataframe(
    df: DataFrame,
    required_columns: List[str],
    min_rows: int = 1
) -> bool:
    """
    Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    # Check if DataFrame is not None
    if df is None:
        raise ValueError("DataFrame is None")
    
    # Check minimum rows
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty DataFrame
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    logging.info(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def hash_string(text: str, algorithm: str = "md5") -> str:
    """
    Generate hash of string using specified algorithm.

    Args:
        text: String to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

    Returns:
        Hexadecimal hash string

    Raises:
        ValueError: If algorithm is not supported
    """
    supported_algorithms = {"md5", "sha1", "sha256"}
    
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use one of {supported_algorithms}")
    
    hash_func = getattr(hashlib, algorithm)
    return hash_func(text.encode('utf-8')).hexdigest()


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero

    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return default


def memory_usage_mb() -> float:
    """
    Get current memory usage in megabytes.

    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback if psutil not available
        return 0.0


def create_directory_structure(base_path: Path, structure: Dict[str, Any]) -> None:
    """
    Create nested directory structure from dictionary specification.

    Args:
        base_path: Base path where to create structure
        structure: Dictionary specifying directory structure

    Example:
        structure = {
            "data": {
                "raw": {},
                "processed": {}
            },
            "reports": {
                "figures": {}
            }
        }
    """
    base_path = Path(base_path)
    
    def create_recursive(current_path: Path, struct: Dict[str, Any]) -> None:
        for name, sub_struct in struct.items():
            dir_path = current_path / name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(sub_struct, dict):
                create_recursive(dir_path, sub_struct)
    
    create_recursive(base_path, structure)
    logging.info(f"Created directory structure at {base_path}")


def backup_file(file_path: Path, backup_dir: Optional[Path] = None) -> Path:
    """
    Create backup copy of file with timestamp.

    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (default: same as original)

    Returns:
        Path to backup file

    Raises:
        FileNotFoundError: If original file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)
    
    logging.info(f"Created backup: {backup_path}")
    return backup_path


def load_pickle(file_path: Path) -> Any:
    """
    Safely load object from pickle file.

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded object

    Raises:
        FileNotFoundError: If pickle file doesn't exist
        pickle.PickleError: If file is corrupted or incompatible
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        
        logging.info(f"Loaded pickle object from {file_path}")
        return obj
    
    except (pickle.PickleError, EOFError) as e:
        raise pickle.PickleError(f"Failed to load pickle file {file_path}: {e}")


def save_pickle(obj: Any, file_path: Path) -> None:
    """
    Safely save object to pickle file.

    Args:
        obj: Object to save
        file_path: Path where to save pickle file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info(f"Saved pickle object to {file_path}")
    
    except Exception as e:
        raise RuntimeError(f"Failed to save pickle file {file_path}: {e}")


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """
    Format number with appropriate units (K, M, B).

    Args:
        number: Number to format
        precision: Decimal precision for formatted number

    Returns:
        Formatted number string

    Example:
        format_number(1500) -> "1.50K"
        format_number(2500000) -> "2.50M"
    """
    if abs(number) >= 1e9:
        return f"{number/1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"


def chunk_dataframe(df: DataFrame, chunk_size: int = 1000) -> List[DataFrame]:
    """
    Split DataFrame into smaller chunks.

    Args:
        df: DataFrame to split
        chunk_size: Maximum rows per chunk

    Returns:
        List of DataFrame chunks
    """
    if len(df) <= chunk_size:
        return [df]
    
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    
    logging.info(f"Split DataFrame into {len(chunks)} chunks of max {chunk_size} rows")
    return chunks


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and logging.

    Returns:
        Dictionary with system information
    """
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "memory_usage_mb": memory_usage_mb(),
    }
    
    # Add optional library versions
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    
    try:
        import matplotlib
        info["matplotlib_version"] = matplotlib.__version__
    except ImportError:
        pass
    
    return info


def calculate_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """
    Calculate hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use

    Returns:
        Hexadecimal hash of file contents

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = getattr(hashlib, algorithm)
    file_hash = hash_func()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            file_hash.update(chunk)
    
    return file_hash.hexdigest()


def timer(func):
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    
    Provides progress logging and estimated time remaining.
    """

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time

    def update(self, increment: int = 1) -> None:
        """
        Update progress by specified increment.

        Args:
            increment: Number of items completed
        """
        self.current += increment
        
        # Update progress every 10% or every 10 seconds
        now = datetime.now()
        progress_pct = (self.current / self.total) * 100
        
        should_update = (
            progress_pct % 10 < (progress_pct - increment / self.total * 100) % 10 or
            (now - self.last_update).seconds >= 10
        )
        
        if should_update or self.current >= self.total:
            elapsed = (now - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            
            if rate > 0 and self.current < self.total:
                eta_seconds = (self.total - self.current) / rate
                eta = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            else:
                eta = "ETA: N/A"
            
            logging.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({progress_pct:.1f}%) - {eta}"
            )
            
            self.last_update = now

    def finish(self) -> None:
        """Mark progress as complete and log final statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.total / elapsed if elapsed > 0 else 0
        
        logging.info(
            f"{self.description} completed: {self.total} items in "
            f"{int(elapsed//60)}m {int(elapsed%60)}s ({rate:.1f} items/sec)"
        )


if __name__ == "__main__":
    # Example usage
    logger = setup_logging("INFO")
    
    # Test system info
    sys_info = get_system_info()
    print("System Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # Test number formatting
    test_numbers = [123, 1234, 1234567, 1234567890]
    print("\nNumber formatting:")
    for num in test_numbers:
        print(f"  {num} -> {format_number(num)}")
    
    # Test progress tracker
    print("\nProgress tracking demo:")
    tracker = ProgressTracker(100, "Demo task")
    for i in range(0, 101, 10):
        tracker.update(10)
        import time
        time.sleep(0.1)  # Simulate work
    tracker.finish()

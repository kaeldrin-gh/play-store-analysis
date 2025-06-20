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
Data preprocessing module for Google Play Store analysis.

This module provides comprehensive data cleaning, feature engineering,
and dataset splitting functionality for app store data analysis.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_installs_column(df: DataFrame) -> DataFrame:
    """
    Convert installs column to integer format.

    Args:
        df: DataFrame with 'installs' column

    Returns:
        DataFrame with cleaned integer installs column

    Note:
        Handles various formats like "1,000+", "10M+", "Free" etc.
    """
    df = df.copy()
    
    def parse_installs(install_str: str) -> int:
        """Parse install string to integer."""
        if pd.isna(install_str) or install_str == "Free":
            return 0
            
        # Remove common characters and convert to lowercase
        clean_str = str(install_str).replace(",", "").replace("+", "").lower()
        
        # Handle multipliers
        if "k" in clean_str:
            return int(float(clean_str.replace("k", "")) * 1000)
        elif "m" in clean_str:
            return int(float(clean_str.replace("m", "")) * 1000000)
        elif "b" in clean_str:
            return int(float(clean_str.replace("b", "")) * 1000000000)
        else:
            # Try to extract number
            numbers = re.findall(r'\d+', clean_str)
            return int(numbers[0]) if numbers else 0
    
    df["installs"] = df["installs"].apply(parse_installs)
    logger.info("Cleaned installs column: converted to integer format")
    return df


def clean_price_column(df: DataFrame) -> DataFrame:
    """
    Convert price column to float format in USD.

    Args:
        df: DataFrame with 'price' column

    Returns:
        DataFrame with cleaned float price column
    """
    df = df.copy()
    
    def parse_price(price_str: str) -> float:
        """Parse price string to float."""
        if pd.isna(price_str) or price_str in ["Free", "0"]:
            return 0.0
            
        # Remove currency symbols and extract number
        clean_str = str(price_str).replace("$", "").replace("€", "").replace("£", "")
        numbers = re.findall(r'\d+\.?\d*', clean_str)
        return float(numbers[0]) if numbers else 0.0
    
    df["price"] = df["price"].apply(parse_price)
    logger.info("Cleaned price column: converted to float USD format")
    return df


def parse_permissions(df: DataFrame) -> DataFrame:
    """
    Parse and normalize permissions data.

    Args:
        df: DataFrame with 'permissions' column

    Returns:
        DataFrame with cleaned permissions list
    """
    df = df.copy()
    
    def normalize_permissions(perm_data: Any) -> List[str]:
        """Normalize permissions to list format."""
        if pd.isna(perm_data) or perm_data == "":
            return []
            
        if isinstance(perm_data, list):
            return [str(p).upper().strip() for p in perm_data]
        elif isinstance(perm_data, str):
            # Split by common delimiters
            perms = re.split(r'[,;|]', perm_data)
            return [p.upper().strip() for p in perms if p.strip()]
        else:
            return []
    
    df["permissions"] = df["permissions"].apply(normalize_permissions)
    logger.info("Parsed and normalized permissions column")
    return df


def parse_description_tokens(df: DataFrame) -> DataFrame:
    """
    Extract and tokenize description text.

    Args:
        df: DataFrame with 'description' column

    Returns:
        DataFrame with additional description-derived features
    """
    df = df.copy()
    
    # Calculate basic text metrics
    df["description_length"] = df["description"].str.len().fillna(0)
    df["description_word_count"] = df["description"].str.split().str.len().fillna(0)
    
    # Extract common keywords
    marketing_keywords = [
        "best", "top", "premium", "pro", "free", "easy", "simple", 
        "fast", "secure", "offline", "online", "new", "updated"
    ]
    
    for keyword in marketing_keywords:
        df[f"desc_has_{keyword}"] = (
            df["description"].str.lower().str.contains(keyword, na=False)
        )
    
    logger.info("Extracted description tokens and features")
    return df


def create_feature_flags(df: DataFrame) -> DataFrame:
    """
    Create boolean feature flags based on app characteristics.

    Args:
        df: DataFrame with app data

    Returns:
        DataFrame with additional boolean feature columns
    """
    df = df.copy()
    
    # In-app purchase flag
    df["has_inapp"] = (df["price"] == 0) & (df["installs"] > 10000)
    
    # Game category flag
    game_categories = ["GAME", "ARCADE", "PUZZLE", "STRATEGY", "SIMULATION"]
    df["is_game"] = df["category"].isin(game_categories)
    
    # Family-friendly flag
    family_ratings = ["Everyone", "Everyone 10+"]
    df["is_family_friendly"] = df["content_rating"].isin(family_ratings)
    
    # Popular app flag (top 20% by installs)
    install_threshold = df["installs"].quantile(0.8)
    df["is_popular"] = df["installs"] >= install_threshold
    
    # High-rated app flag
    df["is_highly_rated"] = (df["rating"] >= 4.0) & (df["rating_count"] >= 100)
    
    # Large app flag (>100MB)
    df["size_mb"] = df["size"].str.extract(r'(\d+\.?\d*)').astype(float, errors='ignore')
    df["is_large_app"] = df["size_mb"] > 100
    
    # Permission intensity flags
    df["permission_count"] = df["permissions"].apply(len)
    df["has_sensitive_permissions"] = df["permissions"].apply(
        lambda perms: any(
            perm in ["CAMERA", "LOCATION", "MICROPHONE", "CONTACTS", "SMS"]
            for perm in perms
        )
    )
    
    logger.info("Created feature flags: has_inapp, is_game, is_family_friendly, etc.")
    return df


def create_popularity_classes(df: DataFrame) -> DataFrame:
    """
    Create popularity classification based on install counts.

    Args:
        df: DataFrame with 'installs' column

    Returns:
        DataFrame with 'popularity_class' column
    """
    df = df.copy()
    
    def classify_popularity(installs: int) -> str:
        """Classify app popularity based on install count."""
        if installs <= 100000:
            return "Low"
        elif installs <= 1000000:
            return "Medium" 
        else:
            return "High"
    
    df["popularity_class"] = df["installs"].apply(classify_popularity)
    
    class_counts = df["popularity_class"].value_counts()
    logger.info(f"Created popularity classes: {dict(class_counts)}")
    return df


def deterministic_train_test_split(
    df: DataFrame, 
    id_column: str = "app_id",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Create deterministic train/validation/test split using hash of ID.

    Args:
        df: DataFrame to split
        id_column: Column name containing unique identifiers
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        random_seed: Seed for deterministic splitting

    Returns:
        Tuple of (train_df, valid_df, test_df)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    df = df.copy()
    
    def hash_to_split(app_id: str) -> str:
        """Deterministically assign split based on hash of ID."""
        hash_val = int(hashlib.md5(f"{app_id}_{random_seed}".encode()).hexdigest()[:8], 16)
        norm_hash = hash_val / (2**32)  # Normalize to [0,1]
        
        if norm_hash < train_ratio:
            return "train"
        elif norm_hash < train_ratio + valid_ratio:
            return "valid"
        else:
            return "test"
    
    df["split"] = df[id_column].astype(str).apply(hash_to_split)
    
    train_df = df[df["split"] == "train"].drop("split", axis=1)
    valid_df = df[df["split"] == "valid"].drop("split", axis=1)
    test_df = df[df["split"] == "test"].drop("split", axis=1)
    
    logger.info(
        f"Created deterministic splits: "
        f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
    )
    
    return train_df, valid_df, test_df


def clean_and_engineer_features(df: DataFrame) -> DataFrame:
    """
    Apply comprehensive data cleaning and feature engineering pipeline.

    Args:
        df: Raw DataFrame with app data

    Returns:
        Cleaned and feature-engineered DataFrame

    Note:
        This is the main preprocessing pipeline that combines all cleaning steps.
    """
    logger.info("Starting comprehensive data cleaning and feature engineering")
      # Start with a copy
    df_clean = df.copy()
    
    # Basic data type conversions and cleaning
    df_clean = clean_installs_column(df_clean)
    df_clean = clean_price_column(df_clean)
    df_clean = parse_permissions(df_clean)
    df_clean = parse_description_tokens(df_clean)
    
    # Clean rating columns before feature engineering (needed for feature flags)
    df_clean["rating"] = pd.to_numeric(df_clean["rating"], errors="coerce").fillna(0.0)
    df_clean["rating_count"] = pd.to_numeric(df_clean["rating_count"], errors="coerce").fillna(0)
    
    # Feature engineering
    df_clean = create_feature_flags(df_clean)
    df_clean = create_popularity_classes(df_clean)
    
    # Clean category names
    df_clean["category"] = df_clean["category"].str.upper().str.strip()
    df_clean["category"] = df_clean["category"].fillna("UNKNOWN")
    
    # Clean content rating
    df_clean["content_rating"] = df_clean["content_rating"].fillna("Unknown")
    
    # Remove duplicates based on app_id
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["app_id"], keep="first")
    final_count = len(df_clean)
    
    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} duplicate apps")
    
    # Final validation
    numeric_columns = ["installs", "price", "rating", "rating_count"]
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)
    
    logger.info(f"Completed preprocessing pipeline: {len(df_clean)} apps processed")
    logger.info(f"Feature columns created: {len(df_clean.columns)} total columns")
    
    return df_clean


def save_processed_data(
    df: DataFrame, 
    output_path: Path = Path("data/processed/apps.parquet")
) -> None:
    """
    Save processed DataFrame to Parquet format.

    Args:
        df: Processed DataFrame to save
        output_path: Path where to save the processed data

    Note:
        Parquet format is used for efficient storage and fast loading.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Saved processed data to {output_path}")
    
    # Save summary statistics
    summary_path = output_path.parent / "data_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Google Play Store Data Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total apps processed: {len(df):,}\n")
        f.write(f"Total features: {len(df.columns)}\n\n")
        
        f.write("Categorical Features:\n")
        categorical_cols = df.select_dtypes(include=["object", "bool"]).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            f.write(f"  {col}: {unique_count} unique values\n")
        
        f.write("\nNumerical Features:\n")
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numerical_cols:
            f.write(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}\n")
        
        f.write(f"\nMissing values per column:\n")
        missing = df.isnull().sum()
        for col, count in missing[missing > 0].items():
            f.write(f"  {col}: {count} ({count/len(df)*100:.1f}%)\n")
    
    logger.info(f"Saved data summary to {summary_path}")


if __name__ == "__main__":
    # Example usage
    from src.data_load import load_sample_data
    
    # Load sample data for testing
    df_raw = load_sample_data()
    print(f"Loaded {len(df_raw)} raw app records")
    
    # Apply preprocessing pipeline
    df_processed = clean_and_engineer_features(df_raw)
    print(f"Processed {len(df_processed)} apps with {len(df_processed.columns)} features")
    
    # Create train/test splits
    train_df, valid_df, test_df = deterministic_train_test_split(df_processed)
    print(f"Created splits: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    
    # Save processed data
    save_processed_data(df_processed)
    print("Saved processed data to data/processed/apps.parquet")

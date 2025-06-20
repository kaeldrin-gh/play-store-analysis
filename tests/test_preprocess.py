# MIT License
#
# Copyright (c) 2025 Play Store Analysis Project

"""
Tests for data preprocessing functionality.
"""

import pandas as pd
import pytest

from src.preprocess import (
    clean_installs_column,
    clean_price_column,
    parse_permissions,
    create_feature_flags,
    create_popularity_classes,
    deterministic_train_test_split,
    clean_and_engineer_features
)


class TestDataCleaning:
    """Test cases for data cleaning functions."""

    def test_clean_installs_column(self):
        """Test installs column cleaning."""
        df = pd.DataFrame({
            'installs': ['1,000+', '10M+', '500K+', 'Free', '100', None]
        })
        
        cleaned_df = clean_installs_column(df)
        
        assert cleaned_df['installs'].iloc[0] == 1000
        assert cleaned_df['installs'].iloc[1] == 10000000
        assert cleaned_df['installs'].iloc[2] == 500000
        assert cleaned_df['installs'].iloc[3] == 0  # 'Free' -> 0
        assert cleaned_df['installs'].iloc[4] == 100
        assert cleaned_df['installs'].iloc[5] == 0  # None -> 0

    def test_clean_price_column(self):
        """Test price column cleaning."""
        df = pd.DataFrame({
            'price': ['Free', '$2.99', '€5.50', '£1.99', '0', None]
        })
        
        cleaned_df = clean_price_column(df)
        
        assert cleaned_df['price'].iloc[0] == 0.0
        assert cleaned_df['price'].iloc[1] == 2.99
        assert cleaned_df['price'].iloc[2] == 5.50
        assert cleaned_df['price'].iloc[3] == 1.99
        assert cleaned_df['price'].iloc[4] == 0.0
        assert cleaned_df['price'].iloc[5] == 0.0

    def test_parse_permissions(self):
        """Test permissions parsing."""
        df = pd.DataFrame({
            'permissions': [
                ['CAMERA', 'STORAGE'],
                'CAMERA,STORAGE,LOCATION',
                'CAMERA;LOCATION',
                '',
                None
            ]
        })
        
        parsed_df = parse_permissions(df)
        
        assert parsed_df['permissions'].iloc[0] == ['CAMERA', 'STORAGE']
        assert parsed_df['permissions'].iloc[1] == ['CAMERA', 'STORAGE', 'LOCATION']
        assert parsed_df['permissions'].iloc[2] == ['CAMERA', 'LOCATION']
        assert parsed_df['permissions'].iloc[3] == []
        assert parsed_df['permissions'].iloc[4] == []

    def test_create_feature_flags(self, sample_dataframe):
        """Test feature flag creation."""
        # Clean the sample data first
        df_clean = clean_installs_column(sample_dataframe)
        df_clean = clean_price_column(df_clean)
        df_clean = parse_permissions(df_clean)
        
        flagged_df = create_feature_flags(df_clean)
        
        # Check that new columns exist
        assert 'has_inapp' in flagged_df.columns
        assert 'is_game' in flagged_df.columns
        assert 'is_family_friendly' in flagged_df.columns
        assert 'is_popular' in flagged_df.columns
        assert 'is_highly_rated' in flagged_df.columns
        assert 'permission_count' in flagged_df.columns
        assert 'has_sensitive_permissions' in flagged_df.columns
        
        # Check data types
        assert flagged_df['has_inapp'].dtype == bool
        assert flagged_df['is_game'].dtype == bool
        assert flagged_df['permission_count'].dtype in ['int64', 'int32']

    def test_create_popularity_classes(self):
        """Test popularity classification."""
        df = pd.DataFrame({
            'installs': [50000, 150000, 2000000, 10000, 5000000]
        })
        
        classified_df = create_popularity_classes(df)
        
        assert 'popularity_class' in classified_df.columns
        assert classified_df['popularity_class'].iloc[0] == 'Low'  # 50k
        assert classified_df['popularity_class'].iloc[1] == 'Medium'  # 150k
        assert classified_df['popularity_class'].iloc[2] == 'High'  # 2M
        assert classified_df['popularity_class'].iloc[3] == 'Low'  # 10k
        assert classified_df['popularity_class'].iloc[4] == 'High'  # 5M

    def test_deterministic_train_test_split(self, sample_dataframe):
        """Test deterministic dataset splitting."""
        train_df, valid_df, test_df = deterministic_train_test_split(
            sample_dataframe,
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15
        )
        
        # Check that splits are reasonable (allowing some variance due to hashing)
        total_rows = len(train_df) + len(valid_df) + len(test_df)
        assert total_rows == len(sample_dataframe)
        
        # Check that train set is largest
        assert len(train_df) > len(valid_df)
        assert len(train_df) > len(test_df)
        
        # Check no overlap in app_ids
        all_train_ids = set(train_df['app_id'])
        all_valid_ids = set(valid_df['app_id'])
        all_test_ids = set(test_df['app_id'])
        
        assert len(all_train_ids & all_valid_ids) == 0
        assert len(all_train_ids & all_test_ids) == 0
        assert len(all_valid_ids & all_test_ids) == 0

    def test_deterministic_split_reproducibility(self, sample_dataframe):
        """Test that splits are reproducible with same seed."""
        train1, valid1, test1 = deterministic_train_test_split(
            sample_dataframe, random_seed=42
        )
        
        train2, valid2, test2 = deterministic_train_test_split(
            sample_dataframe, random_seed=42
        )
        
        # Should be identical
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(valid1, valid2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_split_ratio_validation(self, sample_dataframe):
        """Test validation of split ratios."""
        with pytest.raises(ValueError):
            deterministic_train_test_split(
                sample_dataframe,
                train_ratio=0.5,
                valid_ratio=0.3,
                test_ratio=0.3  # Sum > 1.0
            )


class TestFeatureEngineering:
    """Test cases for feature engineering pipeline."""

    def test_clean_and_engineer_features(self, sample_dataframe):
        """Test complete feature engineering pipeline."""
        engineered_df = clean_and_engineer_features(sample_dataframe)
        
        # Check that original columns are preserved
        assert 'app_id' in engineered_df.columns
        assert 'name' in engineered_df.columns
        assert 'category' in engineered_df.columns
        
        # Check that new features are created
        assert 'has_inapp' in engineered_df.columns
        assert 'is_game' in engineered_df.columns
        assert 'popularity_class' in engineered_df.columns
        assert 'permission_count' in engineered_df.columns
        
        # Check data types are correct
        assert pd.api.types.is_numeric_dtype(engineered_df['installs'])
        assert pd.api.types.is_numeric_dtype(engineered_df['price'])
        assert pd.api.types.is_numeric_dtype(engineered_df['rating'])
        
        # Check that there are no duplicates
        assert engineered_df['app_id'].nunique() == len(engineered_df)

    def test_missing_value_handling(self):
        """Test handling of missing values."""
        df_with_missing = pd.DataFrame({
            'app_id': ['app1', 'app2', 'app3'],
            'name': ['App 1', None, 'App 3'],
            'category': ['GAME', 'SOCIAL', None],
            'rating': [4.5, None, 3.2],
            'rating_count': [100, 200, None],
            'installs': ['1,000+', None, '10,000+'],
            'price': ['Free', '$1.99', None],
            'content_rating': ['Everyone', None, 'Teen'],
            'size': ['50M', '30M', None],
            'permissions': [['CAMERA'], None, []],
            'last_updated': ['Jan 1, 2024', 'Jan 2, 2024', None],
            'android_version_min': ['5.0', None, '6.0'],
            'description': ['Test description', None, 'Another description'],
            'reviews': [['Good app'], None, []]
        })
        
        cleaned_df = clean_and_engineer_features(df_with_missing)
        
        # Check that missing values are handled appropriately
        assert not cleaned_df['rating'].isna().any()
        assert not cleaned_df['rating_count'].isna().any()
        assert not cleaned_df['installs'].isna().any()
        assert not cleaned_df['price'].isna().any()

    def test_description_tokenization(self):
        """Test description tokenization and feature extraction."""
        df = pd.DataFrame({
            'app_id': ['app1', 'app2'],
            'name': ['App 1', 'App 2'],
            'category': ['GAME', 'SOCIAL'],
            'rating': [4.5, 3.8],
            'rating_count': [100, 200],
            'installs': ['1,000+', '10,000+'],
            'price': ['Free', 'Free'],
            'content_rating': ['Everyone', 'Teen'],
            'size': ['50M', '30M'],
            'permissions': [['CAMERA'], ['LOCATION']],
            'last_updated': ['Jan 1, 2024', 'Jan 2, 2024'],
            'android_version_min': ['5.0', '6.0'],
            'description': [
                'This is the best free app for gaming',
                'A simple and easy social networking tool'
            ],
            'reviews': [['Good'], ['Great']]
        })
        
        processed_df = clean_and_engineer_features(df)
        
        # Check description features
        assert 'description_length' in processed_df.columns
        assert 'description_word_count' in processed_df.columns
        assert 'desc_has_best' in processed_df.columns
        assert 'desc_has_free' in processed_df.columns
        
        # Verify specific values
        assert processed_df.loc[0, 'desc_has_best'] == True
        assert processed_df.loc[0, 'desc_has_free'] == True
        assert processed_df.loc[1, 'desc_has_simple'] == True
        assert processed_df.loc[1, 'desc_has_easy'] == True

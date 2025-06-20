# MIT License
#
# Copyright (c) 2025 Play Store Analysis Project

"""
Test utilities and fixtures for Google Play Store analysis tests.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


@pytest.fixture
def sample_app_data():
    """Generate sample app data for testing."""
    return {
        'app_id': 'com.example.test',
        'name': 'Test App',
        'category': 'GAME',
        'rating': 4.2,
        'rating_count': 1500,
        'installs': '10,000+',
        'price': 'Free',
        'content_rating': 'Everyone',
        'size': '50M',
        'permissions': ['CAMERA', 'STORAGE'],
        'last_updated': 'January 1, 2024',
        'android_version_min': '5.0',
        'description': 'A fantastic test application for gaming.',
        'reviews': ['Great app!', 'Could be better.']
    }


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    data = []
    categories = ['GAME', 'SOCIAL', 'PRODUCTIVITY', 'ENTERTAINMENT']
    install_ranges = ['100+', '1,000+', '10,000+', '100,000+', '1,000,000+']
    
    for i in range(100):
        app_data = {
            'app_id': f'com.example.app{i:03d}',
            'name': f'Test App {i+1}',
            'category': categories[i % len(categories)],
            'rating': round(1 + (i % 5), 1),
            'rating_count': (i + 1) * 100,
            'installs': install_ranges[i % len(install_ranges)],
            'price': 'Free' if i % 3 == 0 else f'${i % 10 + 1}.99',
            'content_rating': 'Everyone' if i % 2 == 0 else 'Teen',
            'size': f'{(i % 100) + 10}M',
            'permissions': ['CAMERA', 'STORAGE'] if i % 2 == 0 else ['LOCATION'],
            'last_updated': 'January 1, 2024',
            'android_version_min': '5.0',
            'description': f'Description for test app {i+1}',
            'reviews': [f'Review {j+1}' for j in range(i % 3)]
        }
        data.append(app_data)
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def config_data():
    """Sample configuration data."""
    return {
        'data': {
            'raw_path': 'data/raw',
            'processed_path': 'data/processed'
        },
        'models': {
            'random_state': 42,
            'cv_folds': 5
        },
        'output': {
            'figures_path': 'reports/figures',
            'reports_path': 'reports'
        }
    }

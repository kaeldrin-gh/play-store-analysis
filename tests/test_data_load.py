# MIT License
#
# Copyright (c) 2025 Play Store Analysis Project

"""
Tests for data loading functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_load import (
    create_unified_dataset,
    download_kaggle_dataset,
    load_sample_data,
    EthicalScraper
)


class TestDataLoading:
    """Test cases for data loading functions."""

    def test_load_sample_data(self):
        """Test sample data generation."""
        df = load_sample_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert 'app_id' in df.columns
        assert 'name' in df.columns
        assert 'category' in df.columns
        assert 'rating' in df.columns

    def test_create_unified_dataset_with_sample_data(self, temp_directory):
        """Test creating unified dataset from sample data."""
        # Create a sample CSV file
        sample_df = pd.DataFrame({
            'App': ['Test App 1', 'Test App 2'],
            'Category': ['GAME', 'SOCIAL'],
            'Rating': [4.5, 3.8],
            'Reviews': [100, 50],
            'Installs': ['1,000+', '500+'],
            'Price': ['Free', '$2.99']
        })
        
        csv_path = temp_directory / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)
        
        # Test unified dataset creation
        unified_df = create_unified_dataset(temp_directory)
        
        assert isinstance(unified_df, pd.DataFrame)
        assert 'name' in unified_df.columns  # Should be renamed from 'App'
        assert 'category' in unified_df.columns
        assert 'app_id' in unified_df.columns
        assert len(unified_df) == 2

    def test_create_unified_dataset_missing_files(self, temp_directory):
        """Test error handling when no CSV files found."""
        with pytest.raises(FileNotFoundError):
            create_unified_dataset(temp_directory)

    @patch('src.data_load.KaggleApi')
    def test_download_kaggle_dataset_success(self, mock_kaggle_api, temp_directory):
        """Test successful Kaggle dataset download."""
        # Mock successful API call
        mock_api_instance = Mock()
        mock_kaggle_api.return_value = mock_api_instance
        
        # Create a mock CSV file in the destination
        (temp_directory / "test_data.csv").write_text("App,Category\nTest,GAME\n")
        
        result_path = download_kaggle_dataset(dest=temp_directory)
        
        assert result_path == temp_directory
        mock_api_instance.authenticate.assert_called_once()
        mock_api_instance.dataset_download_files.assert_called_once()

    @patch('src.data_load.KaggleApi')
    def test_download_kaggle_dataset_existing_files(self, mock_kaggle_api, temp_directory):
        """Test skipping download when files already exist."""
        # Create existing CSV file
        (temp_directory / "existing_data.csv").write_text("App,Category\nTest,GAME\n")
        
        result_path = download_kaggle_dataset(dest=temp_directory, force_download=False)
        
        assert result_path == temp_directory
        mock_kaggle_api.return_value.authenticate.assert_not_called()


class TestEthicalScraper:
    """Test cases for ethical scraper functionality."""

    def test_scraper_initialization(self):
        """Test scraper initialization."""
        scraper = EthicalScraper()
        
        assert scraper.base_delay == 1.0
        assert scraper.max_delay == 60.0
        assert scraper.backoff_factor == 2.0
        assert scraper.current_delay == 1.0
        assert scraper.session is not None

    def test_extract_app_id(self):
        """Test app ID extraction from URL."""
        test_url = "https://play.google.com/store/apps/details?id=com.example.app"
        app_id = EthicalScraper._extract_app_id(test_url)
        
        assert app_id == "com.example.app"

    def test_extract_app_id_invalid_url(self):
        """Test app ID extraction from invalid URL."""
        test_url = "https://invalid.url.com"
        app_id = EthicalScraper._extract_app_id(test_url)
        
        assert app_id == "unknown"

    def test_extract_text_safe(self):
        """Test safe text extraction from BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        html = "<h1>Test Title</h1>"
        soup = BeautifulSoup(html, "html.parser")
        
        text = EthicalScraper._extract_text_safe(soup, "h1")
        assert text == "Test Title"
        
        # Test with missing element
        text = EthicalScraper._extract_text_safe(soup, "h2")
        assert text == "Unknown"

    @patch('src.data_load.check_robots_txt')
    @patch('src.data_load.requests.Session.get')
    def test_scrape_app_details_robots_blocked(self, mock_get, mock_robots):
        """Test scraping when blocked by robots.txt."""
        mock_robots.return_value = False  # Blocked by robots.txt
        
        scraper = EthicalScraper()
        result = scraper.scrape_app_details("https://play.google.com/store/apps/details?id=test")
        
        assert result is None
        mock_get.assert_not_called()

    @patch('src.data_load.check_robots_txt')
    @patch('src.data_load.requests.Session.get')
    def test_scrape_app_details_success(self, mock_get, mock_robots):
        """Test successful app scraping."""
        mock_robots.return_value = True  # Allowed by robots.txt
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<h1>Test App</h1><div>Test description</div>"
        mock_get.return_value = mock_response
        
        scraper = EthicalScraper()
        result = scraper.scrape_app_details("https://play.google.com/store/apps/details?id=com.test.app")
        
        assert result is not None
        assert result['app_id'] == 'com.test.app'
        assert result['name'] == 'Test App'
        assert 'category' in result
        assert 'rating' in result

    def test_delay_mechanisms(self):
        """Test delay and backoff mechanisms."""
        scraper = EthicalScraper(base_delay=0.1, max_delay=1.0, backoff_factor=2.0)
        
        initial_delay = scraper.current_delay
        assert initial_delay == 0.1
        
        # Test delay update
        scraper._wait()
        assert scraper.current_delay == 0.2  # 0.1 * 2.0
        
        # Test delay reset
        scraper._reset_delay()
        assert scraper.current_delay == 0.1

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
Data acquisition module for Google Play Store analysis.

This module provides functions to download datasets from Kaggle and optionally
scrape additional data while respecting robots.txt and rate limits.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(
    dataset_slug: str = "lava18/google-play-store-apps",
    dest: Path = Path("data/raw"),
    force_download: bool = False,
) -> Path:
    """
    Download Google Play Store dataset from Kaggle.

    Args:
        dataset_slug: Kaggle dataset identifier (owner/dataset-name)
        dest: Destination directory for downloaded files
        force_download: Whether to re-download if files already exist

    Returns:
        Path to the downloaded dataset directory

    Raises:
        Exception: If Kaggle API credentials are not configured
        FileNotFoundError: If download fails
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    csv_files = list(dest.glob("*.csv"))
    if csv_files and not force_download:
        logger.info(f"Dataset files already exist in {dest}")
        return dest

    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        logger.info(f"Downloading dataset: {dataset_slug}")
        api.dataset_download_files(
            dataset_slug, path=str(dest), unzip=True, quiet=False
        )

        # Verify download
        csv_files = list(dest.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found after download")

        logger.info(f"Successfully downloaded {len(csv_files)} files to {dest}")
        return dest

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def check_robots_txt(base_url: str, user_agent: str = "*") -> bool:
    """
    Check if scraping is allowed according to robots.txt.

    Args:
        base_url: Base URL of the website
        user_agent: User agent string to check permissions for

    Returns:
        True if scraping is allowed, False otherwise
    """
    try:
        robots_url = urljoin(base_url, "/robots.txt")
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, base_url)
    except Exception as e:
        logger.warning(f"Could not check robots.txt: {e}")
        return False


class EthicalScraper:
    """
    Ethical web scraper with rate limiting and robots.txt compliance.
    
    This scraper respects robots.txt, implements exponential backoff,
    and includes proper error handling for responsible data collection.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        user_agent: str = "PlayStoreAnalysis/1.0 (Educational Research)",
    ):
        """
        Initialize the ethical scraper.

        Args:
            base_delay: Initial delay between requests (seconds)
            max_delay: Maximum delay between requests (seconds)
            backoff_factor: Multiplier for exponential backoff
            user_agent: User agent string for requests
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = base_delay
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _wait(self) -> None:
        """Apply current delay and update for next request."""
        time.sleep(self.current_delay)
        self.current_delay = min(
            self.current_delay * self.backoff_factor, self.max_delay
        )

    def _reset_delay(self) -> None:
        """Reset delay to base value after successful request."""
        self.current_delay = self.base_delay

    def scrape_app_details(self, app_url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape app details from a Google Play Store URL.

        Args:
            app_url: URL of the app on Google Play Store

        Returns:
            Dictionary containing app details or None if scraping fails

        Note:
            This is a placeholder implementation for demonstration.
            In production, you would implement actual parsing logic.
        """
        parsed_url = urlparse(app_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Check robots.txt compliance
        if not check_robots_txt(base_url):
            logger.warning(f"Scraping not allowed by robots.txt for {base_url}")
            return None

        try:
            self._wait()
            
            response = self.session.get(app_url, timeout=30)
            response.raise_for_status()
            
            # Reset delay on successful request
            self._reset_delay()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Placeholder parsing logic
            # In a real implementation, you would extract specific elements
            app_data = {
                "app_id": self._extract_app_id(app_url),
                "name": self._extract_text_safe(soup, "h1"),
                "category": "Unknown",
                "rating": 0.0,
                "rating_count": 0,
                "installs": "0",
                "price": "Free",
                "content_rating": "Everyone",
                "size": "Unknown",
                "permissions": [],
                "last_updated": "Unknown",
                "android_version_min": "Unknown",
                "description": self._extract_text_safe(soup, "div"),
                "reviews": [],
            }
            
            logger.info(f"Successfully scraped data for app: {app_data['name']}")
            return app_data
            
        except requests.RequestException as e:
            logger.error(f"Request failed for {app_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Parsing failed for {app_url}: {e}")
            return None

    @staticmethod
    def _extract_app_id(url: str) -> str:
        """Extract app ID from Google Play Store URL."""
        try:
            return url.split("id=")[1].split("&")[0]
        except (IndexError, AttributeError):
            return "unknown"

    @staticmethod
    def _extract_text_safe(soup: BeautifulSoup, selector: str) -> str:
        """Safely extract text from BeautifulSoup element."""
        try:
            element = soup.select_one(selector)
            return element.get_text(strip=True) if element else "Unknown"
        except Exception:
            return "Unknown"


def create_unified_dataset(
    kaggle_data_path: Path, scraped_data: Optional[list] = None
) -> pd.DataFrame:
    """
    Create a unified dataset from Kaggle data and optional scraped data.

    Args:
        kaggle_data_path: Path to directory containing Kaggle CSV files
        scraped_data: Optional list of dictionaries from web scraping

    Returns:
        Unified DataFrame with standardized schema

    Raises:
        FileNotFoundError: If no CSV files found in kaggle_data_path
    """
    csv_files = list(Path(kaggle_data_path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {kaggle_data_path}")

    # Load the main dataset
    main_csv = csv_files[0]  # Assume first CSV is the main apps data
    logger.info(f"Loading main dataset from {main_csv}")
    
    df = pd.read_csv(main_csv)
    
    # Standardize column names to match our schema
    column_mapping = {
        "App": "name",
        "Category": "category", 
        "Rating": "rating",
        "Reviews": "rating_count",
        "Installs": "installs",
        "Price": "price",
        "Content Rating": "content_rating",
        "Size": "size",
        "Last Updated": "last_updated",
        "Android Ver": "android_version_min",
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Add missing columns with default values
    required_columns = [
        "app_id", "name", "category", "rating", "rating_count", "installs",
        "price", "content_rating", "size", "permissions", "last_updated",
        "android_version_min", "description", "reviews"
    ]
    
    for col in required_columns:
        if col not in df.columns:
            if col == "app_id":
                df[col] = df.index.astype(str)
            elif col in ["permissions", "reviews"]:
                df[col] = [[] for _ in range(len(df))]
            elif col == "description":
                df[col] = "No description available"
            else:
                df[col] = "Unknown"
    
    # Append scraped data if available
    if scraped_data:
        scraped_df = pd.DataFrame(scraped_data)
        df = pd.concat([df, scraped_df], ignore_index=True)
        logger.info(f"Added {len(scraped_data)} scraped records")
    
    logger.info(f"Created unified dataset with {len(df)} apps")
    return df[required_columns]


def load_sample_data() -> pd.DataFrame:
    """
    Generate sample Google Play Store data for development/testing.

    Returns:
        DataFrame with sample app data following the unified schema

    Note:
        This function creates realistic sample data when real data is not available.
    """
    import random
    from datetime import datetime, timedelta
    
    categories = [
        "GAME", "SOCIAL", "PRODUCTIVITY", "ENTERTAINMENT", "EDUCATION",
        "TOOLS", "LIFESTYLE", "BUSINESS", "HEALTH_AND_FITNESS", "SHOPPING"
    ]
    
    content_ratings = ["Everyone", "Teen", "Mature 17+", "Everyone 10+"]
    
    sample_data = []
    
    for i in range(1000):  # Generate 1000 sample apps
        # Generate realistic install numbers
        install_ranges = ["1+", "10+", "100+", "1,000+", "10,000+", 
                         "100,000+", "1,000,000+", "10,000,000+"]
        
        app_data = {
            "app_id": f"com.example.app{i:04d}",
            "name": f"Sample App {i+1}",
            "category": random.choice(categories),
            "rating": round(random.uniform(1.0, 5.0), 1),
            "rating_count": random.randint(10, 100000),
            "installs": random.choice(install_ranges),
            "price": "Free" if random.random() > 0.3 else f"${random.randint(1, 50)}.99",
            "content_rating": random.choice(content_ratings),
            "size": f"{random.randint(5, 500)}M",
            "permissions": random.sample([
                "CAMERA", "LOCATION", "STORAGE", "MICROPHONE", "CONTACTS"
            ], random.randint(1, 3)),
            "last_updated": (
                datetime.now() - timedelta(days=random.randint(1, 365))
            ).strftime("%B %d, %Y"),
            "android_version_min": random.choice(["4.1", "4.4", "5.0", "6.0", "7.0"]),
            "description": f"This is a sample description for app {i+1}. "
                          f"It provides excellent functionality in the {random.choice(categories)} category.",
            "reviews": [
                f"Great app! Really useful for {random.choice(['work', 'entertainment', 'learning'])}.",
                f"Could be better. Has some issues with {random.choice(['performance', 'interface', 'features'])}."
            ][:random.randint(0, 2)]
        }
        sample_data.append(app_data)
    
    df = pd.DataFrame(sample_data)
    logger.info(f"Generated {len(df)} sample app records")
    return df


if __name__ == "__main__":
    # Example usage
    try:
        data_dir = download_kaggle_dataset()
        df = create_unified_dataset(data_dir)
        print(f"Loaded dataset with {len(df)} apps")
        print(df.head())
    except Exception as e:
        logger.warning(f"Could not load Kaggle data: {e}")
        logger.info("Generating sample data instead")
        df = load_sample_data()
        print(f"Generated sample dataset with {len(df)} apps")
        print(df.head())

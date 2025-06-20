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
Exploratory Data Analysis module for Google Play Store analysis.

This module provides comprehensive EDA functionality including automated profiling,
custom visualizations, and statistical analysis for app store data.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pandas import DataFrame
from ydata_profiling import ProfileReport

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set style for matplotlib/seaborn
plt.style.use('default')
sns.set_palette("husl")


def generate_automated_profile(
    df: DataFrame, 
    output_path: Path = Path("reports/data_profile.html"),
    title: str = "Google Play Store Apps - Data Profile"
) -> None:
    """
    Generate automated data profiling report using ydata-profiling.

    Args:
        df: DataFrame to profile
        output_path: Path to save HTML report
        title: Title for the profiling report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating automated data profile report...")
    
    # Configure profiling options for performance
    config = {
        "title": title,
        "minimal": False,
        "samples": {"head": 10, "tail": 10},
        "correlations": {"auto": {"calculate": True}},
        "missing_diagrams": {"bar": True, "matrix": True, "heatmap": True},
        "interactions": {"targets": ["rating", "installs", "popularity_class"]},
    }
    
    try:
        profile = ProfileReport(df, **config)
        profile.to_file(output_path)
        logger.info(f"Automated profile report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate profile report: {e}")
        # Fallback to basic summary
        generate_basic_summary(df, output_path.parent / "basic_summary.txt")


def generate_basic_summary(df: DataFrame, output_path: Path) -> None:
    """Generate basic data summary as fallback when profiling fails."""
    with open(output_path, "w") as f:
        f.write("Basic Data Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset shape: {df.shape}\n\n")
        f.write("Data types:\n")
        f.write(str(df.dtypes) + "\n\n")
        f.write("Basic statistics:\n")
        f.write(str(df.describe()) + "\n\n")
        f.write("Missing values:\n")
        f.write(str(df.isnull().sum()) + "\n")
    
    logger.info(f"Basic summary saved to {output_path}")


def plot_installs_distribution(
    df: DataFrame, 
    output_path: Path = Path("reports/figures/installs_distribution.png")
) -> None:
    """
    Create installs distribution plot with log scale.

    Args:
        df: DataFrame containing 'installs' column
        output_path: Path to save the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale histogram
    ax1.hist(df['installs'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Installs')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of App Installs (Linear Scale)')
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Log scale histogram  
    installs_positive = df[df['installs'] > 0]['installs']
    ax2.hist(installs_positive, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Number of Installs (Log Scale)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of App Installs (Log Scale)')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Installs distribution plot saved to {output_path}")


def plot_rating_by_category(
    df: DataFrame,
    output_path: Path = Path("reports/figures/rating_by_category.png")
) -> None:
    """
    Create boxplot of ratings by category.

    Args:
        df: DataFrame containing 'rating' and 'category' columns
        output_path: Path to save the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter valid ratings and limit categories for readability
    df_plot = df[(df['rating'] > 0) & (df['rating'] <= 5)].copy()
    top_categories = df_plot['category'].value_counts().head(15).index
    df_plot = df_plot[df_plot['category'].isin(top_categories)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Rating by category boxplot
    sns.boxplot(data=df_plot, x='category', y='rating', ax=ax1)
    ax1.set_title('App Rating Distribution by Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Rating (1-5)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Rating by content rating
    sns.boxplot(data=df_plot, x='content_rating', y='rating', ax=ax2)
    ax2.set_title('App Rating Distribution by Content Rating', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Content Rating')
    ax2.set_ylabel('Rating (1-5)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Rating by category plot saved to {output_path}")


def plot_permissions_heatmap(
    df: DataFrame,
    output_path: Path = Path("reports/figures/permissions_heatmap.png")
) -> None:
    """
    Create heatmap of permission count vs mean rating.

    Args:
        df: DataFrame containing 'permissions' and 'rating' columns
        output_path: Path to save the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate permission count and group by it
    df_perm = df.copy()
    df_perm['permission_count'] = df_perm['permissions'].apply(len)
    df_perm = df_perm[df_perm['rating'] > 0]  # Filter valid ratings
    
    # Group by permission count and calculate metrics
    perm_stats = df_perm.groupby('permission_count').agg({
        'rating': ['mean', 'std', 'count'],
        'installs': 'mean',
        'price': 'mean'
    }).round(3)
    
    # Flatten column names
    perm_stats.columns = ['_'.join(col).strip() for col in perm_stats.columns.values]
    perm_stats = perm_stats.reset_index()
    
    # Create correlation matrix for heatmap
    corr_data = df_perm[['permission_count', 'rating', 'installs', 'price']].corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap of correlations
    sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlation Matrix: Permissions, Rating, Installs, Price', 
                  fontweight='bold')
    
    # Line plot of permission count vs rating
    if len(perm_stats) > 1:
        ax2.plot(perm_stats['permission_count'], perm_stats['rating_mean'], 
                marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Permissions')
        ax2.set_ylabel('Average Rating')
        ax2.set_title('Average Rating by Permission Count', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Permissions heatmap saved to {output_path}")


def plot_correlation_matrix(
    df: DataFrame,
    output_path: Path = Path("reports/figures/correlation_matrix.png")
) -> None:
    """
    Create correlation matrix heatmap for numeric features.

    Args:
        df: DataFrame with numeric columns
        output_path: Path to save the plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with low variance or too many missing values
    numeric_cols = [col for col in numeric_cols 
                   if df[col].notna().sum() > len(df) * 0.5 and df[col].var() > 0]
    
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation matrix")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8}, fmt='.2f')
    
    ax.set_title('Correlation Matrix of Numeric Features', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Correlation matrix saved to {output_path}")


def create_interactive_plots(
    df: DataFrame,
    output_dir: Path = Path("reports/figures")
) -> None:
    """
    Create interactive plots using Plotly.

    Args:
        df: DataFrame with app data
        output_dir: Directory to save HTML plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter data for plotting
    df_plot = df[(df['rating'] > 0) & (df['installs'] > 0)].copy()
    
    # 1. Interactive scatter plot: Rating vs Installs by Category
    fig1 = px.scatter(
        df_plot.sample(min(1000, len(df_plot))),  # Sample for performance
        x='installs', y='rating',
        color='category',
        size='rating_count',
        hover_data=['name', 'price'],
        title='App Rating vs Installs by Category',
        log_x=True
    )
    fig1.update_layout(height=600, showlegend=True)
    fig1.write_html(output_dir / "interactive_rating_installs.html")
    
    # 2. Category distribution pie chart
    category_counts = df['category'].value_counts().head(10)
    fig2 = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title='Top 10 App Categories Distribution'
    )
    fig2.write_html(output_dir / "category_distribution.html")
    
    # 3. Price distribution by category
    paid_apps = df_plot[df_plot['price'] > 0]
    if len(paid_apps) > 0:
        fig3 = px.box(
            paid_apps,
            x='category',
            y='price',
            title='Price Distribution by Category (Paid Apps Only)'
        )
        fig3.update_xaxes(tickangle=45)
        fig3.update_layout(height=600)
        fig3.write_html(output_dir / "price_by_category.html")
    
    logger.info(f"Interactive plots saved to {output_dir}")


def generate_summary_statistics(df: DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary containing summary statistics
    """
    stats = {}
    
    # Basic counts
    stats['total_apps'] = len(df)
    stats['unique_categories'] = df['category'].nunique()
    stats['free_apps_pct'] = (df['price'] == 0).mean() * 100
    
    # Rating statistics
    valid_ratings = df[df['rating'] > 0]['rating']
    stats['avg_rating'] = valid_ratings.mean()
    stats['rating_std'] = valid_ratings.std()
    stats['highly_rated_pct'] = (valid_ratings >= 4.0).mean() * 100
    
    # Install statistics
    stats['median_installs'] = df[df['installs'] > 0]['installs'].median()
    stats['apps_over_1m_installs'] = (df['installs'] >= 1000000).sum()
    
    # Category insights
    stats['top_category'] = df['category'].value_counts().index[0]
    stats['top_category_count'] = df['category'].value_counts().iloc[0]
    
    # Content rating distribution
    stats['content_rating_dist'] = df['content_rating'].value_counts().to_dict()
    
    # Feature flags summary
    boolean_cols = df.select_dtypes(include=['bool']).columns
    for col in boolean_cols:
        stats[f'{col}_pct'] = df[col].mean() * 100
    
    return stats


def run_comprehensive_eda(
    df: DataFrame,
    output_dir: Path = Path("reports")
) -> None:
    """
    Run comprehensive exploratory data analysis pipeline.

    Args:
        df: DataFrame to analyze
        output_dir: Directory to save all EDA outputs
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    
    logger.info("Starting comprehensive EDA pipeline...")
    
    # 1. Generate automated profile
    try:
        generate_automated_profile(df, output_dir / "data_profile.html")
    except Exception as e:
        logger.warning(f"Automated profiling failed: {e}")
    
    # 2. Create static visualizations
    plot_installs_distribution(df, figures_dir / "installs_distribution.png")
    plot_rating_by_category(df, figures_dir / "rating_by_category.png")
    plot_permissions_heatmap(df, figures_dir / "permissions_heatmap.png")
    plot_correlation_matrix(df, figures_dir / "correlation_matrix.png")
    
    # 3. Create interactive plots
    create_interactive_plots(df, figures_dir)
    
    # 4. Generate summary statistics
    stats = generate_summary_statistics(df)
    
    # Save summary statistics
    stats_path = output_dir / "summary_statistics.txt"
    with open(stats_path, "w") as f:
        f.write("Google Play Store Apps - Summary Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Applications: {stats['total_apps']:,}\n")
        f.write(f"Unique Categories: {stats['unique_categories']}\n")
        f.write(f"Free Apps: {stats['free_apps_pct']:.1f}%\n\n")
        
        f.write("Rating Analysis:\n")
        f.write(f"  Average Rating: {stats['avg_rating']:.2f} ± {stats['rating_std']:.2f}\n")
        f.write(f"  Highly Rated Apps (≥4.0): {stats['highly_rated_pct']:.1f}%\n\n")
        
        f.write("Install Analysis:\n")
        f.write(f"  Median Installs: {stats['median_installs']:,}\n")
        f.write(f"  Apps with >1M Installs: {stats['apps_over_1m_installs']:,}\n\n")
        
        f.write("Category Analysis:\n")
        f.write(f"  Top Category: {stats['top_category']} ({stats['top_category_count']:,} apps)\n\n")
        
        f.write("Content Rating Distribution:\n")
        for rating, count in stats['content_rating_dist'].items():
            f.write(f"  {rating}: {count:,} apps\n")
        
        f.write("\nFeature Flags Summary:\n")
        for key, value in stats.items():
            if key.endswith('_pct') and 'content_rating' not in key:
                f.write(f"  {key.replace('_pct', '').replace('_', ' ').title()}: {value:.1f}%\n")
    
    logger.info(f"Summary statistics saved to {stats_path}")
    logger.info("Comprehensive EDA pipeline completed!")


if __name__ == "__main__":
    # Example usage
    try:
        # Try to load processed data
        df = pd.read_parquet("data/processed/apps.parquet")
        logger.info(f"Loaded processed data: {len(df)} apps")
    except FileNotFoundError:
        # Fallback to sample data
        from src.data_load import load_sample_data
        from src.preprocess import clean_and_engineer_features
        
        logger.info("Processed data not found, generating sample data...")
        df_raw = load_sample_data()
        df = clean_and_engineer_features(df_raw)
    
    # Run comprehensive EDA
    run_comprehensive_eda(df)
    print("EDA completed! Check the reports/ directory for outputs.")

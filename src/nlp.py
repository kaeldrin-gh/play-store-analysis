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
Natural Language Processing module for Google Play Store analysis.

This module provides sentiment analysis for user reviews and topic modeling
for app descriptions using state-of-the-art NLP techniques.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
try:
    from langdetect import detect, LangDetectError
except ImportError:
    # Fallback if langdetect is not available
    def detect(text):
        return 'en'
    class LangDetectError(Exception):
        pass
from pandas import DataFrame
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Conditional imports with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, using basic text preprocessing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class SentimentAnalyzer:
    """
    Sentiment analyzer for app reviews supporting multiple languages.
    
    Uses VADER for English reviews and TextBlob as fallback for other languages.
    Includes language detection and preprocessing capabilities.
    """

    def __init__(self):
        """Initialize sentiment analyzer with VADER and TextBlob."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        logger.info("Sentiment analyzer initialized with VADER and TextBlob")

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using langdetect.

        Args:
            text: Input text to analyze

        Returns:
            ISO language code (e.g., 'en', 'fr', 'es')
        """
        try:
            if not text or len(text.strip()) < 3:
                return 'unknown'
            return detect(text)
        except (LangDetectError, Exception):
            return 'unknown'

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned text ready for analysis
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text

    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER (optimized for English).

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg']
        }

    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob (multilingual fallback).

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # Range: -1 to 1
            subjectivity = blob.sentiment.subjectivity  # Range: 0 to 1
            
            # Convert to VADER-like format
            return {
                'compound': polarity,
                'positive': max(0, polarity),
                'neutral': 1 - abs(polarity),
                'negative': max(0, -polarity),
                'subjectivity': subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return {
                'compound': 0.0,
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'subjectivity': 0.5
            }

    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score.

        Args:
            compound_score: Compound sentiment score

        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis of text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with language, sentiment scores, and classification
        """
        # Preprocess text
        clean_text = self.preprocess_text(text)
        if not clean_text:
            return {
                'language': 'unknown',
                'compound': 0.0,
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0
            }

        # Detect language
        language = self.detect_language(clean_text)
        
        # Choose analyzer based on language
        if language == 'en':
            scores = self.analyze_sentiment_vader(clean_text)
        else:
            scores = self.analyze_sentiment_textblob(clean_text)
        
        # Add metadata
        scores['language'] = language
        scores['sentiment_label'] = self.classify_sentiment(scores['compound'])
        scores['confidence'] = abs(scores['compound'])
        
        return scores


def process_reviews_sentiment(
    df: DataFrame,
    text_column: str = 'reviews',
    output_path: Optional[Path] = None
) -> DataFrame:
    """
    Process sentiment analysis for app reviews.

    Args:
        df: DataFrame containing reviews
        text_column: Name of column containing review text
        output_path: Optional path to save results

    Returns:
        DataFrame with sentiment analysis results
    """
    analyzer = SentimentAnalyzer()
    results = []
    
    logger.info(f"Processing sentiment analysis for {len(df)} apps...")
    
    for idx, row in df.iterrows():
        app_id = row.get('app_id', idx)
        reviews = row.get(text_column, [])
        
        if not isinstance(reviews, list):
            reviews = [str(reviews)] if reviews else []
        
        app_sentiments = []
        
        for review_text in reviews:
            if isinstance(review_text, str) and len(review_text.strip()) > 5:
                sentiment = analyzer.analyze_text(review_text)
                sentiment['app_id'] = app_id
                sentiment['review_text'] = review_text[:200]  # Truncate for storage
                app_sentiments.append(sentiment)
        
        # Aggregate sentiments for the app
        if app_sentiments:
            avg_sentiment = {
                'app_id': app_id,
                'review_count': len(app_sentiments),
                'avg_compound': np.mean([s['compound'] for s in app_sentiments]),
                'avg_positive': np.mean([s['positive'] for s in app_sentiments]),
                'avg_neutral': np.mean([s['neutral'] for s in app_sentiments]),
                'avg_negative': np.mean([s['negative'] for s in app_sentiments]),
                'language_dist': pd.Series([s['language'] for s in app_sentiments]).value_counts().to_dict(),
                'sentiment_dist': pd.Series([s['sentiment_label'] for s in app_sentiments]).value_counts().to_dict()
            }
            
            # Overall sentiment classification
            avg_sentiment['overall_sentiment'] = analyzer.classify_sentiment(avg_sentiment['avg_compound'])
            results.extend(app_sentiments)
    
    if not results:
        logger.warning("No valid reviews found for sentiment analysis")
        return pd.DataFrame()
    
    # Create results DataFrame
    sentiment_df = pd.DataFrame(results)
    
    # Save results if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sentiment_df.to_csv(output_path, index=False)
        logger.info(f"Sentiment analysis results saved to {output_path}")
    
    # Log summary statistics
    sentiment_dist = sentiment_df['sentiment_label'].value_counts()
    logger.info(f"Sentiment distribution: {dict(sentiment_dist)}")
    
    return sentiment_df


class TopicModeler:
    """
    Topic modeling for app descriptions using Gensim LDA.
    
    Provides comprehensive topic modeling pipeline including preprocessing,
    model training, evaluation, and visualization.
    """

    def __init__(self, num_topics: int = 10, random_state: int = 42):
        """
        Initialize topic modeler.

        Args:
            num_topics: Number of topics to extract
            random_state: Random seed for reproducibility
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.nlp = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy English model for advanced preprocessing")
            except OSError:
                logger.warning("spaCy English model not found, using basic preprocessing")
                self.nlp = None

    def preprocess_text_basic(self, text: str) -> List[str]:
        """
        Basic text preprocessing without spaCy.

        Args:
            text: Input text to preprocess

        Returns:
            List of cleaned tokens
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and filter
        words = text.split()
        
        # Remove short words and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'app',
            'application', 'apps', 'mobile', 'phone', 'android', 'google', 'play'
        }
        
        tokens = [
            word for word in words
            if len(word) > 2 and word not in stop_words and word.isalpha()
        ]
        
        return tokens

    def preprocess_text_spacy(self, text: str) -> List[str]:
        """
        Advanced text preprocessing using spaCy.

        Args:
            text: Input text to preprocess

        Returns:
            List of lemmatized tokens
        """
        if not isinstance(text, str) or not self.nlp:
            return self.preprocess_text_basic(text)
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract lemmatized tokens
        tokens = [
            token.lemma_.lower()
            for token in doc
            if (
                not token.is_stop and
                not token.is_punct and
                not token.is_space and
                len(token.lemma_) > 2 and
                token.is_alpha
            )
        ]
        
        # Filter app-specific stop words
        app_stop_words = {'app', 'application', 'mobile', 'phone', 'android', 'google', 'play'}
        tokens = [token for token in tokens if token not in app_stop_words]
        
        return tokens

    def preprocess_descriptions(self, descriptions: List[str]) -> List[List[str]]:
        """
        Preprocess list of app descriptions.

        Args:
            descriptions: List of description texts

        Returns:
            List of tokenized and preprocessed descriptions
        """
        logger.info(f"Preprocessing {len(descriptions)} descriptions...")
        
        processed_docs = []
        for desc in descriptions:
            if self.nlp:
                tokens = self.preprocess_text_spacy(desc)
            else:
                tokens = self.preprocess_text_basic(desc)
            
            if len(tokens) >= 3:  # Only include docs with sufficient tokens
                processed_docs.append(tokens)
        
        logger.info(f"Processed {len(processed_docs)} valid descriptions")
        return processed_docs

    def train_lda_model(self, descriptions: List[str]) -> Tuple[LdaModel, float]:
        """
        Train LDA topic model on descriptions.

        Args:
            descriptions: List of app descriptions

        Returns:
            Tuple of (trained_model, coherence_score)
        """
        # Preprocess descriptions
        processed_docs = self.preprocess_descriptions(descriptions)
        
        if len(processed_docs) < 10:
            logger.error("Insufficient documents for topic modeling")
            return None, 0.0
        
        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(processed_docs)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=2, no_above=0.8)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        if not self.corpus:
            logger.error("Empty corpus after preprocessing")
            return None, 0.0
        
        logger.info(f"Training LDA model with {self.num_topics} topics...")
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=10,
            alpha='auto',
            per_word_topics=True,
            minimum_probability=0.01
        )
        
        # Calculate coherence score
        try:
            coherence_model = CoherenceModel(
                model=self.lda_model,
                texts=processed_docs,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
        except Exception as e:
            logger.warning(f"Could not calculate coherence score: {e}")
            coherence_score = 0.0
        
        logger.info(f"LDA model trained. Coherence score: {coherence_score:.3f}")
        return self.lda_model, coherence_score

    def get_topic_words(self, num_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get top words for each topic.

        Args:
            num_words: Number of top words per topic

        Returns:
            List of topics with their top words and probabilities
        """
        if not self.lda_model:
            return []
        
        topics = []
        for topic_id in range(self.num_topics):
            topic_words = self.lda_model.show_topic(topic_id, num_words)
            topics.append(topic_words)
        
        return topics

    def create_visualization(self, output_path: Path = Path("reports/topic_model.html")) -> None:
        """
        Create interactive topic model visualization using pyLDAvis.

        Args:
            output_path: Path to save HTML visualization
        """
        if not self.lda_model or not self.corpus or not self.dictionary:
            logger.error("Model not trained or components missing")
            return
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare visualization
            vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)
            
            # Save to HTML
            pyLDAvis.save_html(vis_data, str(output_path))
            
            logger.info(f"Topic model visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")


def perform_topic_modeling(
    df: DataFrame,
    description_column: str = 'description',
    num_topics: int = 10,
    output_dir: Path = Path("reports")
) -> Tuple[TopicModeler, float]:
    """
    Perform comprehensive topic modeling on app descriptions.

    Args:
        df: DataFrame containing app descriptions
        description_column: Name of column with description text
        num_topics: Number of topics to extract
        output_dir: Directory to save outputs

    Returns:
        Tuple of (trained_modeler, coherence_score)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract descriptions
    descriptions = df[description_column].dropna().tolist()
    
    if len(descriptions) < 10:
        logger.error("Insufficient descriptions for topic modeling")
        return None, 0.0
    
    # Initialize and train model
    modeler = TopicModeler(num_topics=num_topics)
    model, coherence_score = modeler.train_lda_model(descriptions)
    
    if model:
        # Save topic words
        topics = modeler.get_topic_words()
        
        topics_path = output_dir / "topics.txt"
        with open(topics_path, "w") as f:
            f.write(f"Topic Modeling Results ({num_topics} topics)\n")
            f.write(f"Coherence Score: {coherence_score:.3f}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, topic_words in enumerate(topics):
                f.write(f"Topic {i+1}:\n")
                for word, prob in topic_words:
                    f.write(f"  {word}: {prob:.3f}\n")
                f.write("\n")
        
        logger.info(f"Topic words saved to {topics_path}")
        
        # Create visualization
        modeler.create_visualization(output_dir / "topic_model.html")
        
    return modeler, coherence_score


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
    
    # Perform sentiment analysis
    logger.info("Starting sentiment analysis...")
    sentiment_df = process_reviews_sentiment(
        df, 
        output_path=Path("data/processed/reviews_sentiment.csv")
    )
    
    # Perform topic modeling
    logger.info("Starting topic modeling...")
    modeler, coherence = perform_topic_modeling(df, num_topics=10)
    
    print(f"NLP analysis completed!")
    print(f"Sentiment analysis: {len(sentiment_df)} reviews processed")
    print(f"Topic modeling coherence score: {coherence:.3f}")

"""
Tests for NLP module.

This module contains unit tests for the natural language processing functionality,
including sentiment analysis, topic modeling, and text preprocessing.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.nlp import (
    analyze_review_sentiment,
    detect_languages,
    perform_topic_modeling,
    preprocess_text_for_nlp,
    run_nlp_analysis,
)


class TestNLPFunctions:
    """Test cases for NLP functions."""

    def test_preprocess_text_for_nlp(self) -> None:
        """Test text preprocessing for NLP."""
        text = "This is a GREAT app! Really love it. 5 stars!!!"
        
        processed = preprocess_text_for_nlp(text)
        
        # Should be lowercased and cleaned
        assert processed.islower()
        assert "great" in processed
        assert "love" in processed
        # Should remove excessive punctuation
        assert "!!!" not in processed

    def test_preprocess_empty_text(self) -> None:
        """Test preprocessing empty or None text."""
        assert preprocess_text_for_nlp("") == ""
        assert preprocess_text_for_nlp(None) == ""
        assert preprocess_text_for_nlp("   ") == ""

    def test_analyze_review_sentiment(self) -> None:
        """Test sentiment analysis on reviews."""
        reviews = [
            "This app is amazing! I love it so much!",
            "Terrible app, waste of time and money.",
            "It's okay, nothing special but works fine."
        ]
        
        df = pd.DataFrame({"Reviews": reviews})
        result_df = analyze_review_sentiment(df)
        
        # Check that sentiment columns are added
        assert "sentiment_score" in result_df.columns
        assert "sentiment_label" in result_df.columns
        assert len(result_df) == len(reviews)
        
        # Check sentiment scores are floats
        assert all(isinstance(score, (int, float)) for score in result_df["sentiment_score"])

    def test_analyze_sentiment_empty_reviews(self) -> None:
        """Test sentiment analysis with empty reviews."""
        df = pd.DataFrame({"Reviews": ["", None, "   "]})
        
        result_df = analyze_review_sentiment(df)
        
        # Should handle empty reviews gracefully
        assert "sentiment_score" in result_df.columns
        assert len(result_df) == 3

    def test_detect_languages(self) -> None:
        """Test language detection."""
        texts = [
            "This is an English text",
            "Esto es un texto en español",
            "Ceci est un texte français"
        ]
        
        df = pd.DataFrame({"Reviews": texts})
        result_df = detect_languages(df)
        
        # Check that language column is added
        assert "detected_language" in result_df.columns
        assert len(result_df) == len(texts)

    def test_detect_languages_empty_text(self) -> None:
        """Test language detection with empty text."""
        df = pd.DataFrame({"Reviews": ["", None, "   "]})
        
        result_df = detect_languages(df)
        
        # Should handle empty text gracefully
        assert "detected_language" in result_df.columns
        assert len(result_df) == 3

    @patch('src.nlp.spacy.load')
    @patch('src.nlp.LdaModel')
    @patch('src.nlp.Dictionary')
    def test_perform_topic_modeling(
        self,
        mock_dictionary: Mock,
        mock_lda: Mock,
        mock_spacy: Mock
    ) -> None:
        """Test topic modeling functionality."""
        # Mock spacy nlp
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_token = Mock()
        mock_token.lemma_ = "test"
        mock_token.pos_ = "NOUN"
        mock_token.is_stop = False
        mock_token.is_alpha = True
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp
        
        # Mock dictionary and LDA
        mock_dict_instance = Mock()
        mock_dict_instance.filter_extremes = Mock()
        mock_dictionary.return_value = mock_dict_instance
        
        mock_lda_instance = Mock()
        mock_lda_instance.print_topics.return_value = [
            (0, "0.1*test + 0.05*word")
        ]
        mock_lda.return_value = mock_lda_instance
        
        texts = ["This is a test review", "Another test review"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            topics = perform_topic_modeling(texts, str(output_dir))
            
            # Check that topics are returned
            assert isinstance(topics, list)
            assert len(topics) > 0

    def test_run_nlp_analysis(self) -> None:
        """Test complete NLP analysis pipeline."""
        # Create sample dataframe
        df = pd.DataFrame({
            "Reviews": [
                "Great app, highly recommend!",
                "Not good, crashes frequently.",
                "Average app, could be better."
            ]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should not raise any exceptions
            result_df = run_nlp_analysis(df, str(output_dir))
            
            # Check that analysis columns are added
            assert "sentiment_score" in result_df.columns
            assert "sentiment_label" in result_df.columns
            assert "detected_language" in result_df.columns
            assert len(result_df) == len(df)

    def test_run_nlp_analysis_no_reviews(self) -> None:
        """Test NLP analysis with dataframe without Reviews column."""
        df = pd.DataFrame({"App": ["Test App"], "Category": ["GAME"]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should handle missing Reviews column gracefully
            with pytest.raises(KeyError):
                run_nlp_analysis(df, str(output_dir))

    def test_empty_dataframe_nlp(self) -> None:
        """Test NLP analysis with empty dataframe."""
        empty_df = pd.DataFrame()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should handle empty dataframe gracefully
            with pytest.raises((KeyError, ValueError)):
                run_nlp_analysis(empty_df, str(output_dir))

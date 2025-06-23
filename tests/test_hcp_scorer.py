#!/usr/bin/env python3
"""
Tests for HCP Scorer
"""

import unittest
import sys
from pathlib import Path
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.hcp_scorer import HCPScorer, ScoringResult

class TestHCPScorer(unittest.TestCase):
    """Test cases for HCP Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = HCPScorer()
        self.sample_transcript = """
        HCP: Hello, how are you feeling today?
        Patient: I'm really struggling with my diet and I'm feeling frustrated.
        HCP: I understand this can be challenging. Let's work together to find solutions that work for you. Can you tell me more about what's been difficult?
        Patient: I just can't seem to stick to any diet plan.
        HCP: That's completely normal. Many people find it difficult to maintain strict diets. Let's focus on making small, sustainable changes instead.
        """
    
    def test_scorer_initialization(self):
        """Test that the scorer initializes correctly."""
        self.assertIsNotNone(self.scorer)
        self.assertIsNotNone(self.scorer.model_backend)
    
    def test_transcript_preprocessing(self):
        """Test transcript preprocessing."""
        processed = self.scorer._preprocess_transcript(self.sample_transcript)
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)
    
    def test_prompt_creation(self):
        """Test that scoring prompts are created correctly."""
        prompt = self.scorer._create_scoring_prompt(self.sample_transcript)
        self.assertIsInstance(prompt, str)
        self.assertIn("Healthcare Provider", prompt)
        self.assertIn(self.sample_transcript, prompt)
    
    def test_score_validation(self):
        """Test score validation."""
        valid_scores = {
            "empathy": 4,
            "clarity": 5,
            "accuracy": 3,
            "professionalism": 4
        }
        self.assertTrue(self.scorer._validate_scores(valid_scores))
        
        invalid_scores = {
            "empathy": 6,  # Invalid score
            "clarity": 5,
            "accuracy": 3,
            "professionalism": 4
        }
        self.assertFalse(self.scorer._validate_scores(invalid_scores))
    
    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        scores = {
            "empathy": 4,
            "clarity": 5,
            "accuracy": 3,
            "professionalism": 4
        }
        weighted_score = self.scorer._calculate_weighted_score(scores)
        self.assertIsInstance(weighted_score, float)
        self.assertGreaterEqual(weighted_score, 1.0)
        self.assertLessEqual(weighted_score, 5.0)
    
    def test_json_extraction(self):
        """Test JSON extraction from model response."""
        valid_response = 'Here is my analysis: {"empathy": 4, "clarity": 5, "accuracy": 3, "professionalism": 4, "overall_score": 4.0, "confidence": 0.8, "reasoning": "Good performance", "strengths": ["Clear communication"], "areas_for_improvement": ["Could be more empathetic"]}'
        extracted = self.scorer._extract_json_from_response(valid_response)
        self.assertIsInstance(extracted, dict)
        self.assertIn("empathy", extracted)
        self.assertIn("clarity", extracted)
        self.assertIn("accuracy", extracted)
        self.assertIn("professionalism", extracted)
    
    def test_invalid_json_extraction(self):
        """Test JSON extraction with invalid response."""
        invalid_response = "This is not valid JSON"
        extracted = self.scorer._extract_json_from_response(invalid_response)
        self.assertIsInstance(extracted, dict)
        # Should return default values
        self.assertEqual(extracted["empathy"], 3)
        self.assertEqual(extracted["clarity"], 3)
    
    def test_scoring_result_creation(self):
        """Test ScoringResult object creation."""
        result = ScoringResult(
            empathy=4,
            clarity=5,
            accuracy=3,
            professionalism=4,
            overall_score=4.0,
            confidence=0.8,
            reasoning="Good performance",
            strengths=["Clear communication"],
            areas_for_improvement=["Could be more empathetic"]
        )
        
        self.assertEqual(result.empathy, 4)
        self.assertEqual(result.clarity, 5)
        self.assertEqual(result.accuracy, 3)
        self.assertEqual(result.professionalism, 4)
        self.assertEqual(result.overall_score, 4.0)
        self.assertEqual(result.confidence, 0.8)
        self.assertIn("Clear communication", result.strengths)
        self.assertIn("Could be more empathetic", result.areas_for_improvement)
    
    def test_batch_scoring(self):
        """Test batch scoring functionality."""
        transcripts = [
            (self.sample_transcript, "Dr. Smith"),
            (self.sample_transcript, "Dr. Johnson")
        ]
        
        results = self.scorer.batch_score_transcripts(transcripts)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIsInstance(result, ScoringResult)
            self.assertGreaterEqual(result.overall_score, 1.0)
            self.assertLessEqual(result.overall_score, 5.0)
    
    def test_report_generation(self):
        """Test report generation."""
        # Create sample results
        results = [
            ScoringResult(
                empathy=4, clarity=5, accuracy=3, professionalism=4,
                overall_score=4.0, confidence=0.8,
                reasoning="Good performance", strengths=[], areas_for_improvement=[]
            ),
            ScoringResult(
                empathy=3, clarity=4, accuracy=5, professionalism=3,
                overall_score=3.75, confidence=0.7,
                reasoning="Average performance", strengths=[], areas_for_improvement=[]
            )
        ]
        
        hcp_names = ["Dr. Smith", "Dr. Johnson"]
        report = self.scorer.generate_report(results, hcp_names)
        
        self.assertIsInstance(report, dict)
        self.assertIn("summary", report)
        self.assertIn("detailed_results", report)
        self.assertEqual(report["summary"]["total_transcripts"], 2)
    
    def test_empty_transcript_handling(self):
        """Test handling of empty transcripts."""
        with self.assertRaises(ValueError):
            self.scorer.score_transcript("")
        
        with self.assertRaises(ValueError):
            self.scorer.score_transcript("   ")
    
    def test_backend_info(self):
        """Test backend information retrieval."""
        info = self.scorer.get_backend_info()
        self.assertIsInstance(info, dict)
        self.assertIn("backend", info)
        self.assertIn("available", info)
        self.assertTrue(info["available"])

class TestScoringResult(unittest.TestCase):
    """Test cases for ScoringResult model."""
    
    def test_valid_scores(self):
        """Test valid score ranges."""
        # Valid scores should work
        result = ScoringResult(
            empathy=5, clarity=4, accuracy=3, professionalism=2,
            overall_score=3.5, confidence=0.8,
            reasoning="Test", strengths=[], areas_for_improvement=[]
        )
        self.assertEqual(result.empathy, 5)
        self.assertEqual(result.clarity, 4)
    
    def test_invalid_scores(self):
        """Test invalid score ranges."""
        # Scores should be between 1-5
        with self.assertRaises(ValueError):
            ScoringResult(
                empathy=6, clarity=4, accuracy=3, professionalism=2,
                overall_score=3.5, confidence=0.8,
                reasoning="Test", strengths=[], areas_for_improvement=[]
            )
        
        with self.assertRaises(ValueError):
            ScoringResult(
                empathy=0, clarity=4, accuracy=3, professionalism=2,
                overall_score=3.5, confidence=0.8,
                reasoning="Test", strengths=[], areas_for_improvement=[]
            )
    
    def test_confidence_range(self):
        """Test confidence score range."""
        # Confidence should be between 0-1
        with self.assertRaises(ValueError):
            ScoringResult(
                empathy=4, clarity=4, accuracy=3, professionalism=2,
                overall_score=3.5, confidence=1.5,
                reasoning="Test", strengths=[], areas_for_improvement=[]
            )

if __name__ == "__main__":
    unittest.main() 
#!/usr/bin/env python3
"""
Unit tests for RD Rating System
"""

import unittest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.rd_scorer import RDScorer, ScoringResult

class TestRDScorer(unittest.TestCase):
    """Test cases for RDScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_transcript = """
        RD: Hello, how are you feeling today? 
        Patient: I'm really struggling with my diet. 
        RD: I understand this can be challenging. Let's work together to find solutions that work for you. 
        What specific difficulties are you facing?
        """
        
        # Mock configuration
        self.mock_config = {
            'model': {
                'name': 'mistralai/Mistral-7B-Instruct-v0.1',
                'temperature': 0.1,
                'max_length': 4096,
                'top_p': 0.9
            },
            'scoring': {
                'dimensions': {
                    'empathy': {
                        'weight': 0.25,
                        'description': 'Ability to understand and respond to patient emotions',
                        'criteria': ['Shows genuine concern', 'Uses empathetic language']
                    },
                    'clarity': {
                        'weight': 0.25,
                        'description': 'Clear and understandable communication',
                        'criteria': ['Uses simple language', 'Explains concepts clearly']
                    },
                    'accuracy': {
                        'weight': 0.25,
                        'description': 'Correctness of nutritional information',
                        'criteria': ['Provides evidence-based recommendations']
                    },
                    'professionalism': {
                        'weight': 0.25,
                        'description': 'Maintains professional standards',
                        'criteria': ['Maintains appropriate boundaries']
                    }
                }
            },
            'data': {
                'max_transcript_length': 4000
            }
        }
    
    @patch('src.inference.rd_scorer.yaml.safe_load')
    @patch('src.inference.rd_scorer.openai')
    def test_rd_scorer_initialization(self, mock_openai, mock_yaml_load):
        """Test RDScorer initialization."""
        mock_yaml_load.return_value = self.mock_config
        
        scorer = RDScorer()
        
        self.assertIsNotNone(scorer)
        self.assertEqual(scorer.scoring_criteria, self.mock_config['scoring']['dimensions'])
        mock_openai.api_base = "http://localhost:8000/v1"
        mock_openai.api_key = "EMPTY"
    
    def test_preprocess_transcript(self):
        """Test transcript preprocessing."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            # Test normal preprocessing
            result = scorer._preprocess_transcript("  Hello   world  ")
            self.assertEqual(result, "Hello world")
            
            # Test length truncation
            long_transcript = "A" * 5000
            result = scorer._preprocess_transcript(long_transcript)
            self.assertIn("[truncated]", result)
            self.assertLessEqual(len(result), 4000 + len("... [truncated]"))
    
    def test_calculate_weighted_score(self):
        """Test weighted score calculation."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            scores = {
                'empathy': 4,
                'clarity': 3,
                'accuracy': 5,
                'professionalism': 4
            }
            
            weighted_score = scorer._calculate_weighted_score(scores)
            expected_score = (4 * 0.25 + 3 * 0.25 + 5 * 0.25 + 4 * 0.25) / 1.0
            self.assertEqual(weighted_score, round(expected_score, 2))
    
    def test_validate_scores(self):
        """Test score validation."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            # Valid scores
            valid_scores = {
                'empathy': 4,
                'clarity': 3,
                'accuracy': 5,
                'professionalism': 4
            }
            self.assertTrue(scorer._validate_scores(valid_scores))
            
            # Invalid scores - missing field
            invalid_scores = {
                'empathy': 4,
                'clarity': 3,
                'accuracy': 5
                # Missing professionalism
            }
            self.assertFalse(scorer._validate_scores(invalid_scores))
            
            # Invalid scores - out of range
            invalid_scores = {
                'empathy': 6,  # Out of range
                'clarity': 3,
                'accuracy': 5,
                'professionalism': 4
            }
            self.assertFalse(scorer._validate_scores(invalid_scores))
    
    def test_extract_json_from_response(self):
        """Test JSON extraction from model response."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            # Valid JSON response
            response = 'Here is my evaluation: {"empathy": 4, "clarity": 3, "accuracy": 5, "professionalism": 4}'
            result = scorer._extract_json_from_response(response)
            expected = {"empathy": 4, "clarity": 3, "accuracy": 5, "professionalism": 4}
            self.assertEqual(result, expected)
            
            # Invalid response
            with self.assertRaises(ValueError):
                scorer._extract_json_from_response("No JSON here")
    
    @patch('src.inference.rd_scorer.openai')
    def test_get_model_response(self, mock_openai):
        """Test model response generation."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"empathy": 4, "clarity": 3, "accuracy": 5, "professionalism": 4}'
            mock_openai.ChatCompletion.create.return_value = mock_response
            
            result = scorer._get_model_response("Test prompt")
            self.assertEqual(result, '{"empathy": 4, "clarity": 3, "accuracy": 5, "professionalism": 4}')
    
    @patch('src.inference.rd_scorer.openai')
    def test_score_transcript(self, mock_openai):
        """Test complete transcript scoring."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '''
            {
                "empathy": 4,
                "clarity": 3,
                "accuracy": 5,
                "professionalism": 4,
                "overall_score": 4.0,
                "confidence": 0.85,
                "reasoning": "Good empathy shown",
                "strengths": ["Shows empathy"],
                "areas_for_improvement": ["Could be clearer"]
            }
            '''
            mock_openai.ChatCompletion.create.return_value = mock_response
            
            result = scorer.score_transcript(self.sample_transcript, "Dr. Test")
            
            self.assertIsInstance(result, ScoringResult)
            self.assertEqual(result.empathy, 4)
            self.assertEqual(result.clarity, 3)
            self.assertEqual(result.accuracy, 5)
            self.assertEqual(result.professionalism, 4)
            self.assertEqual(result.overall_score, 4.0)
            self.assertEqual(result.confidence, 0.85)
    
    def test_batch_score_transcripts(self):
        """Test batch transcript scoring."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            # Mock the score_transcript method
            mock_result = ScoringResult(
                empathy=4, clarity=3, accuracy=5, professionalism=4,
                overall_score=4.0, confidence=0.85,
                reasoning="Test reasoning",
                strengths=["Test strength"],
                areas_for_improvement=["Test improvement"]
            )
            
            with patch.object(scorer, 'score_transcript', return_value=mock_result):
                transcripts = [
                    (self.sample_transcript, "Dr. Test1"),
                    (self.sample_transcript, "Dr. Test2")
                ]
                
                results = scorer.batch_score_transcripts(transcripts)
                
                self.assertEqual(len(results), 2)
                self.assertIsInstance(results[0], ScoringResult)
                self.assertIsInstance(results[1], ScoringResult)
    
    def test_generate_report(self):
        """Test report generation."""
        with patch('src.inference.rd_scorer.yaml.safe_load') as mock_yaml_load:
            mock_yaml_load.return_value = self.mock_config
            scorer = RDScorer()
            
            # Create sample results
            results = [
                ScoringResult(
                    empathy=4, clarity=3, accuracy=5, professionalism=4,
                    overall_score=4.0, confidence=0.85,
                    reasoning="Test reasoning 1",
                    strengths=["Test strength 1"],
                    areas_for_improvement=["Test improvement 1"]
                ),
                ScoringResult(
                    empathy=5, clarity=4, accuracy=4, professionalism=5,
                    overall_score=4.5, confidence=0.9,
                    reasoning="Test reasoning 2",
                    strengths=["Test strength 2"],
                    areas_for_improvement=["Test improvement 2"]
                )
            ]
            
            rd_names = ["Dr. Test1", "Dr. Test2"]
            
            report = scorer.generate_report(results, rd_names)
            
            self.assertIn('summary', report)
            self.assertIn('detailed_results', report)
            self.assertEqual(report['summary']['total_evaluations'], 2)
            self.assertEqual(len(report['detailed_results']), 2)
            self.assertEqual(report['summary']['average_scores']['overall'], 4.25)

class TestScoringResult(unittest.TestCase):
    """Test cases for ScoringResult model."""
    
    def test_valid_scoring_result(self):
        """Test valid ScoringResult creation."""
        result = ScoringResult(
            empathy=4,
            clarity=3,
            accuracy=5,
            professionalism=4,
            overall_score=4.0,
            confidence=0.85,
            reasoning="Test reasoning",
            strengths=["Test strength"],
            areas_for_improvement=["Test improvement"]
        )
        
        self.assertEqual(result.empathy, 4)
        self.assertEqual(result.clarity, 3)
        self.assertEqual(result.accuracy, 5)
        self.assertEqual(result.professionalism, 4)
        self.assertEqual(result.overall_score, 4.0)
        self.assertEqual(result.confidence, 0.85)
    
    def test_invalid_scores(self):
        """Test ScoringResult validation."""
        # Test score out of range
        with self.assertRaises(ValueError):
            ScoringResult(
                empathy=6,  # Invalid - should be 1-5
                clarity=3,
                accuracy=5,
                professionalism=4,
                overall_score=4.0,
                confidence=0.85,
                reasoning="Test reasoning",
                strengths=[],
                areas_for_improvement=[]
            )
        
        # Test confidence out of range
        with self.assertRaises(ValueError):
            ScoringResult(
                empathy=4,
                clarity=3,
                accuracy=5,
                professionalism=4,
                overall_score=4.0,
                confidence=1.5,  # Invalid - should be 0-1
                reasoning="Test reasoning",
                strengths=[],
                areas_for_improvement=[]
            )

if __name__ == '__main__':
    unittest.main() 
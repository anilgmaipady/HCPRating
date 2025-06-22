#!/usr/bin/env python3
"""
RD Scorer - Main module for rating Registered Dietitians based on telehealth transcripts
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys
from datetime import datetime
import yaml
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class ScoringResult(BaseModel):
    """Model for scoring results."""
    empathy: int = Field(..., ge=1, le=5, description="Empathy score (1-5)")
    clarity: int = Field(..., ge=1, le=5, description="Clarity score (1-5)")
    accuracy: int = Field(..., ge=1, le=5, description="Accuracy score (1-5)")
    professionalism: int = Field(..., ge=1, le=5, description="Professionalism score (1-5)")
    overall_score: float = Field(..., ge=1.0, le=5.0, description="Weighted overall score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in scoring")
    reasoning: str = Field(..., description="Detailed reasoning for scores")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas for improvement")

class RDScorer:
    """RD Rating System using local model inference."""
    
    def __init__(self, config_path: str = None):
        """Initialize the RD Scorer."""
        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize local model
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Use CPU for Mac compatibility
        
        # Try to load local model
        model_path = "/Users/anilmaipady/ai/RD-RANK/models/mistral-7b-instruct"
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading local model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Local model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")
                self.model = None
                self.tokenizer = None
        
        # Fallback to OpenAI API if local model fails
        if self.model is None:
            logger.info("Using OpenAI API as fallback")
            openai.api_base = "http://localhost:8000/v1"
            openai.api_key = "dummy-key"
            self.use_openai = True
        else:
            self.use_openai = False
        
        # Load scoring criteria
        self.scoring_criteria = self.config.get('scoring', {}).get('dimensions', {})
        
    def _create_scoring_prompt(self, transcript: str) -> str:
        """Create the scoring prompt for the model."""
        criteria_text = ""
        for dimension, config in self.scoring_criteria.items():
            criteria_text += f"\n{dimension.title()}:\n"
            criteria_text += f"  Description: {config.get('description', '')}\n"
            criteria_text += "  Criteria:\n"
            for criterion in config.get('criteria', []):
                criteria_text += f"    - {criterion}\n"
        
        prompt = f"""You are an expert evaluator of Registered Dietitians conducting telehealth sessions. 

Analyze the following transcript and rate the RD across four dimensions on a scale of 1-5:

{criteria_text}

Scoring Guidelines:
- 1: Poor - Significant issues, needs immediate improvement
- 2: Below Average - Several areas need improvement
- 3: Average - Meets basic standards, some room for improvement
- 4: Good - Above average performance, minor areas for improvement
- 5: Excellent - Outstanding performance, exemplary standards

Transcript:
{transcript}

Please provide your evaluation in the following JSON format:
{{
    "empathy": <score>,
    "clarity": <score>,
    "accuracy": <score>,
    "professionalism": <score>,
    "overall_score": <weighted_average>,
    "confidence": <confidence_0_to_1>,
    "reasoning": "<detailed_explanation>",
    "strengths": ["<strength1>", "<strength2>", ...],
    "areas_for_improvement": ["<area1>", "<area2>", ...]
}}

Focus on specific examples from the transcript to support your scores."""
        
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from model response."""
        try:
            # Find JSON pattern in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Error parsing JSON from response: {e}")
            logger.error(f"Response: {response}")
            raise
    
    def _calculate_weighted_score(self, scores: Dict[str, int]) -> float:
        """Calculate weighted overall score."""
        weighted_sum = 0
        total_weight = 0
        
        for dimension, config in self.scoring_criteria.items():
            weight = config.get('weight', 0.25)
            score = scores.get(dimension, 3)
            weighted_sum += score * weight
            total_weight += weight
        
        return round(weighted_sum / total_weight, 2) if total_weight > 0 else 3.0
    
    def _validate_scores(self, scores: Dict[str, Any]) -> bool:
        """Validate that scores are within expected ranges."""
        required_fields = ['empathy', 'clarity', 'accuracy', 'professionalism']
        
        for field in required_fields:
            if field not in scores:
                return False
            score = scores[field]
            if not isinstance(score, int) or score < 1 or score > 5:
                return False
        
        return True
    
    def score_transcript(self, transcript: str, rd_name: Optional[str] = None) -> ScoringResult:
        """Score a telehealth transcript for RD evaluation."""
        try:
            logger.info(f"Scoring transcript for RD: {rd_name or 'Unknown'}")
            
            # Preprocess transcript
            processed_transcript = self._preprocess_transcript(transcript)
            
            # Create scoring prompt
            prompt = self._create_scoring_prompt(processed_transcript)
            
            # Get model response
            response = self._get_model_response(prompt)
            
            # Extract and validate scores
            scores = self._extract_json_from_response(response)
            
            if not self._validate_scores(scores):
                raise ValueError("Invalid scores returned by model")
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(scores)
            
            # Create scoring result
            result = ScoringResult(
                empathy=scores['empathy'],
                clarity=scores['clarity'],
                accuracy=scores['accuracy'],
                professionalism=scores['professionalism'],
                overall_score=overall_score,
                confidence=scores.get('confidence', 0.8),
                reasoning=scores.get('reasoning', ''),
                strengths=scores.get('strengths', []),
                areas_for_improvement=scores.get('areas_for_improvement', [])
            )
            
            logger.info(f"Scoring completed. Overall score: {overall_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error scoring transcript: {e}")
            raise
    
    def _preprocess_transcript(self, transcript: str) -> str:
        """Preprocess transcript for analysis."""
        # Remove extra whitespace
        transcript = re.sub(r'\s+', ' ', transcript.strip())
        
        # Limit length if needed
        max_length = self.config.get('data', {}).get('max_transcript_length', 4000)
        if len(transcript) > max_length:
            transcript = transcript[:max_length] + "... [truncated]"
        
        return transcript
    
    def _get_model_response(self, prompt: str) -> str:
        """Get response from the model."""
        try:
            if self.use_openai:
                response = openai.ChatCompletion.create(
                    model="mistral",
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of Registered Dietitians."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.get('model', {}).get('temperature', 0.1),
                    max_tokens=self.config.get('model', {}).get('max_length', 4096),
                    top_p=self.config.get('model', {}).get('top_p', 0.9)
                )
                return response.choices[0].message.content
            else:
                # Use local model
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 512,
                        temperature=self.config.get('model', {}).get('temperature', 0.1),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                return response
            
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            raise
    
    def batch_score_transcripts(self, transcripts: List[Tuple[str, Optional[str]]]) -> List[ScoringResult]:
        """Score multiple transcripts in batch."""
        results = []
        
        for i, (transcript, rd_name) in enumerate(transcripts):
            try:
                logger.info(f"Processing transcript {i+1}/{len(transcripts)}")
                result = self.score_transcript(transcript, rd_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing transcript {i+1}: {e}")
                # Create a default result for failed scoring
                default_result = ScoringResult(
                    empathy=3, clarity=3, accuracy=3, professionalism=3,
                    overall_score=3.0, confidence=0.0,
                    reasoning=f"Scoring failed: {str(e)}",
                    strengths=[], areas_for_improvement=[]
                )
                results.append(default_result)
        
        return results
    
    def generate_report(self, results: List[ScoringResult], rd_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a comprehensive report from scoring results."""
        if not results:
            return {}
        
        # Calculate statistics
        avg_scores = {
            'empathy': sum(r.empathy for r in results) / len(results),
            'clarity': sum(r.clarity for r in results) / len(results),
            'accuracy': sum(r.accuracy for r in results) / len(results),
            'professionalism': sum(r.professionalism for r in results) / len(results),
            'overall': sum(r.overall_score for r in results) / len(results)
        }
        
        # Find top performers
        top_performers = sorted(
            [(i, r.overall_score) for i, r in enumerate(results)],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        report = {
            'summary': {
                'total_evaluations': len(results),
                'average_scores': {k: round(v, 2) for k, v in avg_scores.items()},
                'top_performers': [
                    {
                        'rank': i + 1,
                        'rd_name': rd_names[idx] if rd_names and idx < len(rd_names) else f"RD_{idx+1}",
                        'score': score
                    }
                    for i, (idx, score) in enumerate(top_performers)
                ]
            },
            'detailed_results': [
                {
                    'rd_name': rd_names[i] if rd_names and i < len(rd_names) else f"RD_{i+1}",
                    'scores': {
                        'empathy': r.empathy,
                        'clarity': r.clarity,
                        'accuracy': r.accuracy,
                        'professionalism': r.professionalism,
                        'overall': r.overall_score
                    },
                    'confidence': r.confidence,
                    'reasoning': r.reasoning,
                    'strengths': r.strengths,
                    'areas_for_improvement': r.areas_for_improvement
                }
                for i, r in enumerate(results)
            ]
        }
        
        return report 
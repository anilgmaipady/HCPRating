#!/usr/bin/env python3
"""
Local Model Scorer - Direct model inference without vLLM server
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pydantic import BaseModel, Field

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

class LocalModelScorer:
    """Main class for scoring Registered Dietitians using local model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Local Model Scorer."""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"
        self.config = self._load_config(config_path)
        
        # Load model and tokenizer
        self._load_model()
        
        # Load scoring criteria
        self.scoring_criteria = self.config.get('scoring', {}).get('dimensions', {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
    
    def _load_model(self):
        """Load the local model and tokenizer."""
        try:
            model_config = self.config.get('model', {})
            model_path = model_config.get('name', 'models/mistral-7b-instruct')
            
            self.logger.info(f"Loading model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization config or dtype
            quantization_config = None
            torch_dtype = torch.float16
            load_in_4bit = model_config.get('load_in_4bit', False)
            
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
                # On Mac/CPU, use float16 if possible, else fallback to float32
                if not torch.cuda.is_available():
                    import platform
                    if platform.system() == "Darwin":
                        torch_dtype = torch.float32
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _create_scoring_prompt(self, transcript: str) -> str:
        """Create the scoring prompt for the model."""
        criteria_text = ""
        for dimension, config in self.scoring_criteria.items():
            criteria_text += f"\n{dimension.title()}:\n"
            criteria_text += f"  Description: {config.get('description', '')}\n"
            criteria_text += "  Criteria:\n"
            for criterion in config.get('criteria', []):
                criteria_text += f"    - {criterion}\n"
        
        prompt = f"""<s>[INST] You are an expert evaluator of Registered Dietitians conducting telehealth sessions. 

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

Focus on specific examples from the transcript to support your scores. [/INST]"""
        
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
            self.logger.error(f"Error parsing JSON from response: {e}")
            self.logger.error(f"Response: {response}")
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
    
    def _get_model_response(self, prompt: str) -> str:
        """Get response from the local model."""
        try:
            model_config = self.config.get('model', {})
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_config.get('max_length', 4096)
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response with optimized settings for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Reduced from 1024 for faster inference
                    temperature=model_config.get('temperature', 0.1),
                    top_p=model_config.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Prevent repetitive output
                    early_stopping=True,  # Stop when EOS token is generated
                    num_beams=1  # Use greedy decoding for speed
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting model response: {e}")
            raise
    
    def score_transcript(self, transcript: str, rd_name: Optional[str] = None) -> ScoringResult:
        """Score a telehealth transcript for RD evaluation."""
        try:
            self.logger.info(f"Scoring transcript for RD: {rd_name or 'Unknown'}")
            
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
            
            self.logger.info(f"Scoring completed. Overall score: {overall_score}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error scoring transcript: {e}")
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
    
    def batch_score_transcripts(self, transcripts: List[Tuple[str, Optional[str]]]) -> List[ScoringResult]:
        """Score multiple transcripts in batch."""
        results = []
        
        for i, (transcript, rd_name) in enumerate(transcripts):
            try:
                self.logger.info(f"Processing transcript {i+1}/{len(transcripts)}")
                result = self.score_transcript(transcript, rd_name)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing transcript {i+1}: {e}")
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
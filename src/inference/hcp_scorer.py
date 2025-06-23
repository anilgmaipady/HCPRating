#!/usr/bin/env python3
"""
HCP Scorer - Main module for rating Healthcare Providers based on telehealth transcripts
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

# Import Ollama client
try:
    from .ollama_client import OllamaClient, OllamaManager, check_ollama_availability
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaClient = None
    OllamaManager = None
    check_ollama_availability = None

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

class HCPScorer:
    """HCP Rating System using multiple model backends."""
    
    def __init__(self, config_path: str = None, model_backend: str = "auto"):
        """Initialize the HCP Scorer.
        
        Args:
            config_path: Path to configuration file
            model_backend: Model backend to use ("auto", "ollama", "vllm", "openai", "local")
        """
        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model backends
        self.model = None
        self.tokenizer = None
        self.ollama_client = None
        self.device = "cpu"  # Use CPU for Mac compatibility
        
        # Determine which backend to use
        self.model_backend = self._select_backend(model_backend)
        self._initialize_backend()
        
        # Load scoring criteria
        self.scoring_criteria = self.config.get('scoring', {}).get('dimensions', {})
        
    def _select_backend(self, model_backend: str) -> str:
        """Select the best available model backend."""
        if model_backend != "auto":
            return model_backend
        
        # Priority order: Ollama > vLLM > Local > OpenAI
        if OLLAMA_AVAILABLE and check_ollama_availability():
            logger.info("Ollama backend selected")
            return "ollama"
        
        # Check for vLLM server
        try:
            import requests
            response = requests.get("http://localhost:8000/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM backend selected")
                return "vllm"
        except:
            pass
        
        # Check for local model
        model_path = project_root / "models" / "mistral-7b-instruct"
        if model_path.exists():
            logger.info("Local model backend selected")
            return "local"
        
        # Fallback to OpenAI
        logger.info("OpenAI backend selected (fallback)")
        return "openai"
    
    def _initialize_backend(self):
        """Initialize the selected model backend."""
        if self.model_backend == "ollama":
            self._initialize_ollama()
        elif self.model_backend == "vllm":
            self._initialize_vllm()
        elif self.model_backend == "local":
            self._initialize_local()
        elif self.model_backend == "openai":
            self._initialize_openai()
        else:
            raise ValueError(f"Unknown model backend: {self.model_backend}")
    
    def _initialize_ollama(self):
        """Initialize Ollama backend."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama client not available. Install with: pip install requests")
        
        ollama_config = self.config.get('ollama', {})
        base_url = ollama_config.get('base_url', 'http://localhost:11434')
        model_name = ollama_config.get('model_name', 'mistral')
        
        # Create Ollama manager and ensure model is available
        manager = OllamaManager(base_url)
        if not manager.ensure_model_available(model_name):
            raise RuntimeError(f"Failed to ensure model {model_name} is available in Ollama")
        
        self.ollama_client = manager.get_client(model_name)
        logger.info(f"Ollama backend initialized with model: {model_name}")
    
    def _initialize_vllm(self):
        """Initialize vLLM backend."""
        try:
            openai.api_base = "http://localhost:8000/v1"
            openai.api_key = "dummy-key"
            self.use_openai = True
            logger.info("vLLM backend initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM backend: {e}")
            raise
    
    def _initialize_local(self):
        """Initialize local model backend."""
        model_path = project_root / "models" / "mistral-7b-instruct"
        if not model_path.exists():
            raise FileNotFoundError(f"Local model not found at {model_path}")
        
        try:
            logger.info(f"Loading local model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Local model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def _initialize_openai(self):
        """Initialize OpenAI backend."""
        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "dummy-key"
        self.use_openai = True
        logger.info("OpenAI backend initialized")
        
    def _create_scoring_prompt(self, transcript: str) -> str:
        """Create the scoring prompt for the model."""
        criteria_text = ""
        for dimension, config in self.scoring_criteria.items():
            criteria_text += f"\n{dimension.title()}:\n"
            criteria_text += f"  Description: {config.get('description', '')}\n"
            criteria_text += "  Criteria:\n"
            for criterion in config.get('criteria', []):
                criteria_text += f"    - {criterion}\n"
        
        prompt = f"""You are an expert evaluator of Healthcare Providers conducting telehealth sessions. 

Analyze the following transcript and rate the HCP across four dimensions on a scale of 1-5:

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

Ensure all scores are integers between 1-5, confidence is between 0-1, and overall_score is the weighted average of the four dimensions."""

        return prompt
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from model response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, return default structure
        logger.warning("Could not extract valid JSON from response")
        return {
            "empathy": 3,
            "clarity": 3,
            "accuracy": 3,
            "professionalism": 3,
            "overall_score": 3.0,
            "confidence": 0.5,
            "reasoning": "Unable to parse model response",
            "strengths": [],
            "areas_for_improvement": ["Unable to analyze response"]
        }
    
    def _calculate_weighted_score(self, scores: Dict[str, int]) -> float:
        """Calculate weighted overall score."""
        weights = {
            'empathy': 0.25,
            'clarity': 0.25,
            'accuracy': 0.25,
            'professionalism': 0.25
        }
        
        weighted_sum = sum(scores[dim] * weights[dim] for dim in weights.keys())
        return round(weighted_sum, 2)
    
    def _validate_scores(self, scores: Dict[str, Any]) -> bool:
        """Validate scoring results."""
        required_fields = ['empathy', 'clarity', 'accuracy', 'professionalism']
        
        for field in required_fields:
            if field not in scores:
                return False
            score = scores[field]
            if not isinstance(score, int) or score < 1 or score > 5:
                return False
        
        return True
    
    def score_transcript(self, transcript: str, hcp_name: Optional[str] = None) -> ScoringResult:
        """Score a telehealth transcript for HCP evaluation."""
        if not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        
        logger.info(f"Scoring transcript for HCP: {hcp_name or 'Unknown'} using {self.model_backend} backend")
        
        # Preprocess transcript
        processed_transcript = self._preprocess_transcript(transcript)
        
        # Create scoring prompt
        prompt = self._create_scoring_prompt(processed_transcript)
        
        # Get model response
        response = self._get_model_response(prompt)
        
        # Extract and validate scores
        scores = self._extract_json_from_response(response)
        
        if not self._validate_scores(scores):
            logger.warning("Invalid scores received, using defaults")
            scores = {
                "empathy": 3,
                "clarity": 3,
                "accuracy": 3,
                "professionalism": 3,
                "overall_score": 3.0,
                "confidence": 0.5,
                "reasoning": "Default scores due to validation failure",
                "strengths": [],
                "areas_for_improvement": ["Unable to analyze transcript"]
            }
        
        # Calculate weighted score if not provided
        if 'overall_score' not in scores:
            scores['overall_score'] = self._calculate_weighted_score(scores)
        
        # Create result object
        result = ScoringResult(
            empathy=scores['empathy'],
            clarity=scores['clarity'],
            accuracy=scores['accuracy'],
            professionalism=scores['professionalism'],
            overall_score=scores['overall_score'],
            confidence=scores.get('confidence', 0.8),
            reasoning=scores.get('reasoning', ''),
            strengths=scores.get('strengths', []),
            areas_for_improvement=scores.get('areas_for_improvement', [])
        )
        
        return result
    
    def _preprocess_transcript(self, transcript: str) -> str:
        """Preprocess transcript for scoring."""
        # Remove extra whitespace
        transcript = re.sub(r'\s+', ' ', transcript.strip())
        
        # Limit length if too long
        max_length = self.config.get('data', {}).get('max_transcript_length', 4000)
        if len(transcript) > max_length:
            transcript = transcript[:max_length] + "..."
        
        return transcript
    
    def _get_model_response(self, prompt: str) -> str:
        """Get response from the selected model backend."""
        if self.model_backend == "ollama":
            return self._get_ollama_response(prompt)
        elif self.model_backend == "vllm":
            return self._get_vllm_response(prompt)
        elif self.model_backend == "local":
            return self._get_local_response(prompt)
        elif self.model_backend == "openai":
            return self._get_openai_response(prompt)
        else:
            raise ValueError(f"Unknown model backend: {self.model_backend}")
    
    def _get_ollama_response(self, prompt: str) -> str:
        """Get response from Ollama."""
        return self.ollama_client.generate(prompt)
    
    def _get_vllm_response(self, prompt: str) -> str:
        """Get response from vLLM server."""
        try:
            response = openai.ChatCompletion.create(
                model="mistral-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            raise
    
    def _get_local_response(self, prompt: str) -> str:
        """Get response from local model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):]  # Remove the input prompt
        except Exception as e:
            logger.error(f"Local model inference failed: {e}")
            raise
    
    def _get_openai_response(self, prompt: str) -> str:
        """Get response from OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise
    
    def batch_score_transcripts(self, transcripts: List[Tuple[str, Optional[str]]]) -> List[ScoringResult]:
        """Score multiple transcripts in batch."""
        results = []
        
        for i, (transcript, hcp_name) in enumerate(transcripts):
            try:
                logger.info(f"Processing transcript {i+1}/{len(transcripts)}")
                result = self.score_transcript(transcript, hcp_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to score transcript {i+1}: {e}")
                # Create default result for failed transcript
                default_result = ScoringResult(
                    empathy=3,
                    clarity=3,
                    accuracy=3,
                    professionalism=3,
                    overall_score=3.0,
                    confidence=0.0,
                    reasoning=f"Failed to analyze transcript: {str(e)}",
                    strengths=[],
                    areas_for_improvement=["Analysis failed"]
                )
                results.append(default_result)
        
        return results
    
    def generate_report(self, results: List[ScoringResult], hcp_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive report from scoring results."""
        if not results:
            return {"error": "No results to generate report from"}
        
        # Calculate statistics
        total_results = len(results)
        avg_scores = {
            'empathy': sum(r.empathy for r in results) / total_results,
            'clarity': sum(r.clarity for r in results) / total_results,
            'accuracy': sum(r.accuracy for r in results) / total_results,
            'professionalism': sum(r.professionalism for r in results) / total_results,
            'overall': sum(r.overall_score for r in results) / total_results
        }
        
        # Find top performers
        sorted_results = sorted(enumerate(results), key=lambda x: x[1].overall_score, reverse=True)
        top_performers = []
        
        for idx, result in sorted_results[:5]:  # Top 5
            hcp_name = hcp_names[idx] if hcp_names and idx < len(hcp_names) else f"HCP_{idx+1}"
            top_performers.append({
                'rank': len(top_performers) + 1,
                'hcp_name': hcp_name,
                'score': result.overall_score,
                'empathy': result.empathy,
                'clarity': result.clarity,
                'accuracy': result.accuracy,
                'professionalism': result.professionalism
            })
        
        # Generate detailed results
        detailed_results = []
        for i, result in enumerate(results):
            hcp_name = hcp_names[i] if hcp_names and i < len(hcp_names) else f"HCP_{i+1}"
            detailed_results.append({
                'hcp_name': hcp_name,
                'empathy': result.empathy,
                'clarity': result.clarity,
                'accuracy': result.accuracy,
                'professionalism': result.professionalism,
                'overall_score': result.overall_score,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'strengths': result.strengths,
                'areas_for_improvement': result.areas_for_improvement
            })
        
        return {
            'summary': {
                'total_transcripts': total_results,
                'average_scores': avg_scores,
                'top_performers': top_performers
            },
            'detailed_results': detailed_results,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            'backend': self.model_backend,
            'available': True,
            'model_name': self.config.get('ollama', {}).get('model_name', 'unknown') if self.model_backend == 'ollama' else 'unknown'
        } 
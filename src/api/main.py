#!/usr/bin/env python3
"""
FastAPI REST API for HCP Rating System
"""

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
import yaml
import pandas as pd
from datetime import datetime

from src.inference.hcp_scorer import HCPScorer, ScoringResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title=config['api']['title'],
    description=config['api']['description'],
    version=config['api']['version'],
    docs_url=config['api']['docs_url'],
    redoc_url=config['api']['redoc_url']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize HCP Scorer
try:
    # Try to use Ollama backend first, then fallback to auto
    hcp_scorer = HCPScorer(model_backend="ollama")
    logger.info("HCP Scorer initialized successfully with Ollama backend")
except Exception as e:
    logger.warning(f"Failed to initialize with Ollama backend: {e}")
    try:
        # Fallback to auto backend selection
        hcp_scorer = HCPScorer()
        logger.info("HCP Scorer initialized successfully with auto backend selection")
    except Exception as e2:
        logger.error(f"Failed to initialize HCP Scorer: {e2}")
        hcp_scorer = None

# Pydantic models for API
class TranscriptRequest(BaseModel):
    transcript: str = Field(..., description="Telehealth session transcript")
    hcp_name: Optional[str] = Field(None, description="Name of the Healthcare Provider")
    session_date: Optional[str] = Field(None, description="Date of the session (YYYY-MM-DD)")

class BatchTranscriptRequest(BaseModel):
    transcripts: List[TranscriptRequest] = Field(..., description="List of transcripts to score")

class ScoringResponse(BaseModel):
    hcp_name: Optional[str]
    session_date: Optional[str]
    scores: Dict[str, Any]
    overall_score: float
    confidence: float
    reasoning: str
    strengths: List[str]
    areas_for_improvement: List[str]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HCP Rating System API",
        "version": config['api']['version'],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if hcp_scorer is None:
        raise HTTPException(status_code=503, detail="HCP Scorer not initialized")
    
    backend_info = hcp_scorer.get_backend_info()
    
    return HealthResponse(
        status="healthy" if backend_info["available"] else "unhealthy",
        model_loaded=backend_info["available"],
        backend=backend_info["backend"]
    )

@app.post("/score", response_model=ScoringResponse)
async def score_transcript(request: TranscriptRequest):
    """Score a single telehealth transcript."""
    if hcp_scorer is None:
        raise HTTPException(status_code=503, detail="HCP Scorer not available")
    
    try:
        logger.info(f"Scoring transcript for HCP: {request.hcp_name or 'Unknown'}")
        
        # Score the transcript
        result = hcp_scorer.score_transcript(request.transcript, request.hcp_name)
        
        # Create response
        response = ScoringResponse(
            hcp_name=request.hcp_name,
            session_date=request.session_date,
            scores={
                'empathy': result.empathy,
                'clarity': result.clarity,
                'accuracy': result.accuracy,
                'professionalism': result.professionalism
            },
            overall_score=result.overall_score,
            confidence=result.confidence,
            reasoning=result.reasoning,
            strengths=result.strengths,
            areas_for_improvement=result.areas_for_improvement,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error scoring transcript: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/score/batch", response_model=List[ScoringResponse])
async def score_batch_transcripts(request: BatchTranscriptRequest):
    """Score multiple transcripts in batch."""
    if hcp_scorer is None:
        raise HTTPException(status_code=503, detail="HCP Scorer not available")
    
    try:
        logger.info(f"Scoring {len(request.transcripts)} transcripts in batch")
        
        # Prepare transcripts for batch processing
        transcripts = [(t.transcript, t.hcp_name) for t in request.transcripts]
        
        # Score transcripts
        results = hcp_scorer.batch_score_transcripts(transcripts)
        
        # Create responses
        responses = []
        for i, (request_item, result) in enumerate(zip(request.transcripts, results)):
            response = ScoringResponse(
                hcp_name=request_item.hcp_name,
                session_date=request_item.session_date,
                scores={
                    'empathy': result.empathy,
                    'clarity': result.clarity,
                    'accuracy': result.accuracy,
                    'professionalism': result.professionalism
                },
                overall_score=result.overall_score,
                confidence=result.confidence,
                reasoning=result.reasoning,
                strengths=result.strengths,
                areas_for_improvement=result.areas_for_improvement,
                timestamp=datetime.now().isoformat()
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file with transcripts for batch scoring."""
    if hcp_scorer is None:
        raise HTTPException(status_code=503, detail="HCP Scorer not available")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        df = pd.read_csv(file.file)
        
        # Validate required columns
        required_columns = ['transcript']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Prepare transcripts
        transcripts = []
        for _, row in df.iterrows():
            transcript = TranscriptRequest(
                transcript=row['transcript'],
                hcp_name=row.get('hcp_name'),
                session_date=row.get('session_date')
            )
            transcripts.append(transcript)
        
        # Score transcripts
        batch_request = BatchTranscriptRequest(transcripts=transcripts)
        results = await score_batch_transcripts(batch_request)
        
        # Generate report
        hcp_names = [t.hcp_name for t in transcripts]
        report = hcp_scorer.generate_report(
            [ScoringResult(**r.dict()) for r in results], 
            hcp_names
        )
        
        return {
            "message": f"Successfully processed {len(results)} transcripts",
            "results": results,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """Get a previously generated report."""
    # This would typically fetch from a database
    # For now, return a placeholder
    return {"message": f"Report {report_id} not found", "status": "not_implemented"}

@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "scoring_criteria": config.get('scoring', {}),
        "model_config": config.get('model', {}),
        "api_config": config.get('api', {})
    }

@app.post("/config/update")
async def update_config(new_config: Dict[str, Any]):
    """Update configuration (admin only)."""
    # This would typically require authentication
    # For now, return a placeholder
    return {"message": "Configuration update not implemented", "status": "not_implemented"}

@app.get("/metrics")
async def get_metrics():
    """Get API usage metrics."""
    # This would typically fetch from a metrics database
    return {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_response_time": 0.0
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config['server']['host'],
        port=config['server']['port'],
        reload=True
    ) 
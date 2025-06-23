#!/usr/bin/env python3
"""
Data Processing Utilities for HCP Rating System
"""

import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Utility class for processing and cleaning transcript data."""
    
    def __init__(self, max_length: int = 4000, chunk_size: int = 1000, overlap: int = 200):
        """Initialize the data processor."""
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_transcript(self, transcript: str) -> str:
        """Clean and normalize transcript text."""
        if not transcript:
            return ""
        
        # Remove extra whitespace
        transcript = re.sub(r'\s+', ' ', transcript.strip())
        
        # Remove special characters that might interfere with analysis
        transcript = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', transcript)
        
        # Normalize quotes
        transcript = transcript.replace('"', '"').replace('"', '"')
        transcript = transcript.replace(''', "'").replace(''', "'")
        
        # Normalize speaker labels
        transcript = re.sub(r'\b(?:RD|Registered Dietitian|Dietitian)\b', 'HCP', transcript, flags=re.IGNORECASE)
        transcript = re.sub(r'\b(?:Patient|Client|User)\b', 'Patient', transcript, flags=re.IGNORECASE)
        
        # Truncate if too long
        if len(transcript) > self.max_length:
            # Try to truncate at sentence boundary
            sentences = re.split(r'[.!?]+', transcript)
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) < self.max_length:
                    truncated += sentence + "."
                else:
                    break
            
            # If no sentences fit, truncate at word boundary
            words = transcript.split()
            truncated = " ".join(words[:self.max_length // 5])  # Approximate word count
            
            transcript = truncated + "..."
        
        return transcript
    
    def truncate_transcript(self, transcript: str) -> str:
        """Truncate transcript if it exceeds maximum length."""
        if len(transcript) <= self.max_length:
            return transcript
        
        # Try to truncate at sentence boundaries
        sentences = re.split(r'[.!?]', transcript)
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence) <= self.max_length - 50:  # Leave room for truncation notice
                truncated += sentence + "."
            else:
                break
        
        if not truncated:
            # If no sentences fit, truncate at word boundary
            words = transcript.split()
            truncated = " ".join(words[:self.max_length // 5])  # Approximate word count
        
        return truncated + "... [truncated]"
    
    def chunk_transcript(self, transcript: str) -> List[str]:
        """Split long transcript into overlapping chunks."""
        if len(transcript) <= self.chunk_size:
            return [transcript]
        
        chunks = []
        start = 0
        
        while start < len(transcript):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(transcript):
                # Look for sentence ending within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_text = transcript[search_start:end]
                
                # Find last sentence boundary
                sentence_end = max(
                    search_text.rfind('.'),
                    search_text.rfind('!'),
                    search_text.rfind('?')
                )
                
                if sentence_end != -1:
                    end = search_start + sentence_end + 1
            
            chunk = transcript[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(transcript):
                break
        
        return chunks
    
    def extract_speakers(self, transcript: str) -> Dict[str, List[str]]:
        """Extract speaker turns from transcript."""
        speakers = {"HCP": [], "Patient": []}
        
        # Split by speaker labels
        lines = transcript.split('\n')
        current_speaker = None
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for speaker labels
            if re.match(r'^(HCP|Patient):', line, re.IGNORECASE):
                # Save previous speaker's text
                if current_speaker and current_text:
                    speakers[current_speaker].append(current_text.strip())
                
                # Start new speaker
                speaker_match = re.match(r'^(HCP|Patient):', line, re.IGNORECASE)
                current_speaker = speaker_match.group(1).title()
                current_text = line[len(speaker_match.group(0)):].strip()
            else:
                # Continue current speaker's text
                if current_speaker:
                    current_text += " " + line
        
        # Add final speaker's text
        if current_speaker and current_text:
            speakers[current_speaker].append(current_text.strip())
        
        return speakers
    
    def validate_csv_format(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate CSV format for batch processing."""
        errors = []
        
        # Check required columns
        required_columns = ['transcript']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check data types
        if 'transcript' in df.columns:
            if not df['transcript'].dtype == 'object':
                errors.append("'transcript' column must contain text data")
            
            # Check for empty transcripts
            empty_transcripts = df['transcript'].isna().sum()
            if empty_transcripts > 0:
                errors.append(f"Found {empty_transcripts} empty transcripts")
        
        # Check optional columns
        optional_columns = ['hcp_name', 'session_date']
        for col in optional_columns:
            if col in df.columns:
                if col == 'session_date':
                    # Try to parse dates
                    try:
                        pd.to_datetime(df[col], errors='coerce')
                    except:
                        errors.append(f"'{col}' column contains invalid dates")
        
        return len(errors) == 0, errors
    
    def process_csv_batch(self, csv_path: str) -> List[Dict[str, Any]]:
        """Process CSV file for batch scoring."""
        try:
            df = pd.read_csv(csv_path)
            
            # Validate format
            is_valid, errors = self.validate_csv_format(df)
            if not is_valid:
                raise ValueError(f"CSV validation failed: {'; '.join(errors)}")
            
            # Process each row
            processed_data = []
            for idx, row in df.iterrows():
                transcript = self.clean_transcript(str(row['transcript']))
                transcript = self.truncate_transcript(transcript)
                
                processed_row = {
                    'transcript': transcript,
                    'hcp_name': row.get('hcp_name'),
                    'session_date': row.get('session_date'),
                    'original_index': idx
                }
                
                processed_data.append(processed_row)
            
            logger.info(f"Processed {len(processed_data)} transcripts from {csv_path}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_path}: {e}")
            raise
    
    def create_sample_csv(self, output_path: str = "data/sample_transcripts.csv") -> str:
        """Create a sample CSV file for testing."""
        sample_data = [
            {
                'transcript': 'HCP: Hello, how are you feeling today? Patient: I\'m really struggling with my diet. HCP: I understand this can be challenging. Let\'s work together to find solutions that work for you.',
                'hcp_name': 'Dr. Sarah Johnson',
                'session_date': '2024-01-15'
            },
            {
                'transcript': 'HCP: Good morning! I see from your records that you\'ve been working on managing your diabetes. How has that been going? Patient: It\'s been tough. I love sweets and it\'s hard to give them up.',
                'hcp_name': 'Dr. Michael Chen',
                'session_date': '2024-01-16'
            },
            {
                'transcript': 'HCP: You need to eat more vegetables. Patient: I don\'t like vegetables. HCP: You have to eat them anyway. It\'s good for you.',
                'hcp_name': 'Dr. Emily Rodriguez',
                'session_date': '2024-01-17'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Created sample CSV file: {output_path}")
        
        return output_path 
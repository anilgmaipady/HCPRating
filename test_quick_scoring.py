#!/usr/bin/env python3
"""
Quick Test Script for RD Scoring - Fast inference test
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_quick_scoring():
    """Test scoring with optimized settings for speed."""
    print("🚀 Quick Scoring Test")
    print("-" * 40)
    
    try:
        from src.inference.local_model_scorer import LocalModelScorer
        
        print("Loading model with optimized settings...")
        
        # Create a shorter test transcript
        test_transcript = """RD: Hello, how are you feeling today? 
Patient: I'm struggling with my diet. 
RD: I understand. Let's work together to find solutions. What specific difficulties are you facing? 
Patient: I don't know where to start. 
RD: That's normal. Let's start with what you're currently eating."""
        
        print("Starting inference...")
        start_time = time.time()
        
        scorer = LocalModelScorer()
        
        print("Scoring transcript...")
        result = scorer.score_transcript(test_transcript, "Dr. Test")
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"✅ Scoring completed in {inference_time:.2f} seconds!")
        print(f"📊 Overall Score: {result.overall_score}/5.0")
        print(f"❤️  Empathy: {result.empathy}/5")
        print(f"💬 Clarity: {result.clarity}/5")
        print(f"✅ Accuracy: {result.accuracy}/5")
        print(f"👔 Professionalism: {result.professionalism}/5")
        
        print(f"\n💭 Reasoning: {result.reasoning[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_smaller_model():
    """Test with a smaller model for faster inference."""
    print("\n🔧 Testing with smaller model...")
    print("-" * 40)
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a smaller model for testing
        model_name = "microsoft/DialoGPT-medium"  # Much smaller than Mistral-7B
        
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully!")
        print("This smaller model can be used for testing while Mistral-7B loads.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading smaller model: {e}")
        return False

if __name__ == "__main__":
    print("Testing RD Scoring System - Quick Version")
    print("=" * 50)
    
    # Test with smaller model first
    test_with_smaller_model()
    
    # Test with full model (will be slower)
    print("\n" + "=" * 50)
    print("Testing with full Mistral-7B model (this may take several minutes)...")
    test_quick_scoring() 
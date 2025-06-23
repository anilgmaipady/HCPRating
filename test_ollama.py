#!/usr/bin/env python3
"""
Test Ollama Integration for HCP Rating System
"""

import sys
import requests
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.inference.hcp_scorer import HCPScorer

def test_ollama_integration():
    """Test Ollama integration with sample transcript."""
    try:
        # Initialize HCP Scorer with Ollama backend
        scorer = HCPScorer(model_backend="ollama")
        
        # Test transcript
        test_transcript = """
        HCP: Hello, how are you feeling today?
        Patient: I'm really struggling with my diet.
        HCP: I understand this can be challenging. Let's work together to find solutions that work for you.
        """
        
        # Score transcript
        result = scorer.score_transcript(test_transcript, "Test HCP")
        
        # Validate results
        assert result.empathy >= 1 and result.empathy <= 5
        assert result.clarity >= 1 and result.clarity <= 5
        assert result.accuracy >= 1 and result.accuracy <= 5
        assert result.professionalism >= 1 and result.professionalism <= 5
        assert result.overall_score >= 1.0 and result.overall_score <= 5.0
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        
        print("✅ Ollama integration test passed!")
        print(f"   Backend: {scorer.model_backend}")
        print(f"   Overall Score: {result.overall_score:.2f}/5.0")
        print(f"   Empathy: {result.empathy}/5")
        print(f"   Clarity: {result.clarity}/5")
        print(f"   Accuracy: {result.accuracy}/5")
        print(f"   Professionalism: {result.professionalism}/5")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Reasoning: {result.reasoning[:100]}...")
        
        if result.strengths:
            print(f"   Strengths: {', '.join(result.strengths)}")
        
        if result.areas_for_improvement:
            print(f"   Areas for Improvement: {', '.join(result.areas_for_improvement)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama integration test failed: {e}")
        return False

def check_ollama_availability():
    """Check if Ollama is available and running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is available and running")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama is not available")
        return False

def check_model_availability(model_name="mistral"):
    """Check if the specified model is available in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if model_name in model_names:
                print(f"✅ Model '{model_name}' is available")
                return True
            else:
                print(f"❌ Model '{model_name}' is not available")
                print(f"   Available models: {', '.join(model_names)}")
                return False
    except Exception as e:
        print(f"❌ Error checking model availability: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Ollama Integration")
    print("=" * 40)
    
    # Check Ollama availability
    print("1. Checking Ollama availability...")
    if not check_ollama_availability():
        print("\n💡 To fix this:")
        print("   - Install Ollama: https://ollama.ai")
        print("   - Start Ollama: ollama serve")
        return False
    
    # Check model availability
    print("\n2. Checking model availability...")
    if not check_model_availability("mistral"):
        print("\n💡 To fix this:")
        print("   - Download model: ollama pull mistral")
        return False
    
    # Test integration
    print("\n3. Testing HCP scoring integration...")
    if not test_ollama_integration():
        print("\n💡 To fix this:")
        print("   - Check Ollama logs: ollama logs")
        print("   - Restart Ollama: ollama serve")
        return False
    
    print("\n🎉 All tests passed! Ollama integration is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
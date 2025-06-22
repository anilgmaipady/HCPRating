#!/usr/bin/env python3
"""
RD Rating System - Demo Script
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)

def print_banner():
    """Print the system banner."""
    print("=" * 60)
    print("🥗 RD Rating System - Demo")
    print("=" * 60)
    print("A comprehensive system for rating Registered Dietitians")
    print("based on telehealth session transcripts using Mistral-7B")
    print("=" * 60)

def demo_transcripts():
    """Sample transcripts for demonstration."""
    return [
        {
            "name": "Dr. Sarah Johnson",
            "transcript": """RD: Hello, how are you feeling today? 
Patient: I'm really struggling with my diet. I've been trying to eat healthy but nothing seems to work. 
RD: I understand this can be challenging. Let's work together to find solutions that work for you. What specific difficulties are you facing? 
Patient: I don't know where to start. 
RD: That's completely normal. Let's start with what you're currently eating and how you're feeling about it. Can you tell me about your typical day?""",
            "expected_score": "High empathy and collaborative approach"
        },
        {
            "name": "Dr. Michael Chen", 
            "transcript": """RD: Good morning! I see from your records that you've been working on managing your diabetes. How has that been going? 
Patient: It's been tough. I love sweets and it's hard to give them up. 
RD: I hear you. Sweet foods can be really enjoyable. Instead of thinking about giving them up completely, let's talk about how we can enjoy them in a way that works with your health goals. What are your favorite sweet foods? 
Patient: I love chocolate and ice cream. 
RD: Great choices! Let me show you some ways to enjoy these while keeping your blood sugar in check.""",
            "expected_score": "Excellent empathy and practical guidance"
        },
        {
            "name": "Dr. Emily Rodriguez",
            "transcript": """RD: You need to eat more vegetables. 
Patient: I don't like vegetables. 
RD: You have to eat them anyway. It's good for you. 
Patient: But I really don't like the taste. 
RD: Well, you'll just have to get used to it. There's no other way to be healthy.""",
            "expected_score": "Low empathy, directive approach"
        }
    ]

def demo_local_model_scoring():
    """Demonstrate local model scoring functionality."""
    print("\n🤖 Local Model Scoring Demo")
    print("-" * 40)
    
    try:
        from src.inference.local_model_scorer import LocalModelScorer
        
        print("Loading local model...")
        scorer = LocalModelScorer()
        
        # Test with sample transcript
        sample_transcript = """RD: Hello, how are you feeling today? 
Patient: I'm really struggling with my diet. I've been trying to eat healthy but nothing seems to work. 
RD: I understand this can be challenging. Let's work together to find solutions that work for you. What specific difficulties are you facing? 
Patient: I don't know where to start. 
RD: That's completely normal. Let's start with what you're currently eating and how you're feeling about it. Can you tell me about your typical day?"""
        
        print("Scoring sample transcript...")
        result = scorer.score_transcript(sample_transcript, "Dr. Sarah Johnson")
        
        print(f"✅ Scoring completed!")
        print(f"📊 Overall Score: {result.overall_score}/5.0")
        print(f"❤️  Empathy: {result.empathy}/5")
        print(f"💬 Clarity: {result.clarity}/5")
        print(f"✅ Accuracy: {result.accuracy}/5")
        print(f"👔 Professionalism: {result.professionalism}/5")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        
        print(f"\n💭 Reasoning: {result.reasoning[:200]}...")
        
        if result.strengths:
            print(f"\n✨ Strengths:")
            for strength in result.strengths[:3]:
                print(f"  • {strength}")
        
        if result.areas_for_improvement:
            print(f"\n🔧 Areas for Improvement:")
            for area in result.areas_for_improvement[:3]:
                print(f"  • {area}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in local model scoring: {e}")
        print("💡 Make sure the model is downloaded and dependencies are installed")
        return False

def demo_cli_usage():
    """Demonstrate CLI usage."""
    print("\n💻 Command Line Interface Demo")
    print("-" * 40)
    
    print("1. Single transcript scoring:")
    print("   python src/cli.py score \"RD: Hello, how are you feeling today?\"")
    
    print("\n2. Batch processing from CSV:")
    print("   python src/cli.py csv data/sample_transcripts.csv")
    
    print("\n3. Interactive mode:")
    print("   python src/cli.py interactive")

def demo_api_usage():
    """Demonstrate API usage."""
    print("\n🔧 API Usage Demo")
    print("-" * 40)
    
    print("1. Single transcript scoring:")
    print("""
import requests

url = "http://localhost:8001/score"
data = {
    "transcript": "RD: Hello, how are you feeling today?",
    "rd_name": "Dr. Smith"
}

response = requests.post(url, json=data)
result = response.json()
print(f"Overall Score: {result['overall_score']}")
""")
    
    print("2. Batch processing:")
    print("""
url = "http://localhost:8001/score/batch"
data = {"transcripts": [{"transcript": "...", "rd_name": "Dr. Smith"}]}
response = requests.post(url, json=data)
""")

def demo_web_interface():
    """Demonstrate web interface features."""
    print("\n📊 Web Interface Demo")
    print("-" * 40)
    
    print("1. Single Transcript Scoring:")
    print("   - Navigate to 'Single Transcript'")
    print("   - Enter RD name and session date")
    print("   - Paste transcript and click 'Score Transcript'")
    print("   - View detailed results with charts")
    
    print("\n2. Batch Processing:")
    print("   - Navigate to 'Batch Processing'")
    print("   - Add multiple transcripts")
    print("   - Process batch and view summary")
    
    print("\n3. CSV Upload:")
    print("   - Navigate to 'Upload CSV'")
    print("   - Upload CSV file with transcripts")
    print("   - Download results in various formats")

def demo_scoring_criteria():
    """Demonstrate scoring criteria."""
    print("\n🎯 Scoring Criteria Demo")
    print("-" * 40)
    
    criteria = {
        "Empathy (25%)": [
            "Shows genuine concern for patient's feelings",
            "Uses empathetic language and tone", 
            "Acknowledges patient's emotional state"
        ],
        "Clarity (25%)": [
            "Uses simple, jargon-free language",
            "Explains concepts clearly",
            "Provides structured information"
        ],
        "Accuracy (25%)": [
            "Provides evidence-based recommendations",
            "Avoids misinformation",
            "References current guidelines"
        ],
        "Professionalism (25%)": [
            "Maintains appropriate boundaries",
            "Shows respect for patient autonomy",
            "Follows ethical guidelines"
        ]
    }
    
    for dimension, items in criteria.items():
        print(f"\n{dimension}:")
        for item in items:
            print(f"  • {item}")

def demo_sample_data():
    """Show sample data structure."""
    print("\n📁 Sample Data Structure")
    print("-" * 40)
    
    print("CSV Format:")
    print("transcript,rd_name,session_date")
    print('"RD: Hello, how are you feeling today? Patient: I\'m struggling...",Dr. Smith,2024-01-15')
    
    print("\nJSON Output Format:")
    print("""
{
  "rd_name": "Dr. Smith",
  "scores": {
    "empathy": 4,
    "clarity": 3,
    "accuracy": 5,
    "professionalism": 4
  },
  "overall_score": 4.0,
  "confidence": 0.85,
  "reasoning": "The RD shows good empathy...",
  "strengths": ["Shows empathy", "Professional tone"],
  "areas_for_improvement": ["Could be clearer"]
}
""")

def demo_quick_start():
    """Show quick start instructions."""
    print("\n🚀 Quick Start Instructions")
    print("-" * 40)
    
    steps = [
        "1. Install dependencies: pip install -r requirements.txt",
        "2. Download model: python download_model.py",
        "3. Test local scoring: python demo.py",
        "4. Start all services: python start.py all",
        "5. Access web interface: http://localhost:8501",
        "6. Access API docs: http://localhost:8001/docs"
    ]
    
    for step in steps:
        print(step)

def main():
    """Main demo function."""
    print_banner()
    
    # Show quick start
    demo_quick_start()
    
    # Show scoring criteria
    demo_scoring_criteria()
    
    # Test local model scoring
    demo_local_model_scoring()
    
    # Show sample data
    demo_sample_data()
    
    # Show CLI usage
    demo_cli_usage()
    
    # Show API usage
    demo_api_usage()
    
    # Show web interface
    demo_web_interface()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! The RD Rating System is ready to use.")
    print("=" * 60)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
HCP Rating System Demo Script

This script provides an automated demonstration of the HCP Rating System
with sample transcripts and various use cases.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_ollama_health():
    """Check if Ollama is running and healthy."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            mistral_available = any('mistral' in model.get('name', '') for model in models)
            if mistral_available:
                print("✅ Ollama is running and Mistral model is available")
                return True
            else:
                print("❌ Ollama is running but Mistral model not found")
                return False
        else:
            print("❌ Ollama is not responding properly")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def check_api_health():
    """Check if the API server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print("❌ API server is not responding properly")
            return False
    except Exception as e:
        print("ℹ️ API server is not running (will use direct Ollama)")
        return False

def demo_single_scoring():
    """Demonstrate single transcript scoring."""
    print("\n" + "="*60)
    print("DEMO: Single Transcript Scoring")
    print("="*60)
    
    # Sample transcripts for demonstration
    sample_transcripts = [
        {
            "name": "High-Performing HCP",
            "transcript": """Patient: "I'm really worried about my test results."
HCP: "I can see this is causing you a lot of anxiety. Let me explain what these results mean and what our next steps will be. We're going to work through this together."
Patient: "Thank you, that makes me feel better."
HCP: "You're welcome. Remember, I'm here to support you throughout this process. Do you have any questions about what we discussed?" """,
            "hcp_name": "Dr. Johnson"
        },
        {
            "name": "Average-Performing HCP", 
            "transcript": """Patient: "My back has been hurting for a week."
HCP: "Let me examine you. Can you describe the pain?"
Patient: "It's sharp and gets worse when I move."
HCP: "I'll order some tests and prescribe pain medication. Come back in a week." """,
            "hcp_name": "Dr. Smith"
        },
        {
            "name": "Low-Performing HCP",
            "transcript": """Patient: "I'm feeling very depressed lately."
HCP: "Take these pills. Next patient."
Patient: "But I have questions about side effects."
HCP: "Read the label. Next!" """,
            "hcp_name": "Dr. Brown"
        }
    ]
    
    api_available = check_api_health()
    
    for i, sample in enumerate(sample_transcripts, 1):
        print(f"\n{i}. Testing: {sample['name']}")
        print("-" * 40)
        print(f"Transcript: {sample['transcript'][:100]}...")
        
        try:
            if api_available:
                # Use API
                payload = {
                    "transcript": sample["transcript"],
                    "hcp_name": sample["hcp_name"],
                    "session_date": "2024-01-15"
                }
                response = requests.post("http://localhost:8000/score", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                else:
                    print(f"❌ API error: {response.status_code}")
                    continue
            else:
                # Use direct Ollama
                from src.inference.hcp_scorer import HCPScorer
                scorer = HCPScorer()
                result_obj = scorer.score_transcript(sample["transcript"], sample["hcp_name"])
                result = {
                    "hcp_name": sample["hcp_name"],
                    "scores": {
                        'empathy': result_obj.empathy,
                        'clarity': result_obj.clarity,
                        'accuracy': result_obj.accuracy,
                        'professionalism': result_obj.professionalism
                    },
                    "overall_score": result_obj.overall_score,
                    "confidence": result_obj.confidence,
                    "reasoning": result_obj.reasoning,
                    "strengths": result_obj.strengths,
                    "areas_for_improvement": result_obj.areas_for_improvement
                }
            
            # Display results
            print(f"✅ Overall Score: {result['overall_score']:.2f}/5.0")
            print(f"📊 Individual Scores:")
            for dimension, score in result['scores'].items():
                print(f"   {dimension.title()}: {score:.2f}/5.0")
            print(f"🎯 Confidence: {result['confidence']:.1%}")
            print(f"💡 Key Strength: {result['strengths'][0] if result['strengths'] else 'None'}")
            print(f"🔧 Improvement Area: {result['areas_for_improvement'][0] if result['areas_for_improvement'] else 'None'}")
            
        except Exception as e:
            print(f"❌ Error scoring transcript: {e}")
        
        time.sleep(1)  # Brief pause between samples

def demo_batch_scoring():
    """Demonstrate batch transcript scoring."""
    print("\n" + "="*60)
    print("DEMO: Batch Transcript Scoring")
    print("="*60)
    
    # Sample batch data
    batch_transcripts = [
        {
            "transcript": "Patient: I have a headache. HCP: How long have you had it? Can you describe the pain?",
            "hcp_name": "Dr. Wilson"
        },
        {
            "transcript": "Patient: I'm concerned about my medication. HCP: I understand your concern. Let's review your current treatment and discuss any side effects you're experiencing.",
            "hcp_name": "Dr. Davis"
        },
        {
            "transcript": "Patient: My symptoms are getting worse. HCP: I need to examine you more thoroughly. Let's run some additional tests to understand what's happening.",
            "hcp_name": "Dr. Miller"
        }
    ]
    
    api_available = check_api_health()
    
    try:
        if api_available:
            # Use API
            payload = {"transcripts": batch_transcripts}
            response = requests.post("http://localhost:8000/score/batch", json=payload, timeout=60)
            if response.status_code == 200:
                results = response.json()
            else:
                print(f"❌ API error: {response.status_code}")
                return
        else:
            # Use direct Ollama
            from src.inference.hcp_scorer import HCPScorer
            scorer = HCPScorer()
            
            # Prepare transcripts for batch processing
            transcripts = [(t["transcript"], t.get("hcp_name")) for t in batch_transcripts]
            result_objects = scorer.batch_score_transcripts(transcripts)
            
            # Convert to API response format
            results = []
            for i, (request_item, result_obj) in enumerate(zip(batch_transcripts, result_objects)):
                response = {
                    "hcp_name": request_item.get("hcp_name"),
                    "session_date": "2024-01-15",
                    "scores": {
                        'empathy': result_obj.empathy,
                        'clarity': result_obj.clarity,
                        'accuracy': result_obj.accuracy,
                        'professionalism': result_obj.professionalism
                    },
                    "overall_score": result_obj.overall_score,
                    "confidence": result_obj.confidence,
                    "reasoning": result_obj.reasoning,
                    "strengths": result_obj.strengths,
                    "areas_for_improvement": result_obj.areas_for_improvement
                }
                results.append(response)
        
        # Display batch results
        print(f"✅ Successfully processed {len(results)} transcripts")
        print("\n📊 Batch Results Summary:")
        print("-" * 50)
        
        total_score = 0
        dimension_scores = {'empathy': 0, 'clarity': 0, 'accuracy': 0, 'professionalism': 0}
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['hcp_name'] or 'Unknown HCP'}")
            print(f"   Overall Score: {result['overall_score']:.2f}/5.0")
            total_score += result['overall_score']
            
            for dimension, score in result['scores'].items():
                dimension_scores[dimension] += score
        
        # Calculate averages
        avg_overall = total_score / len(results)
        print(f"\n📈 Batch Averages:")
        print(f"   Overall Score: {avg_overall:.2f}/5.0")
        for dimension, total in dimension_scores.items():
            avg = total / len(results)
            print(f"   {dimension.title()}: {avg:.2f}/5.0")
        
        # Find top performer
        top_performer = max(results, key=lambda x: x['overall_score'])
        print(f"\n🏆 Top Performer: {top_performer['hcp_name']} (Score: {top_performer['overall_score']:.2f})")
        
    except Exception as e:
        print(f"❌ Error in batch scoring: {e}")

def demo_export_functionality():
    """Demonstrate export functionality."""
    print("\n" + "="*60)
    print("DEMO: Export Functionality")
    print("="*60)
    
    # Sample result for export
    sample_result = {
        "hcp_name": "Dr. Demo",
        "session_date": "2024-01-15",
        "scores": {
            'empathy': 4.2,
            'clarity': 3.8,
            'accuracy': 4.5,
            'professionalism': 4.0
        },
        "overall_score": 4.1,
        "confidence": 0.85,
        "reasoning": "The healthcare provider demonstrated excellent communication skills and showed genuine concern for the patient's well-being.",
        "strengths": ["Strong empathetic communication", "Clear explanation of medical terms"],
        "areas_for_improvement": ["Could provide more specific follow-up instructions"],
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    # Export as JSON
    json_export = json.dumps(sample_result, indent=2)
    print("📄 JSON Export:")
    print(json_export[:300] + "..." if len(json_export) > 300 else json_export)
    
    # Export as CSV (simplified)
    import csv
    import io
    
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    
    # Write header
    csv_writer.writerow(['HCP Name', 'Session Date', 'Overall Score', 'Empathy', 'Clarity', 'Accuracy', 'Professionalism', 'Confidence'])
    
    # Write data
    csv_writer.writerow([
        sample_result['hcp_name'],
        sample_result['session_date'],
        sample_result['overall_score'],
        sample_result['scores']['empathy'],
        sample_result['scores']['clarity'],
        sample_result['scores']['accuracy'],
        sample_result['scores']['professionalism'],
        sample_result['confidence']
    ])
    
    print(f"\n📊 CSV Export:")
    print(csv_buffer.getvalue())

def main():
    """Main demo function."""
    print("🚀 HCP Rating System - Demo Script")
    print("="*60)
    
    # Check system health
    print("\n🔍 System Health Check:")
    ollama_healthy = check_ollama_health()
    
    if not ollama_healthy:
        print("\n❌ Cannot proceed without Ollama. Please:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Pull Mistral model: ollama pull mistral")
        return
    
    # Run demos
    try:
        demo_single_scoring()
        demo_batch_scoring()
        demo_export_functionality()
        
        print("\n" + "="*60)
        print("🎉 Demo Completed Successfully!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Start the frontend: streamlit run frontend/app.py")
        print("2. Start the API server: python src/deployment/start_server.py")
        print("3. Access the web interface at: http://localhost:8501")
        print("4. Try the CLI: python src/cli.py --help")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")

if __name__ == "__main__":
    main() 
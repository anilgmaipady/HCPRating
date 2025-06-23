#!/usr/bin/env python3
"""
Command Line Interface for HCP Rating System
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.hcp_scorer import HCPScorer

def score_single_transcript(transcript: str, hcp_name: Optional[str] = None,
                          config_path: Optional[str] = None, backend: str = "auto"):
    """Score a single transcript."""
    try:
        scorer = HCPScorer(config_path, model_backend=backend)
        
        print(f"🔍 Analyzing transcript...")
        print(f"📋 Backend: {scorer.model_backend}")
        print(f"👤 HCP: {hcp_name or 'Unknown'}")
        
        result = scorer.score_transcript(transcript, hcp_name)
        
        print(f"\n🎯 HCP Rating Results for: {hcp_name or 'Unknown HCP'}")
        print("=" * 50)
        print(f"📊 Overall Score: {result.overall_score:.2f}/5.0")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print()
        print("📈 Dimension Scores:")
        print(f"  ❤️  Empathy: {result.empathy}/5")
        print(f"  💬 Clarity: {result.clarity}/5")
        print(f"  ✅ Accuracy: {result.accuracy}/5")
        print(f"  👔 Professionalism: {result.professionalism}/5")
        print()
        print("🧠 Reasoning:")
        print(f"  {result.reasoning}")
        print()
        if result.strengths:
            print("✅ Strengths:")
            for strength in result.strengths:
                print(f"  • {strength}")
            print()
        if result.areas_for_improvement:
            print("🔧 Areas for Improvement:")
            for area in result.areas_for_improvement:
                print(f"  • {area}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error scoring transcript: {e}")
        return None

def score_from_file(file_path: str, hcp_name: Optional[str] = None,
                   config_path: Optional[str] = None, backend: str = "auto"):
    """Score transcript from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        
        if not transcript:
            print("❌ File is empty")
            return None
        
        return score_single_transcript(transcript, hcp_name, config_path, backend)
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def score_csv_batch(csv_path: str, config_path: Optional[str] = None, backend: str = "auto"):
    """Score multiple transcripts from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        if 'transcript' not in df.columns:
            print("❌ CSV must contain a 'transcript' column")
            return None
        
        print(f"📊 Processing {len(df)} transcripts from {csv_path}")
        print(f"🔧 Backend: {backend}")
        print()
        
        scorer = HCPScorer(config_path, model_backend=backend)
        results = []
        
        for idx, row in df.iterrows():
            transcript = row['transcript']
            hcp_name = row.get('hcp_name', f"HCP_{idx+1}")
            
            print(f"\nProcessing {hcp_name}...")
            result = scorer.score_transcript(transcript, hcp_name)
            results.append(result)
        
        # Display summary
        print("\n" + "=" * 60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 60)
        
        avg_overall = sum(r.overall_score for r in results) / len(results)
        print(f"📈 Average Overall Score: {avg_overall:.2f}/5.0")
        print()
        
        # Sort by overall score
        sorted_results = sorted(enumerate(results), key=lambda x: x[1].overall_score, reverse=True)
        
        print("🏆 Top Performers:")
        for i, (idx, result) in enumerate(sorted_results[:5]):
            hcp_name = df.iloc[idx].get('hcp_name', f"HCP_{idx+1}")
            print(f"  {i+1}. {hcp_name}: {result.overall_score:.2f}/5.0")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return None

def test_backend(backend: str = "auto", config_path: Optional[str] = None):
    """Test a specific backend."""
    try:
        print(f"🧪 Testing {backend} backend...")
        
        test_backend = backend if backend != "auto" else "ollama"
        scorer = HCPScorer(config_path, model_backend=test_backend)
        
        # Test with a simple transcript
        test_transcript = "HCP: Hello, how are you feeling today? Patient: I'm struggling with my health. HCP: I understand this can be challenging. Let's work together to find solutions that work for you."
        
        print("📝 Testing with sample transcript...")
        result = scorer.score_transcript(test_transcript, "Test HCP")
        
        print("✅ Backend test successful!")
        print(f"📊 Test score: {result.overall_score:.2f}/5.0")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def interactive_mode(config_path: Optional[str] = None, backend: str = "auto"):
    """Start interactive mode for transcript scoring."""
    print("🥗 HCP Rating System - Interactive Mode")
    print("=" * 50)
    print("Enter transcripts to score. Type 'quit' to exit, 'help' for commands.")
    print()
    
    try:
        scorer = HCPScorer(config_path, model_backend=backend)
        print(f"🔧 Using {scorer.model_backend} backend")
        print()
        
        while True:
            try:
                command = input("🎯 Enter command (score/file/csv/test/help/quit): ").strip().lower()
                
                if command == 'quit':
                    print("👋 Goodbye!")
                    break
                elif command == 'help':
                    print("Available commands:")
                    print("  score - Score a single transcript")
                    print("  file  - Score transcript from file")
                    print("  csv   - Process CSV batch file")
                    print("  test  - Test backend")
                    print("  help  - Show this help")
                    print("  quit  - Exit")
                elif command == 'score':
                    hcp_name = input("👤 HCP Name (optional): ").strip() or None
                    print("📝 Enter transcript (press Enter twice to finish):")
                    
                    lines = []
                    while True:
                        line = input()
                        if line == "" and lines:
                            break
                        lines.append(line)
                    
                    transcript = "\n".join(lines)
                    if transcript.strip():
                        score_single_transcript(transcript, hcp_name, config_path, backend)
                    else:
                        print("❌ No transcript entered")
                elif command == 'file':
                    file_path = input("📁 File path: ").strip()
                    hcp_name = input("👤 HCP Name (optional): ").strip() or None
                    score_from_file(file_path, hcp_name, config_path, backend)
                elif command == 'csv':
                    csv_path = input("📊 CSV file path: ").strip()
                    score_csv_batch(csv_path, config_path, backend)
                elif command == 'test':
                    test_backend(backend, config_path)
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
                
    except Exception as e:
        print(f"❌ Error initializing interactive mode: {e}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="HCP Rating System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score a single transcript
  python src/cli.py score "HCP: Hello, how are you feeling today? Patient: I'm struggling..."

  # Score with specific backend
  python src/cli.py score "HCP: Hello..." --backend ollama

  # Score from file
  python src/cli.py file transcript.txt --hcp-name "Dr. Smith"

  # Process CSV batch
  python src/cli.py csv data/transcripts.csv

  # Test backend
  python src/cli.py test --backend ollama

  # Interactive mode
  python src/cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score a single transcript')
    score_parser.add_argument('transcript', help='Transcript text to score')
    score_parser.add_argument('--hcp-name', help='Name of the Healthcare Provider')
    score_parser.add_argument('--config', help='Path to configuration file')
    score_parser.add_argument('--backend', default='auto', 
                             choices=['auto', 'ollama', 'vllm', 'local', 'openai'],
                             help='Model backend to use')
    
    # File command
    file_parser = subparsers.add_parser('file', help='Score transcript from file')
    file_parser.add_argument('file_path', help='Path to transcript file')
    file_parser.add_argument('--hcp-name', help='Name of the Healthcare Provider')
    file_parser.add_argument('--config', help='Path to configuration file')
    file_parser.add_argument('--backend', default='auto',
                             choices=['auto', 'ollama', 'vllm', 'local', 'openai'],
                             help='Model backend to use')
    
    # CSV command
    csv_parser = subparsers.add_parser('csv', help='Process CSV batch file')
    csv_parser.add_argument('csv_path', help='Path to CSV file')
    csv_parser.add_argument('--config', help='Path to configuration file')
    csv_parser.add_argument('--backend', default='auto',
                           choices=['auto', 'ollama', 'vllm', 'local', 'openai'],
                           help='Model backend to use')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test backend')
    test_parser.add_argument('--backend', default='auto',
                            choices=['auto', 'ollama', 'vllm', 'local', 'openai'],
                            help='Backend to test')
    test_parser.add_argument('--config', help='Path to configuration file')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.add_argument('--config', help='Path to configuration file')
    interactive_parser.add_argument('--backend', default='auto',
                                   choices=['auto', 'ollama', 'vllm', 'local', 'openai'],
                                   help='Model backend to use')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'score':
            score_single_transcript(args.transcript, args.hcp_name, args.config, args.backend)
        elif args.command == 'file':
            score_from_file(args.file_path, args.hcp_name, args.config, args.backend)
        elif args.command == 'csv':
            score_csv_batch(args.csv_path, args.config, args.backend)
        elif args.command == 'test':
            test_backend(args.backend, args.config)
        elif args.command == 'interactive':
            interactive_mode(args.config, args.backend)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
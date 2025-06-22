#!/usr/bin/env python3
"""
Command Line Interface for RD Rating System
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.rd_scorer import RDScorer

def score_single_transcript(transcript: str, rd_name: Optional[str] = None, 
                          config_path: Optional[str] = None):
    """Score a single transcript."""
    try:
        scorer = RDScorer(config_path)
        result = scorer.score_transcript(transcript, rd_name)
        
        print(f"\n🎯 RD Rating Results for: {rd_name or 'Unknown RD'}")
        print("=" * 50)
        print(f"Overall Score: {result.overall_score}/5.0")
        print(f"Confidence: {result.confidence:.1%}")
        print("\n📊 Individual Scores:")
        print(f"  Empathy: {result.empathy}/5")
        print(f"  Clarity: {result.clarity}/5")
        print(f"  Accuracy: {result.accuracy}/5")
        print(f"  Professionalism: {result.professionalism}/5")
        
        print(f"\n📝 Reasoning:")
        print(f"  {result.reasoning}")
        
        if result.strengths:
            print(f"\n✅ Strengths:")
            for strength in result.strengths:
                print(f"  • {strength}")
        
        if result.areas_for_improvement:
            print(f"\n🔧 Areas for Improvement:")
            for area in result.areas_for_improvement:
                print(f"  • {area}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error scoring transcript: {e}")
        return None

def score_from_file(file_path: str, rd_name: Optional[str] = None, 
                   config_path: Optional[str] = None):
    """Score transcript from file."""
    try:
        with open(file_path, 'r') as f:
            transcript = f.read().strip()
        
        return score_single_transcript(transcript, rd_name, config_path)
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def score_batch_from_csv(csv_path: str, config_path: Optional[str] = None):
    """Score multiple transcripts from CSV file."""
    try:
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        if 'transcript' not in df.columns:
            print("❌ CSV must contain 'transcript' column")
            return None
        
        scorer = RDScorer(config_path)
        
        print(f"📊 Processing {len(df)} transcripts...")
        
        results = []
        for idx, row in df.iterrows():
            transcript = row['transcript']
            rd_name = row.get('rd_name', f"RD_{idx+1}")
            
            print(f"\nProcessing {rd_name}...")
            result = scorer.score_transcript(transcript, rd_name)
            results.append(result)
        
        # Generate summary
        print(f"\n📋 Batch Processing Complete!")
        print("=" * 50)
        
        avg_overall = sum(r.overall_score for r in results) / len(results)
        print(f"Average Overall Score: {avg_overall:.2f}/5.0")
        
        # Show top performers
        sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
        print(f"\n🏆 Top Performers:")
        for i, result in enumerate(sorted_results[:3]):
            rd_name = df.iloc[i].get('rd_name', f"RD_{i+1}")
            print(f"  {i+1}. {rd_name}: {result.overall_score:.2f}/5.0")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return None

def interactive_mode(config_path: Optional[str] = None):
    """Interactive mode for scoring transcripts."""
    print("🥗 RD Rating System - Interactive Mode")
    print("=" * 40)
    print("Enter 'quit' to exit")
    print("Enter 'help' for commands")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                print("👋 Goodbye!")
                break
            elif command == 'help':
                print("Commands:")
                print("  score <transcript> - Score a transcript")
                print("  file <path> - Score transcript from file")
                print("  csv <path> - Score batch from CSV")
                print("  quit - Exit")
                print("  help - Show this help")
            elif command.startswith('score '):
                transcript = command[6:]
                if transcript:
                    score_single_transcript(transcript, config_path=config_path)
                else:
                    print("❌ Please provide a transcript")
            elif command.startswith('file '):
                file_path = command[5:]
                if file_path:
                    score_from_file(file_path, config_path=config_path)
                else:
                    print("❌ Please provide a file path")
            elif command.startswith('csv '):
                csv_path = command[4:]
                if csv_path:
                    score_batch_from_csv(csv_path, config_path=config_path)
                else:
                    print("❌ Please provide a CSV file path")
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="RD Rating System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score a single transcript
  python src/cli.py score "RD: Hello, how are you feeling today? Patient: I'm struggling..."

  # Score from file
  python src/cli.py file transcript.txt --rd-name "Dr. Smith"

  # Score batch from CSV
  python src/cli.py csv data/transcripts.csv

  # Interactive mode
  python src/cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Score single transcript
    score_parser = subparsers.add_parser('score', help='Score a single transcript')
    score_parser.add_argument('transcript', help='Transcript text to score')
    score_parser.add_argument('--rd-name', help='Name of the Registered Dietitian')
    score_parser.add_argument('--config', help='Path to custom config file')
    
    # Score from file
    file_parser = subparsers.add_parser('file', help='Score transcript from file')
    file_parser.add_argument('file_path', help='Path to file containing transcript')
    file_parser.add_argument('--rd-name', help='Name of the Registered Dietitian')
    file_parser.add_argument('--config', help='Path to custom config file')
    
    # Score batch from CSV
    csv_parser = subparsers.add_parser('csv', help='Score batch from CSV file')
    csv_parser.add_argument('csv_path', help='Path to CSV file with transcripts')
    csv_parser.add_argument('--config', help='Path to custom config file')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'score':
            score_single_transcript(args.transcript, args.rd_name, args.config)
        elif args.command == 'file':
            score_from_file(args.file_path, args.rd_name, args.config)
        elif args.command == 'csv':
            score_batch_from_csv(args.csv_path, args.config)
        elif args.command == 'interactive':
            interactive_mode(args.config)
        else:
            print(f"❌ Unknown command: {args.command}")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
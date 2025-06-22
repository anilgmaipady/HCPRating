#!/usr/bin/env python3
"""
RD Rating System - Main Startup Script
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers
        import vllm
        import streamlit
        import fastapi
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def start_vllm_server():
    """Start the vLLM server."""
    print("🚀 Starting vLLM server...")
    try:
        subprocess.run([
            sys.executable, "src/deployment/start_server.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start vLLM server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  vLLM server stopped")
        return True

def start_api_server():
    """Start the FastAPI server."""
    print("🌐 Starting API server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "src.api.main:app", 
            "--host", "0.0.0.0", "--port", "8001", "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  API server stopped")
        return True

def start_streamlit_app():
    """Start the Streamlit app."""
    print("📊 Starting Streamlit app...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "frontend/app.py",
            "--server.port", "8501", "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Streamlit app stopped")
        return True

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    try:
        subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], check=True)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing."""
    print("📝 Creating sample data...")
    try:
        subprocess.run([
            sys.executable, "src/training/fine_tune.py", "--create_sample"
        ], check=True)
        print("✅ Sample data created!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="RD Rating System - Main Control Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start vLLM server
  python start.py vllm

  # Start API server
  python start.py api

  # Start Streamlit app
  python start.py streamlit

  # Start all services
  python start.py all

  # Run tests
  python start.py test

  # Create sample data
  python start.py sample
        """
    )
    
    parser.add_argument(
        'command',
        choices=['vllm', 'api', 'streamlit', 'all', 'test', 'sample', 'check'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_dependencies()
    elif args.command == 'vllm':
        start_vllm_server()
    elif args.command == 'api':
        start_api_server()
    elif args.command == 'streamlit':
        start_streamlit_app()
    elif args.command == 'all':
        print("🎯 Starting all RD Rating System services...")
        print("=" * 50)
        
        # Check dependencies first
        if not check_dependencies():
            return
        
        # Start vLLM server in background
        print("\n1️⃣  Starting vLLM server...")
        vllm_process = subprocess.Popen([
            sys.executable, "src/deployment/start_server.py"
        ])
        
        # Wait a moment for vLLM to start
        import time
        time.sleep(10)
        
        # Start API server in background
        print("\n2️⃣  Starting API server...")
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "src.api.main:app", 
            "--host", "0.0.0.0", "--port", "8001"
        ])
        
        # Wait a moment for API to start
        time.sleep(5)
        
        # Start Streamlit app
        print("\n3️⃣  Starting Streamlit app...")
        print("🌐 Streamlit will be available at: http://localhost:8501")
        print("📚 API documentation at: http://localhost:8001/docs")
        print("⏹️  Press Ctrl+C to stop all services")
        
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", "frontend/app.py",
                "--server.port", "8501", "--server.address", "0.0.0.0"
            ])
        except KeyboardInterrupt:
            print("\n⏹️  Stopping all services...")
            vllm_process.terminate()
            api_process.terminate()
            print("✅ All services stopped")
            
    elif args.command == 'test':
        run_tests()
    elif args.command == 'sample':
        create_sample_data()
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main() 
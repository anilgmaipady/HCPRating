#!/usr/bin/env python3
"""
HCP Rating System - Main Startup Script

This script provides multiple ways to start the HCP Rating System:
- Quick start with Ollama (recommended)
- vLLM deployment
- Individual service startup
- Ollama setup and testing
"""

import os
import sys
import subprocess
import time
import requests
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def check_ollama_availability():
    """Check if Ollama is available and running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_model_availability(model_name="mistral"):
    """Check if the specified model is available in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model["name"] == model_name for model in models)
    except:
        pass
    return False

def pull_model(model_name="mistral"):
    """Pull the specified model from Ollama."""
    print(f"📥 Pulling {model_name} model...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        if result.returncode == 0:
            print(f"✅ Successfully pulled {model_name} model")
            return True
        else:
            print(f"❌ Failed to pull {model_name} model: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout while pulling {model_name} model")
        return False
    except Exception as e:
        print(f"❌ Error pulling {model_name} model: {e}")
        return False

def test_ollama_integration():
    """Test Ollama integration with a simple request."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": "Hello, this is a test.",
                "stream": False
            },
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ollama integration test failed: {e}")
        return False

def start_frontend():
    """Start the Streamlit frontend."""
    print("🌐 Starting Streamlit frontend...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped")
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")

def show_help():
    """Show help information"""
    print("HCP Rating System - Startup Script")
    print("=" * 40)
    print()
    print("Available commands:")
    print()
    print("  quick          - Quick start with Ollama (recommended)")
    print("  vllm           - Start vLLM server")
    print("  api            - Start FastAPI server only")
    print("  streamlit      - Start Streamlit frontend only")
    print("  ollama-setup   - Set up Ollama with recommended model")
    print("  ollama-test    - Test Ollama integration")
    print("  help           - Show this help message")
    print()
    print("Examples:")
    print("  python start.py quick")
    print("  python start.py ollama-setup")
    print("  python start.py vllm")

def main():
    """Main startup function."""
    print("🚀 HCP Rating System - Startup")
    print("=" * 50)
    
    # Check if Ollama is available
    print("🔍 Checking Ollama availability...")
    if not check_ollama_availability():
        print("❌ Ollama is not running or not available")
        print("💡 Please start Ollama first:")
        print("   ollama serve")
        return False
    
    print("✅ Ollama is available")
    
    # Check if mistral model is available
    print("🔍 Checking model availability...")
    if not check_model_availability("mistral"):
        print("📥 Mistral model not found, pulling...")
        if not pull_model("mistral"):
            print("❌ Failed to pull mistral model")
            print("💡 You can try pulling it manually:")
            print("   ollama pull mistral")
            return False
    else:
        print("✅ Mistral model is available")
    
    # Test integration
    print("🧪 Testing Ollama integration...")
    if not test_ollama_integration():
        print("❌ Ollama integration test failed")
        return False
    
    print("✅ Ollama integration test passed")
    
    # Start the frontend
    print("🚀 Starting HCP Rating System...")
    start_frontend()
    
    return True

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
HCP Rating System - Simple One-Command Startup

This script provides the simplest way to start the HCP Rating System using Ollama.
Just run: python run.py
"""

import os
import sys
import subprocess
import time
import requests
import platform
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, "Ollama is not installed"
        
        # Check if server is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama server is not running"
        
        return True, "Ollama is ready"
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Ollama is not installed"
    except requests.RequestException:
        return False, "Ollama server is not running"

def install_ollama_guide():
    """Show installation guide for Ollama"""
    print("🔧 Ollama Installation Required")
    print("=" * 40)
    
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        print("📱 For macOS:")
        print("1. Visit https://ollama.ai")
        print("2. Download the macOS installer (.dmg file)")
        print("3. Run the installer and follow the wizard")
        print("4. Or use Homebrew: brew install ollama")
        print()
        print("After installation:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull a model: ollama pull mistral")
        print("3. Run this script again: python run.py")
        
    elif system == "linux":
        print("🐧 For Linux:")
        print("Run: curl -fsSL https://ollama.ai/install.sh | sh")
        print("Then: ollama serve")
        print("Then: ollama pull mistral")
        
    elif system == "windows":
        print("🪟 For Windows:")
        print("1. Visit https://ollama.ai")
        print("2. Download the Windows installer")
        print("3. Run the installer and follow the wizard")
        
    else:
        print("❓ Unknown OS. Please visit https://ollama.ai")
    
    print("\nAfter installing Ollama, run this script again.")

def setup_ollama():
    """Set up Ollama with recommended model"""
    print("🚀 Setting up Ollama...")
    
    # Start Ollama server if not running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Starting Ollama server...")
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            print("⏳ Waiting for server to start...")
            time.sleep(5)
    except:
        print("❌ Could not start Ollama server")
        return False
    
    # Check if mistral model is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        
        if 'mistral' not in model_names:
            print("📥 Downloading Mistral model (this may take a few minutes)...")
            subprocess.run(['ollama', 'pull', 'mistral'], check=True)
            print("✅ Mistral model downloaded!")
        else:
            print("✅ Mistral model is already available")
            
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Ollama: {e}")
        return False

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

def main():
    print("🚀 HCP Rating System - Quick Start")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected.")
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created.")
        print("Please activate it and run this script again:")
        print("source venv/bin/activate  # On macOS/Linux")
        print("venv\\Scripts\\activate     # On Windows")
        return
    
    # Check Ollama
    ollama_ready, message = check_ollama()
    if not ollama_ready:
        print(f"❌ {message}")
        install_ollama_guide()
        return
    
    # Install dependencies if needed
    try:
        import streamlit
        import requests
        import yaml
    except ImportError:
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    
    # Setup Ollama
    if not setup_ollama():
        print("❌ Failed to setup Ollama")
        return
    
    # Check Ollama availability
    print("🔍 Checking Ollama availability...")
    if not check_ollama_availability():
        print("❌ Ollama is not running or not available")
        print("💡 Please start Ollama first:")
        print("   ollama serve")
        return False
    
    print("✅ Ollama is available")
    
    # Check model availability
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
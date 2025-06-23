#!/usr/bin/env python3
"""
Setup Checker for HCP Rating System
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import requests
import yaml

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_required_files():
    """Check if required files exist."""
    print("\n📁 Checking required files...")
    
    required_files = [
        'requirements.txt', 'README.md', 'configs/config.yaml',
        'src/api/main.py', 'src/inference/hcp_scorer.py', 'src/training/fine_tune.py',
        'frontend/app.py', 'tests/test_hcp_scorer.py', 'data/sample_transcripts.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'fastapi', 'uvicorn', 'pandas', 'requests',
        'pyyaml', 'plotly', 'pydantic', 'torch', 'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_ollama():
    """Check if Ollama is available."""
    print("\n🤖 Checking Ollama...")
    
    # Check if ollama command exists
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
        else:
            print("❌ Ollama command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found")
        return False
    
    # Check if Ollama server is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print("❌ Ollama server is not responding")
            return False
    except requests.RequestException:
        print("❌ Ollama server is not running")
        return False

def check_ollama_models():
    """Check if required Ollama models are available."""
    print("\n📚 Checking Ollama models...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            # Check for recommended models
            recommended_models = ["mistral", "llama2", "codellama"]
            available_recommended = []
            
            for model in recommended_models:
                if model in model_names:
                    print(f"✅ {model}")
                    available_recommended.append(model)
                else:
                    print(f"❌ {model} - Not available")
            
            if available_recommended:
                print(f"✅ {len(available_recommended)} recommended models available")
                return True
            else:
                print("❌ No recommended models available")
                return False
        else:
            print("❌ Could not check models")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def check_configuration():
    """Check configuration files."""
    print("\n⚙️ Checking configuration...")
    
    # Check main config
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['model', 'ollama', 'scoring', 'api']
            missing_sections = []
            
            for section in required_sections:
                if section in config:
                    print(f"✅ {section} configuration")
                else:
                    print(f"❌ {section} configuration - Missing")
                    missing_sections.append(section)
            
            return len(missing_sections) == 0
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return False
    else:
        print("❌ configs/config.yaml - Missing")
        return False

def test_hcp_scorer():
    """Test HCP Scorer functionality."""
    print("\n🧪 Testing HCP Scorer...")
    
    try:
        from src.inference.hcp_scorer import HCPScorer
        
        # Test initialization
        scorer = HCPScorer()
        print("✅ HCP Scorer initialized")
        
        # Test backend info
        info = scorer.get_backend_info()
        print(f"✅ Backend: {info['backend']}")
        
        # Test with sample transcript
        test_transcript = """
        HCP: Hello, how are you feeling today?
        Patient: I'm really struggling with my diet.
        HCP: I understand this can be challenging. Let's work together to find solutions that work for you.
        """
        
        result = scorer.score_transcript(test_transcript, "Test HCP")
        print(f"✅ Test scoring completed - Overall Score: {result.overall_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ HCP Scorer test failed: {e}")
        return False

def test_api():
    """Test API functionality."""
    print("\n🌐 Testing API...")
    
    try:
        # Test API health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API is running - Status: {health_data.get('status')}")
            print(f"✅ Backend: {health_data.get('backend')}")
            return True
        else:
            print("❌ API is not responding properly")
            return False
    except requests.RequestException:
        print("❌ API is not running")
        return False

def test_frontend():
    """Test frontend functionality."""
    print("\n🎨 Testing frontend...")
    
    try:
        # Test if Streamlit can import the app
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Check if frontend app can be imported
        sys.path.append(str(Path(__file__).parent))
        from frontend.app import main
        print("✅ Frontend app can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def generate_setup_report():
    """Generate a comprehensive setup report."""
    print("🚀 HCP Rating System - Setup Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Files", check_required_files),
        ("Dependencies", check_dependencies),
        ("Ollama Installation", check_ollama),
        ("Ollama Models", check_ollama_models),
        ("Configuration", check_configuration),
        ("HCP Scorer", test_hcp_scorer),
        ("API", test_api),
        ("Frontend", test_frontend)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your HCP Rating System is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Start the system: python run.py")
        print("   2. Open browser: http://localhost:8501")
        print("   3. Or use CLI: python src/cli.py score 'Your transcript'")
    else:
        print(f"\n⚠️ {total - passed} checks failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Install Ollama: https://ollama.ai")
        print("   - Start Ollama: ollama serve")
        print("   - Download models: ollama pull mistral")
    
    return passed == total

def main():
    """Main function."""
    success = generate_setup_report()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 
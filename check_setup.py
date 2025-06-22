#!/usr/bin/env python3
"""
RD Rating System - Setup Verification Script
"""

import os
import sys
import importlib
from pathlib import Path
import subprocess
import json

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.9+")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'vllm', 'streamlit', 'fastapi', 
        'pandas', 'numpy', 'openai', 'pydantic', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check if all required directories and files exist."""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        'src', 'src/api', 'src/inference', 'src/training', 'src/deployment', 'src/utils',
        'frontend', 'configs', 'tests', 'data', 'models', 'logs', 'exports', 'docs'
    ]
    
    required_files = [
        'requirements.txt', 'README.md', 'configs/config.yaml', 'start.py', 'demo.py',
        'src/api/main.py', 'src/inference/rd_scorer.py', 'src/training/fine_tune.py',
        'frontend/app.py', 'tests/test_rd_scorer.py', 'data/sample_transcripts.csv'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ - OK")
        else:
            print(f"❌ {dir_path}/ - Missing")
            missing_dirs.append(dir_path)
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - OK")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print(f"\n⚠️  Missing directories: {missing_dirs}")
        print(f"⚠️  Missing files: {missing_files}")
        return False
    
    return True

def check_configuration():
    """Check if configuration files are valid."""
    print("\n⚙️  Checking configuration...")
    
    try:
        import yaml
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['model', 'server', 'scoring', 'training', 'api']
        missing_sections = []
        
        for section in required_sections:
            if section in config:
                print(f"✅ {section} configuration - OK")
            else:
                print(f"❌ {section} configuration - Missing")
                missing_sections.append(section)
        
        if missing_sections:
            print(f"\n⚠️  Missing configuration sections: {missing_sections}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability."""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU available - will use CPU (slower)")
            return True
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False

def check_ports():
    """Check if required ports are available."""
    print("\n🔌 Checking port availability...")
    
    ports = [8000, 8001, 8501]
    occupied_ports = []
    
    for port in ports:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"⚠️  Port {port} - Occupied")
                occupied_ports.append(port)
            else:
                print(f"✅ Port {port} - Available")
        except Exception as e:
            print(f"❌ Port {port} - Error checking: {e}")
            occupied_ports.append(port)
    
    if occupied_ports:
        print(f"\n⚠️  Occupied ports: {occupied_ports}")
        print("You may need to stop other services using these ports")
        return False
    
    return True

def check_sample_data():
    """Check if sample data is properly formatted."""
    print("\n📊 Checking sample data...")
    
    try:
        import pandas as pd
        df = pd.read_csv('data/sample_transcripts.csv')
        
        required_columns = ['transcript']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns in sample data: {missing_columns}")
            return False
        
        print(f"✅ Sample data - OK ({len(df)} transcripts)")
        return True
        
    except Exception as e:
        print(f"❌ Sample data error: {e}")
        return False

def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test configuration loading
        from src.inference.rd_scorer import RDScorer
        print("✅ RDScorer import - OK")
        
        # Test data processing
        from src.utils.data_processor import DataProcessor
        processor = DataProcessor()
        print("✅ DataProcessor - OK")
        
        # Test export utilities
        from src.utils.export_utils import ExportUtils
        export_utils = ExportUtils()
        print("✅ ExportUtils - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 RD Rating System - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Configuration", check_configuration),
        ("GPU Availability", check_gpu_availability),
        ("Port Availability", check_ports),
        ("Sample Data", check_sample_data),
        ("Basic Tests", run_basic_tests)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your RD Rating System is ready to use.")
        print("\nNext steps:")
        print("1. Start the system: python start.py all")
        print("2. Access web interface: http://localhost:8501")
        print("3. View API docs: http://localhost:8001/docs")
        print("4. Run demo: python demo.py")
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check configuration: configs/config.yaml")
        print("3. Free up ports: Stop other services using ports 8000, 8001, 8501")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
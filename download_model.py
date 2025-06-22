#!/usr/bin/env python3
"""
Script to download the Mistral-7B-Instruct-v0.1 model for the RD Rating System.
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    """Download the Mistral-7B-Instruct-v0.1 model."""
    
    # Model configuration
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    models_dir = Path("models")
    model_dir = models_dir / "mistral-7b-instruct"
    
    print(f"🚀 Downloading {model_name}...")
    print(f"📁 Model will be saved to: {model_dir}")
    
    # Check if user is authenticated
    print("\n⚠️  IMPORTANT: You need to be authenticated to download this model.")
    print("1. Go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1")
    print("2. Accept the model terms of use")
    print("3. Run: huggingface-cli login")
    print("4. Then run this script again\n")
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=models_dir,
            trust_remote_code=True
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(model_dir)
        print(f"✅ Tokenizer saved to {model_dir}")
        
        # Download model (with 4-bit quantization to save memory)
        print("📥 Downloading model (4-bit quantized)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=models_dir,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Save model
        model.save_pretrained(model_dir)
        print(f"✅ Model saved to {model_dir}")
        
        print(f"\n🎉 Successfully downloaded {model_name}!")
        print(f"📊 Model size: {model_dir.stat().st_size / (1024**3):.2f} GB")
        
        # Update config to use local model
        update_config(model_dir)
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("\n💡 If you see an authentication error:")
        print("1. Run: huggingface-cli login")
        print("2. Accept the model terms at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1")
        print("3. Try running this script again")
        sys.exit(1)

def update_config(model_dir):
    """Update the config to use the local model path."""
    config_file = Path("configs/config.yaml")
    
    if config_file.exists():
        print("📝 Updating config to use local model...")
        
        # Read current config
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Replace model name with local path
        new_content = content.replace(
            'name: "mistralai/Mistral-7B-Instruct-v0.1"',
            f'name: "{model_dir}"'
        )
        
        # Write updated config
        with open(config_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Config updated to use local model")

if __name__ == "__main__":
    download_model() 
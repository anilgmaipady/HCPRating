#!/usr/bin/env python3
"""
Ollama Client for RD Rating System
"""

import json
import logging
import requests
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ModelClient(ABC):
    """Abstract base class for model clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from model."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model client is available."""
        pass

class OllamaClient(ModelClient):
    """Client for interacting with Ollama models."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "mistral"):
        """Initialize Ollama client."""
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = requests.Session()
        self.session.timeout = 60  # 60 second timeout
        
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server not available: {e}")
            return False
    
    def list_models(self) -> list:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from Ollama model."""
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.1),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 40),
                    "num_predict": kwargs.get('max_tokens', 2048),
                    "stop": kwargs.get('stop', [])
                }
            }
            
            # Make request to Ollama
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # Longer timeout for generation
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            return data.get('response', '')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_stream(self, prompt: str, **kwargs):
        """Generate streaming response from Ollama model."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get('temperature', 0.1),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 40),
                    "num_predict": kwargs.get('max_tokens', 2048),
                    "stop": kwargs.get('stop', [])
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
    def pull_model(self, model_name: str = None) -> bool:
        """Pull a model from Ollama library."""
        model_to_pull = model_name or self.model_name
        try:
            logger.info(f"Pulling model: {model_to_pull}")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_to_pull},
                timeout=300  # 5 minute timeout for model pulling
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {model_to_pull}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_to_pull}: {e}")
            return False
    
    def get_model_info(self, model_name: str = None) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        model_to_check = model_name or self.model_name
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_to_check}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info for {model_to_check}: {e}")
            return None

class OllamaManager:
    """Manager for Ollama operations."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama manager."""
        self.base_url = base_url
        self.clients = {}
    
    def get_client(self, model_name: str = "mistral") -> OllamaClient:
        """Get or create an Ollama client for a specific model."""
        if model_name not in self.clients:
            self.clients[model_name] = OllamaClient(self.base_url, model_name)
        return self.clients[model_name]
    
    def ensure_model_available(self, model_name: str = "mistral") -> bool:
        """Ensure a model is available, pulling it if necessary."""
        client = self.get_client(model_name)
        
        # Check if model is already available
        if client.is_available():
            models = client.list_models()
            if model_name in models:
                logger.info(f"Model {model_name} is already available")
                return True
        
        # Try to pull the model
        logger.info(f"Model {model_name} not found, attempting to pull...")
        return client.pull_model(model_name)
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        try:
            client = self.get_client()
            if client.is_available():
                return client.list_models()
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            client = self.get_client()
            return client.is_available()
        except Exception as e:
            logger.error(f"Error testing Ollama connection: {e}")
            return False

# Convenience functions
def create_ollama_client(model_name: str = "mistral", base_url: str = "http://localhost:11434") -> OllamaClient:
    """Create an Ollama client with default settings."""
    return OllamaClient(base_url, model_name)

def check_ollama_availability(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is available."""
    manager = OllamaManager(base_url)
    return manager.test_connection()

def get_recommended_models() -> list:
    """Get list of recommended models for RD scoring."""
    return [
        "mistral",
        "llama2",
        "codellama",
        "neural-chat",
        "vicuna",
        "wizard-vicuna-uncensored"
    ] 
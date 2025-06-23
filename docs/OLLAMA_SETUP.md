# Ollama Setup Guide for RD Rating System

## 🚀 Quick Start with Ollama

Ollama provides a simple way to run large language models locally without needing to start a vLLM server. This guide will help you set up Ollama for use with the RD Rating System.

## 📋 Prerequisites

1. **Install Ollama**: Download and install Ollama from [ollama.ai](https://ollama.ai)
2. **Python Environment**: Ensure you have Python 3.9+ installed
3. **RD Rating System**: Clone and set up the RD Rating System project

## 🔧 Installation Steps

### 1. Install Ollama

**macOS:**
```bash
# Option A: Download from website (recommended)
# Visit https://ollama.ai and download the macOS installer
# Run the downloaded .dmg file and follow the installation wizard

# Option B: Use Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from [ollama.ai](https://ollama.ai) and run it.

### 2. Start Ollama Service

```bash
ollama serve
```

This starts the Ollama server on `http://localhost:11434`.

### 3. Pull a Model

Pull a recommended model for RD scoring:

```bash
# Pull Mistral (recommended)
ollama pull mistral

# Or try other models
ollama pull llama2
ollama pull codellama
ollama pull neural-chat
```

### 4. Install RD Rating System Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Testing the Setup

### 1. Test Ollama Integration

Run the Ollama test script:

```bash
python test_ollama.py
```

This will:
- Check if Ollama server is running
- Verify model availability
- Test RD scoring functionality
- Show backend information

### 2. Test with CLI

```bash
# Test single transcript scoring
python src/cli.py score "RD: Hello, how are you feeling today? Patient: I'm struggling..." --backend ollama

# Show backend information
python src/cli.py backend

# Interactive mode
python src/cli.py interactive --backend ollama
```

### 3. Test with Web Interface

```bash
# Start the web interface
streamlit run frontend/app.py
```

The system will automatically detect and use Ollama if available.

## ⚙️ Configuration

### Ollama Settings

Edit `configs/config.yaml` to customize Ollama settings:

```yaml
# Ollama Configuration
ollama:
  base_url: "http://localhost:11434"  # Ollama server URL
  model_name: "mistral"               # Model to use
  temperature: 0.1                    # Generation temperature
  top_p: 0.9                          # Top-p sampling
  top_k: 40                           # Top-k sampling
  max_tokens: 2048                    # Maximum tokens to generate
  timeout: 120                        # Request timeout in seconds
```

### Model Selection

You can use different models by changing the `model_name` in the config:

```yaml
ollama:
  model_name: "llama2"  # Use Llama 2 instead of Mistral
```

Available models include:
- `mistral` - Mistral 7B (recommended)
- `llama2` - Llama 2 7B
- `codellama` - Code Llama
- `neural-chat` - Neural Chat
- `vicuna` - Vicuna
- `wizard-vicuna-uncensored` - Wizard Vicuna

## 🔄 Using Different Backends

The RD Rating System supports multiple backends. You can specify which one to use:

```bash
# Use Ollama (recommended for local use)
python src/cli.py score "transcript" --backend ollama

# Use vLLM (if you have a vLLM server running)
python src/cli.py score "transcript" --backend vllm

# Use local HuggingFace model
python src/cli.py score "transcript" --backend local

# Use OpenAI-compatible API
python src/cli.py score "transcript" --backend openai

# Auto-select best available backend
python src/cli.py score "transcript" --backend auto
```

## 📊 Performance Comparison

| Backend | Setup Complexity | Speed | Memory Usage | Model Variety |
|---------|------------------|-------|--------------|---------------|
| **Ollama** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| vLLM | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Local | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| OpenAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🛠️ Troubleshooting

### Common Issues

**1. Ollama server not running**
```bash
# Start Ollama service
ollama serve
```

**2. Model not found**
```bash
# List available models
ollama list

# Pull the model
ollama pull mistral
```

**3. Connection timeout**
- Check if Ollama is running on the correct port (11434)
- Verify firewall settings
- Try increasing timeout in config

**4. Memory issues**
- Use smaller models (e.g., `mistral:7b` instead of `llama2:70b`)
- Close other applications to free up memory
- Consider using model quantization

**5. macOS-specific issues**
- Ensure Ollama has necessary permissions
- Check if macOS security settings are blocking Ollama
- Try restarting Ollama after installation

### Debug Commands

```bash
# Check Ollama status
ollama list

# Check model info
ollama show mistral

# Test model generation
ollama run mistral "Hello, how are you?"

# Check system resources
ollama ps
```

## 🚀 Advanced Usage

### Custom Models

You can use custom models with Ollama:

```bash
# Pull a specific model variant
ollama pull mistral:7b-instruct

# Use a custom model file
ollama create mymodel -f Modelfile
ollama run mymodel
```

### Model Management

```bash
# List all models
ollama list

# Remove unused models
ollama rm modelname

# Copy models
ollama cp source target
```

### Performance Optimization

1. **Use quantized models** for better memory efficiency:
   ```bash
   ollama pull mistral:7b-instruct-q4_0
   ```

2. **Adjust generation parameters** in config:
   ```yaml
   ollama:
     temperature: 0.1    # Lower for more consistent results
     max_tokens: 1024    # Reduce for faster generation
   ```

3. **Monitor system resources**:
   ```bash
   # Check memory usage
   ollama ps
   
   # Monitor GPU usage (if available)
   nvidia-smi
   ```

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Model Library](https://ollama.ai/library)
- [Community Models](https://ollama.ai/library?sort=popular)
- [Performance Tips](https://ollama.ai/docs/performance)

## 🎉 Next Steps

Once Ollama is set up and working:

1. **Test the system** with sample transcripts
2. **Customize scoring criteria** in the config file
3. **Try different models** to find the best performance
4. **Set up batch processing** for multiple transcripts
5. **Integrate with your workflow** using the API or CLI

The RD Rating System with Ollama provides a powerful, local solution for evaluating Registered Dietitians without requiring external API calls or complex server setups. 
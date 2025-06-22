# RD Rating System - Usage Guide

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd RD-RANK

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the System

```bash
# Start all services (recommended)
python start.py all

# Or start services individually:
python start.py vllm    # Start vLLM server
python start.py api     # Start API server  
python start.py streamlit  # Start web interface
```

### 3. Access the System

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8001/docs
- **API Health Check**: http://localhost:8001/health

## 📊 Using the Web Interface

### Single Transcript Scoring

1. Navigate to "Single Transcript" in the sidebar
2. Enter the RD name (optional)
3. Select the session date (optional)
4. Paste the telehealth session transcript
5. Click "Score Transcript"
6. View detailed results with charts and analysis

### Batch Processing

1. Navigate to "Batch Processing" in the sidebar
2. Add multiple transcripts using the form
3. Click "Process Batch" to score all transcripts
4. View summary results and export data

### CSV Upload

1. Navigate to "Upload CSV" in the sidebar
2. Upload a CSV file with the following columns:
   - `transcript` (required): The session transcript
   - `rd_name` (optional): Name of the RD
   - `session_date` (optional): Date of the session
3. Click "Process CSV" to score all transcripts
4. Download results in various formats

## 🔧 Using the API

### Single Transcript Scoring

```python
import requests

url = "http://localhost:8001/score"
data = {
    "transcript": "RD: Hello, how are you feeling today? Patient: I'm struggling...",
    "rd_name": "Dr. Smith",
    "session_date": "2024-01-15"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Overall Score: {result['overall_score']}")
print(f"Empathy: {result['scores']['empathy']}")
print(f"Reasoning: {result['reasoning']}")
```

### Batch Scoring

```python
import requests

url = "http://localhost:8001/score/batch"
data = {
    "transcripts": [
        {
            "transcript": "Transcript 1...",
            "rd_name": "Dr. Smith"
        },
        {
            "transcript": "Transcript 2...",
            "rd_name": "Dr. Johnson"
        }
    ]
}

response = requests.post(url, json=data)
results = response.json()

for result in results:
    print(f"{result['rd_name']}: {result['overall_score']}")
```

### CSV Upload

```python
import requests

url = "http://localhost:8001/upload/csv"
files = {"file": open("transcripts.csv", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Processed {len(result['results'])} transcripts")
```

## 💻 Command Line Interface

### Single Transcript

```bash
# Score a transcript directly
python src/cli.py score "RD: Hello, how are you feeling today? Patient: I'm struggling..."

# Score with RD name
python src/cli.py score "Transcript text..." --rd-name "Dr. Smith"

# Score from file
python src/cli.py file transcript.txt --rd-name "Dr. Smith"
```

### Batch Processing

```bash
# Process CSV file
python src/cli.py csv data/transcripts.csv
```

### Interactive Mode

```bash
# Start interactive CLI
python src/cli.py interactive

# Available commands:
# score <transcript> - Score a transcript
# file <path> - Score from file
# csv <path> - Process CSV batch
# quit - Exit
# help - Show help
```

## 🎯 Scoring Criteria

The system evaluates RDs across four dimensions:

### Empathy (25% weight)
- Shows genuine concern for patient's feelings
- Uses empathetic language and tone
- Acknowledges patient's emotional state

### Clarity (25% weight)
- Uses simple, jargon-free language
- Explains concepts clearly
- Provides structured information

### Accuracy (25% weight)
- Provides evidence-based recommendations
- Avoids misinformation
- References current guidelines

### Professionalism (25% weight)
- Maintains appropriate boundaries
- Shows respect for patient autonomy
- Follows ethical guidelines

### Scoring Scale
- **1: Poor** - Significant issues, needs immediate improvement
- **2: Below Average** - Several areas need improvement
- **3: Average** - Meets basic standards, some room for improvement
- **4: Good** - Above average performance, minor areas for improvement
- **5: Excellent** - Outstanding performance, exemplary standards

## 🔄 Fine-tuning the Model

### Prepare Training Data

Create a JSONL file with the following format:

```json
{
  "transcript": "RD: Hello, how are you feeling today? Patient: I'm struggling...",
  "empathy": 4,
  "clarity": 3,
  "accuracy": 5,
  "professionalism": 4,
  "reasoning": "Good empathy shown through validation...",
  "strengths": ["Shows empathy", "Professional tone"],
  "areas_for_improvement": ["Could ask more specific questions"]
}
```

### Run Fine-tuning

```bash
# Create sample training data
python src/training/fine_tune.py --create_sample

# Fine-tune with custom data
python src/training/fine_tune.py --data_path data/training_data.jsonl --output_dir models/custom_model
```

### Use Fine-tuned Model

Update the configuration in `configs/config.yaml`:

```yaml
model:
  name: "models/custom_model"  # Path to your fine-tuned model
```

## 📁 File Formats

### Input CSV Format

```csv
transcript,rd_name,session_date
"RD: Hello, how are you feeling today? Patient: I'm struggling...",Dr. Smith,2024-01-15
"RD: Good morning! I see from your records...",Dr. Johnson,2024-01-16
```

### Output JSON Format

```json
{
  "rd_name": "Dr. Smith",
  "session_date": "2024-01-15",
  "scores": {
    "empathy": 4,
    "clarity": 3,
    "accuracy": 5,
    "professionalism": 4
  },
  "overall_score": 4.0,
  "confidence": 0.85,
  "reasoning": "The RD shows good empathy...",
  "strengths": ["Shows empathy", "Professional tone"],
  "areas_for_improvement": ["Could be clearer"],
  "timestamp": "2024-01-15T10:30:00"
}
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

### Model Settings
```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.1"
  temperature: 0.1
  max_length: 4096
```

### Scoring Weights
```yaml
scoring:
  dimensions:
    empathy:
      weight: 0.25
    clarity:
      weight: 0.25
    accuracy:
      weight: 0.25
    professionalism:
      weight: 0.25
```

### Server Settings
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  tensor_parallel_size: 1
```

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
python start.py test

# Or run directly
python -m pytest tests/ -v
```

### Test API Endpoints

```bash
# Test health endpoint
curl http://localhost:8001/health

# Test scoring endpoint
curl -X POST http://localhost:8001/score \
  -H "Content-Type: application/json" \
  -d '{"transcript": "RD: Hello, how are you feeling today?"}'
```

## 📈 Performance Optimization

### For Large Batches

1. **Increase batch size** in configuration
2. **Use GPU acceleration** if available
3. **Process in chunks** for very large datasets

### Memory Management

1. **Reduce max_length** for shorter transcripts
2. **Use 4-bit quantization** (enabled by default)
3. **Adjust tensor_parallel_size** based on GPU count

## 🔍 Troubleshooting

### Common Issues

1. **vLLM server not starting**
   - Check GPU availability
   - Verify model download
   - Check port availability

2. **API connection errors**
   - Ensure vLLM server is running
   - Check API server status
   - Verify port configurations

3. **Memory errors**
   - Reduce batch size
   - Enable 4-bit quantization
   - Use smaller model variant

### Logs

Check logs in the `logs/` directory:
- `rd_rating.log` - Application logs
- `training.log` - Training logs

### Health Checks

```bash
# Check API health
curl http://localhost:8001/health

# Check vLLM status
curl http://localhost:8000/v1/models
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Run the test suite
4. Check the logs for error messages

## 🔄 Updates

To update the system:

```bash
# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
python start.py all
``` 
# HCP Rating System - Usage Guide

## Introduction

Welcome to the HCP Rating System! This comprehensive guide will help you get up and running with the HCP Rating System quickly and efficiently.

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Ollama (for local model inference)
- 8GB+ RAM (16GB+ recommended)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd RD-RANK
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   
   **macOS:**
   ```bash
   # Option 1: Homebrew
   brew install ollama
   
   # Option 2: Manual download
   # Visit https://ollama.ai and download the macOS installer
   # Run the downloaded .dmg file and follow the installation wizard
   ```
   
   **Linux:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
   
   **Windows:**
   ```bash
   # Visit https://ollama.ai and download the Windows installer
   # Run the installer and follow the wizard
   ```

5. **Start Ollama and Download Model**
   ```bash
   # Start Ollama service
   ollama serve
   
   # In a new terminal, download the model
   ollama pull mistral
   ```

6. **Start the System**
   ```bash
   python run.py
   ```

The web interface will open automatically at `http://localhost:8501`

## Using the Web Interface

### Single Transcript Scoring

1. **Access the Interface**
   - Open your browser and go to `http://localhost:8501`
   - You'll see the main HCP Rating System interface

2. **Use the "🎲 Generate Random Transcript" button**
   - Click this button to instantly fill the transcript area with a realistic example for quick testing or demo.
   - You can also paste or type your own transcript.

3. **Enter HCP Information**:
   - HCP Name (optional)
   - Session Date (optional)

4. **Score Transcript**:
   - Click "Score Transcript"
   - Wait for the analysis to complete
   - View detailed results with scores, reasoning, and recommendations

### Batch Processing

1. **Navigate to Batch Processing**:
   - Click on "Batch Processing" in the sidebar

2. **Add Transcripts**:
   - Enter multiple transcripts one by one
   - Add HCP names and session dates for each
   - Or use the CSV upload feature

3. **Process Batch**:
   - Click "Process Batch" to score all transcripts
   - View summary statistics and individual results
   - Export results in JSON or CSV format

### CSV Upload

1. **Prepare CSV File**:
   - Create a CSV file with the following columns:
   ```
   transcript,hcp_name,session_date
   "HCP: Hello, how are you feeling today? Patient: I'm struggling...","Dr. Smith","2024-01-15"
   "HCP: Let's discuss your nutrition goals...","Dr. Johnson","2024-01-16"
   ```

2. **Upload and Process**:
   - Go to "CSV Upload" in the sidebar
   - Upload your CSV file
   - Click "Process CSV" to analyze all transcripts
   - Download results in your preferred format

## Using the Command Line Interface

### Basic Commands

1. **Score a Single Transcript**:
   ```bash
   python src/cli.py score "HCP: Hello, how are you feeling today? Patient: I'm struggling..."
   ```

2. **Score with HCP Name**:
   ```bash
   python src/cli.py score "HCP: Hello..." --hcp-name "Dr. Smith" --session-date "2024-01-15"
   ```

3. **Use Specific Backend**:
   ```bash
   python src/cli.py score "HCP: Hello..." --backend ollama
   ```

4. **Score from File**:
   ```bash
   python src/cli.py file transcript.txt --hcp-name "Dr. Smith"
   ```

5. **Process CSV Batch**:
   ```bash
   python src/cli.py csv data/transcripts.csv
   ```

6. **Test Backend**:
   ```bash
   python src/cli.py test --backend ollama
   ```

7. **Interactive Mode**:
   ```bash
   python src/cli.py interactive
   ```

### Interactive Mode Commands

When in interactive mode, you can use these commands:

- `score` - Score a single transcript
- `file` - Score transcript from file
- `csv` - Process CSV batch file
- `test` - Test backend
- `help` - Show available commands
- `quit` - Exit interactive mode

## Using the API

### Starting the API Server

1. **Start API Server**:
   ```bash
   python start_api.py
   ```

2. **Access API Documentation**:
   - Open `http://localhost:8000/docs` for interactive API docs
   - Open `http://localhost:8000/redoc` for ReDoc documentation

### API Endpoints

#### 1. Score Single Transcript

**Endpoint**: `POST /score`

**Request**:
```json
{
  "transcript": "HCP: Hello, how are you feeling today? Patient: I'm struggling...",
  "hcp_name": "Dr. Smith"
}
```

**Response**:
```json
{
  "hcp_name": "Dr. Smith",
  "empathy": 4,
  "clarity": 5,
  "accuracy": 4,
  "professionalism": 4,
  "overall_score": 4.25,
  "confidence": 0.85,
  "reasoning": "The HCP demonstrates good empathy and clear communication...",
  "strengths": ["Clear communication", "Professional demeanor"],
  "areas_for_improvement": ["Could ask more follow-up questions"]
}
```

#### 2. Batch Processing

**Endpoint**: `POST /batch`

**Request**:
```json
{
  "transcripts": [
    {"transcript": "Transcript 1...", "hcp_name": "Dr. Smith"},
    {"transcript": "Transcript 2...", "hcp_name": "Dr. Johnson"}
  ]
}
```

#### 3. Upload CSV

**Endpoint**: `POST /upload-csv`

**Request**: Multipart form data with CSV file

#### 4. Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "ollama"
}
```

### API Usage Examples

#### Using curl

```bash
# Score single transcript
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "HCP: Hello, how are you feeling today? Patient: I'm struggling...",
    "hcp_name": "Dr. Smith"
  }'

# Batch processing
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transcripts": [
      {"transcript": "Transcript 1...", "hcp_name": "Dr. Smith"},
      {"transcript": "Transcript 2...", "hcp_name": "Dr. Johnson"}
    ]
  }'

# Upload CSV
curl -X POST "http://localhost:8000/upload-csv" \
  -F "file=@transcripts.csv"
```

#### Using Python requests

```python
import requests

# Score single transcript
response = requests.post(
    "http://localhost:8000/score",
    json={
        "transcript": "HCP: Hello, how are you feeling today? Patient: I'm struggling...",
        "hcp_name": "Dr. Smith"
    }
)
result = response.json()
print(f"Overall Score: {result['overall_score']}")

# Batch processing
response = requests.post(
    "http://localhost:8000/batch",
    json={
        "transcripts": [
            {"transcript": "Transcript 1...", "hcp_name": "Dr. Smith"},
            {"transcript": "Transcript 2...", "hcp_name": "Dr. Johnson"}
        ]
    }
)
results = response.json()
print(f"Processed {len(results['results'])} transcripts")
```

## Understanding Scoring Results

### Scoring Dimensions

The system evaluates HCPs across four key dimensions:

1. **Empathy (25%)**
   - **Description**: Shows understanding and compassion
   - **Criteria**:
     - Demonstrates emotional awareness
     - Shows genuine concern for patient
     - Uses empathetic language
     - Validates patient feelings

2. **Clarity (25%)**
   - **Description**: Communicates clearly and effectively
   - **Criteria**:
     - Uses simple, understandable language
     - Explains concepts clearly
     - Provides structured information
     - Confirms patient understanding

3. **Accuracy (25%)**
   - **Description**: Provides accurate information and advice
   - **Criteria**:
     - Gives evidence-based recommendations
     - Corrects misconceptions appropriately
     - Provides accurate medical information
     - Refers to reliable sources

4. **Professionalism (25%)**
   - **Description**: Maintains professional standards and boundaries
   - **Criteria**:
     - Maintains appropriate boundaries
     - Shows respect and courtesy
     - Follows professional protocols
     - Demonstrates ethical behavior

### Scoring Scale

- **1: Poor** - Significant issues, needs immediate improvement
- **2: Below Average** - Several areas need improvement
- **3: Average** - Meets basic standards, some room for improvement
- **4: Good** - Above average performance, minor areas for improvement
- **5: Excellent** - Outstanding performance, exemplary standards

### Result Components

Each scoring result includes:

- **Dimension Scores**: Individual scores for each dimension (1-5)
- **Overall Score**: Weighted average of all dimensions
- **Confidence**: Model's confidence in the scoring (0-1)
- **Reasoning**: Detailed explanation of the scores
- **Strengths**: Identified positive aspects
- **Areas for Improvement**: Specific suggestions for enhancement

## Data Formats

### CSV Format

**Required columns**:
- `transcript` - The telehealth session transcript

**Optional columns**:
- `hcp_name` - Healthcare Provider name
- `session_date` - Date of the session (YYYY-MM-DD format)

**Example**:
```csv
transcript,hcp_name,session_date
"HCP: Hello, how are you feeling today? Patient: I'm struggling...","Dr. Smith","2024-01-15"
"HCP: Let's discuss your nutrition goals...","Dr. Johnson","2024-01-16"
```

### JSON Format

**Single transcript**:
```json
{
  "transcript": "HCP: Hello, how are you feeling today? Patient: I'm struggling...",
  "hcp_name": "Dr. Smith",
  "session_date": "2024-01-15"
}
```

**Batch transcripts**:
```json
{
  "transcripts": [
    {"transcript": "Transcript 1...", "hcp_name": "Dr. Smith"},
    {"transcript": "Transcript 2...", "hcp_name": "Dr. Johnson"}
  ]
}
```

## Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

**Key configuration options**:
```bash
# Model configuration
MODEL_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=mistral

# API configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend configuration
FRONTEND_PORT=8501
FRONTEND_HOST=0.0.0.0

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/hcp_rating.log
```

### Configuration File

Edit `configs/config.yaml` for advanced configuration:

```yaml
# Model configuration
model:
  name: "mistral-7b-instruct"
  temperature: 0.1
  max_length: 4096

# Ollama configuration
ollama:
  base_url: "http://localhost:11434"
  model_name: "mistral"
  temperature: 0.1
  max_tokens: 2048

# Scoring configuration
scoring:
  dimensions:
    empathy:
      weight: 0.25
      description: "Shows understanding and compassion"
    clarity:
      weight: 0.25
      description: "Communicates clearly and effectively"
    accuracy:
      weight: 0.25
      description: "Provides accurate information and advice"
    professionalism:
      weight: 0.25
      description: "Maintains professional standards"
```

## Troubleshooting

### Common Issues

1. **Ollama Not Running**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Check if running
   curl http://localhost:11434/api/tags
   ```

2. **Model Not Available**
   ```bash
   # Download model
   ollama pull mistral
   
   # List available models
   ollama list
   ```

3. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8501
   lsof -i :8000
   
   # Kill process or use different port
   ```

4. **Memory Issues**
   ```bash
   # Check available memory
   free -h
   
   # Close other applications
   # Consider using smaller model
   ```

### Performance Optimization

1. **Use GPU Acceleration** (if available):
   ```bash
   # Install CUDA version of PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Optimize Batch Size**:
   - Reduce batch size for large datasets
   - Process in smaller chunks

3. **Use Efficient Backend**:
   - Ollama for local processing
   - vLLM for high-performance inference

### Logging and Debugging

1. **Enable Debug Logging**:
   ```bash
   # Set log level
   export LOG_LEVEL=DEBUG
   
   # Check logs
   tail -f logs/hcp_rating.log
   ```

2. **Test Backend**:
   ```bash
   # Test Ollama integration
   python test_ollama.py
   
   # Test quick scoring
   python test_quick_scoring.py
   ```

## Best Practices

### Transcript Preparation

1. **Format Consistency**:
   - Use clear speaker labels (HCP:, Patient:)
   - Include complete conversation context
   - Maintain chronological order

2. **Content Quality**:
   - Ensure transcripts are complete and accurate
   - Include relevant medical context
   - Avoid heavily edited or summarized content

3. **Privacy Considerations**:
   - Remove personally identifiable information
   - Use anonymized data for testing
   - Follow HIPAA guidelines

### Scoring Interpretation

1. **Context Matters**:
   - Consider the specific medical context
   - Account for patient complexity
   - Factor in session duration

2. **Use Multiple Samples**:
   - Score multiple sessions per HCP
   - Look for consistent patterns
   - Consider trend analysis

3. **Combine with Other Metrics**:
   - Patient satisfaction scores
   - Clinical outcomes
   - Peer reviews

### System Maintenance

1. **Regular Updates**:
   ```bash
   # Update dependencies
   pip install -r requirements.txt --upgrade
   
   # Update Ollama models
   ollama pull mistral
   ```

2. **Backup Configuration**:
   ```bash
   # Backup config files
   cp configs/config.yaml configs/config.yaml.backup
   cp .env .env.backup
   ```

3. **Monitor Performance**:
   - Check system resources
   - Monitor API response times
   - Review error logs

## Advanced Features

### Custom Model Integration

1. **Use Custom Ollama Model**:
   ```bash
   # Create custom model
   ollama create my-model -f Modelfile
   
   # Update configuration
   # Edit configs/config.yaml
   ollama:
     model_name: "my-model"
   ```

2. **Fine-tuned Models**:
   - Train custom models on domain-specific data
   - Use LoRA for parameter-efficient fine-tuning
   - Deploy custom models via Ollama

### Batch Processing Optimization

1. **Parallel Processing**:
   ```python
   # Use concurrent processing for large batches
   from concurrent.futures import ThreadPoolExecutor
   
   def process_batch_parallel(transcripts, max_workers=4):
       with ThreadPoolExecutor(max_workers=max_workers) as executor:
           results = list(executor.map(score_transcript, transcripts))
       return results
   ```

2. **Memory Management**:
   ```python
   # Process in chunks to manage memory
   def process_large_batch(transcripts, chunk_size=100):
       results = []
       for i in range(0, len(transcripts), chunk_size):
           chunk = transcripts[i:i+chunk_size]
           chunk_results = process_batch(chunk)
           results.extend(chunk_results)
       return results
   ```

### Integration with External Systems

1. **API Integration**:
   ```python
   # Integrate with existing systems
   import requests
   
   def score_transcript_api(transcript, hcp_name):
       response = requests.post(
           "http://localhost:8000/score",
           json={"transcript": transcript, "hcp_name": hcp_name}
       )
       return response.json()
   ```

2. **Database Integration**:
   ```python
   # Store results in database
   import sqlite3
   
   def store_result(result):
       conn = sqlite3.connect('hcp_rating.db')
       cursor = conn.cursor()
       cursor.execute("""
           INSERT INTO scoring_results 
           (hcp_name, overall_score, empathy, clarity, accuracy, professionalism)
           VALUES (?, ?, ?, ?, ?, ?)
       """, (result['hcp_name'], result['overall_score'], 
             result['empathy'], result['clarity'], 
             result['accuracy'], result['professionalism']))
       conn.commit()
       conn.close()
   ```

## Support and Resources

### Documentation
- **Architecture**: `docs/ARCHITECTURE.md`
- **Technical Design**: `docs/TECHNICAL_DESIGN.md`
- **Project Summary**: `docs/PROJECT_SUMMARY.md`

### Community Resources
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support and discussions
- **Wiki**: Additional documentation and tutorials

### Getting Help
1. **Check Documentation**: Review relevant documentation files
2. **Search Issues**: Look for similar issues on GitHub
3. **Create Issue**: Report bugs with detailed information
4. **Community Support**: Ask questions in discussions

This usage guide provides comprehensive instructions for using the HCP Rating System effectively. For additional help, refer to the documentation or community resources. 
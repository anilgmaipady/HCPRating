# HCP Rating System - Demo Guide

This guide will walk you through how to set up and demonstrate the HCP Rating System, a comprehensive tool for evaluating healthcare provider performance using AI.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Demo Scenarios](#demo-scenarios)
4. [Frontend Demo](#frontend-demo)
5. [API Demo](#api-demo)
6. [CLI Demo](#cli-demo)
7. [Batch Processing Demo](#batch-processing-demo)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before running the demo, ensure you have:

- **Python 3.8+** installed
- **Ollama** installed and running
- **Mistral model** downloaded
- **Git** for cloning the repository

### Installing Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### Setting up the Environment

```bash
# Clone the repository
git clone <repository-url>
cd RD-RANK

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the Mistral model
ollama pull mistral
```

## Quick Start

### 1. Start Ollama Server

```bash
# Start Ollama in the background
ollama serve
```

### 2. Launch the Frontend

```bash
# Start the Streamlit frontend
streamlit run frontend/app.py
```

The frontend will be available at: `http://localhost:8501`

## Demo Scenarios

### Scenario 1: Single Transcript Evaluation

**Use Case**: Evaluate a single healthcare provider transcript

**Demo Steps**:
1. Open the frontend at `http://localhost:8501`
2. Navigate to "Single Transcript" page
3. **Click the "🎲 Generate Random Transcript" button to instantly fill the transcript area with a realistic example.**
4. (Optional) Enter or edit your own sample transcript.
5. Click "Score Transcript"
6. Review the results showing:
   - Radar chart of performance dimensions
   - Overall score (1-5 scale)
   - Detailed analysis and reasoning
   - Strengths and areas for improvement

### Scenario 2: Batch Processing

**Use Case**: Evaluate multiple transcripts at once

**Demo Steps**:
1. Navigate to "Batch Processing" page
2. Add multiple transcripts using the form
3. Process the entire batch
4. Review summary statistics and individual results
5. Export results as CSV or JSON

### Scenario 3: CSV Upload

**Use Case**: Process transcripts from a CSV file

**Demo Steps**:
1. Navigate to "Upload CSV" page
2. Upload a CSV file with columns:
   - `transcript` (required)
   - `hcp_name` (optional)
   - `session_date` (optional)
3. Review the data preview
4. Process the file
5. Download results

### Scenario 4: API Integration

**Use Case**: Integrate the scoring system into other applications

**Demo Steps**:
1. Start the API server:
   ```bash
   python src/deployment/start_server.py
   ```

2. Test the API endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Score single transcript
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Patient: I have a headache. HCP: How long have you had it?",
    "hcp_name": "Dr. Smith",
    "session_date": "2024-01-15"
  }'

# Batch scoring
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transcripts": [
      {
        "transcript": "Patient: I have a headache. HCP: How long have you had it?",
        "hcp_name": "Dr. Smith"
      }
    ]
  }'
```

## Frontend Demo

### Key Features to Demonstrate

1. **Real-time Scoring**: Show how quickly the system evaluates transcripts
2. **Visual Analytics**: Demonstrate the radar charts and bar graphs
3. **Detailed Analysis**: Show the reasoning and improvement suggestions
4. **Export Functionality**: Export results in multiple formats
5. **Configuration Management**: Show how to adjust scoring criteria

### Sample Transcripts for Demo

#### High-Performing HCP
```
Patient: "I'm really worried about my test results."
HCP: "I can see this is causing you a lot of anxiety. Let me explain what these results mean and what our next steps will be. We're going to work through this together."
Patient: "Thank you, that makes me feel better."
HCP: "You're welcome. Remember, I'm here to support you throughout this process. Do you have any questions about what we discussed?"
```

#### Average-Performing HCP
```
Patient: "My back has been hurting for a week."
HCP: "Let me examine you. Can you describe the pain?"
Patient: "It's sharp and gets worse when I move."
HCP: "I'll order some tests and prescribe pain medication. Come back in a week."
```

#### Low-Performing HCP
```
Patient: "I'm feeling very depressed lately."
HCP: "Take these pills. Next patient."
Patient: "But I have questions about side effects."
HCP: "Read the label. Next!"
```

## API Demo

### Starting the API Server

```bash
# Start the vLLM server
python src/deployment/start_server.py
```

### Testing API Endpoints

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Test health endpoint
response = requests.get(f"{BASE_URL}/health")
print(f"Health check: {response.json()}")

# Test single scoring
transcript_data = {
    "transcript": "Patient: I'm concerned about my symptoms. HCP: I understand your concern. Let's discuss what you're experiencing and develop a plan together.",
    "hcp_name": "Dr. Johnson",
    "session_date": "2024-01-15"
}

response = requests.post(f"{BASE_URL}/score", json=transcript_data)
result = response.json()
print(f"Scoring result: {json.dumps(result, indent=2)}")

# Test batch scoring
batch_data = {
    "transcripts": [
        {
            "transcript": "Patient: I have a fever. HCP: Let me check your temperature and symptoms.",
            "hcp_name": "Dr. Smith"
        },
        {
            "transcript": "Patient: My medication isn't working. HCP: Let's review your current treatment and make adjustments.",
            "hcp_name": "Dr. Brown"
        }
    ]
}

response = requests.post(f"{BASE_URL}/score/batch", json=batch_data)
results = response.json()
print(f"Batch results: {json.dumps(results, indent=2)}")
```

## CLI Demo

### Using the Command Line Interface

```bash
# Score a single transcript
python src/cli.py score --transcript "Patient: I have a headache. HCP: How long have you had it?"

# Score with additional metadata
python src/cli.py score \
  --transcript "Patient: I'm feeling anxious. HCP: I understand. Let's talk about what's causing this." \
  --hcp-name "Dr. Smith" \
  --session-date "2024-01-15"

# Batch scoring from file
python src/cli.py batch --input-file data/sample_transcripts.csv

# Export results
python src/cli.py score \
  --transcript "Patient: I have a headache. HCP: How long have you had it?" \
  --output-format json \
  --output-file results.json
```

## Batch Processing Demo

### Creating Sample Data

Create a CSV file (`demo_transcripts.csv`) with the following content:

```csv
transcript,hcp_name,session_date
"Patient: I'm worried about my symptoms. HCP: I understand your concern. Let's work together to address this.","Dr. Johnson","2024-01-15"
"Patient: My medication isn't working. HCP: Let me review your treatment plan and make adjustments.","Dr. Smith","2024-01-16"
"Patient: I have questions about my diagnosis. HCP: I'm here to help. What would you like to know?","Dr. Brown","2024-01-17"
```

### Processing the Batch

1. Upload the CSV file through the frontend
2. Review the preview
3. Process the batch
4. Analyze the results:
   - Average scores by dimension
   - Performance distribution
   - Top performers
   - Areas for improvement

## Advanced Demo Features

### 1. Configuration Management

Show how to customize scoring criteria:

```yaml
# configs/config.yaml
scoring:
  dimensions:
    empathy:
      weight: 0.3
      description: "Ability to understand and respond to patient emotions"
      criteria:
        - "Shows genuine concern for patient feelings"
        - "Uses empathetic language"
        - "Validates patient experiences"
    
    clarity:
      weight: 0.25
      description: "Clear and understandable communication"
      criteria:
        - "Explains medical terms in simple language"
        - "Provides clear instructions"
        - "Uses appropriate pace and tone"
```

### 2. Model Configuration

Demonstrate different model settings:

```yaml
model:
  name: "mistral:latest"
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.9
```

### 3. Export and Reporting

Show the various export options:
- JSON format for API integration
- CSV format for spreadsheet analysis
- Summary reports for management review

## Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Server Not Running
```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve
```

#### 2. Model Not Found
```bash
# List available models
ollama list

# Pull the required model
ollama pull mistral
```

#### 3. Frontend Connection Issues
```bash
# Check if the frontend is accessible
curl http://localhost:8501

# Restart the frontend
streamlit run frontend/app.py
```

#### 4. API Server Issues
```bash
# Check API health
curl http://localhost:8000/health

# Restart the API server
python src/deployment/start_server.py
```

#### 5. Memory Issues
If you encounter memory issues:
- Reduce batch size
- Use smaller model variants
- Close other applications
- Restart the system

### Performance Optimization

1. **Use GPU acceleration** when available
2. **Adjust batch sizes** based on available memory
3. **Optimize model parameters** for your use case
4. **Use caching** for repeated evaluations

## Demo Best Practices

### 1. Preparation
- Have sample transcripts ready
- Test all features beforehand
- Prepare backup options
- Set up proper lighting and audio

### 2. Presentation Flow
1. Start with a simple single transcript
2. Show the visual results
3. Demonstrate batch processing
4. Show API integration
5. Highlight export capabilities

### 3. Audience Engagement
- Ask for sample scenarios
- Show real-time scoring
- Demonstrate different use cases
- Highlight business value

### 4. Follow-up
- Provide contact information
- Share documentation links
- Offer additional demos
- Collect feedback

## Conclusion

This demo guide covers all the essential features of the HCP Rating System. The system is designed to be flexible and can be adapted to various healthcare evaluation scenarios. Whether you're demonstrating to healthcare administrators, IT teams, or clinical staff, this guide provides the tools needed for an effective presentation.

For additional support or questions, please refer to the main documentation or contact the development team. 
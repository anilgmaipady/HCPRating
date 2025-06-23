# HCP Rating System - Technical Design Documentation

## 🎯 System Overview

The HCP Rating System is a comprehensive AI-powered platform designed to evaluate Healthcare Providers based on telehealth session transcripts. The system employs a modular architecture with multiple backend options, providing flexibility for different deployment scenarios and user requirements.

## 🏗️ Technical Architecture

### **High-Level Design**

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Streamlit     │   CLI Tools     │   REST API                  │
│   Frontend      │   (Direct)      │   (FastAPI)                 │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   RD Scorer     │   Batch Proc    │   Training Pipeline         │
│   (Core Logic)  │   (Background)  │   (LoRA Fine-tuning)        │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Ollama        │   vLLM          │   Local Models              │
│   (Primary)     │   (Secondary)   │   (Tertiary)                │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## 🔧 Core Components Design

### **1. HCP Scorer Engine** (`src/inference/hcp_scorer.py`)

**Purpose**: Central scoring engine with multi-backend support

**Key Design Decisions**:
- **Backend Auto-Selection**: Intelligent backend selection based on availability
- **Fallback Mechanism**: Graceful degradation when preferred backend unavailable
- **Modular Design**: Easy addition of new backends
- **Result Validation**: Comprehensive score validation and error handling

**Class Structure**:
```python
class HCPScorer:
    def __init__(self, config_path: str = None, model_backend: str = "auto"):
        # Initialize with backend selection logic
        
    def _select_backend(self, model_backend: str) -> str:
        # Priority: Ollama > vLLM > Local > OpenAI
        
    def _initialize_backend(self):
        # Initialize selected backend
        
    def score_transcript(self, transcript: str, hcp_name: Optional[str] = None) -> ScoringResult:
        # Main scoring method
        
    def batch_score_transcripts(self, transcripts: List[Tuple[str, Optional[str]]]) -> List[ScoringResult]:
        # Batch processing
```

**Backend Selection Logic**:
```python
def _select_backend(self, model_backend: str) -> str:
    if model_backend != "auto":
        return model_backend
    
    # Priority order: Ollama > vLLM > Local > OpenAI
    if OLLAMA_AVAILABLE and check_ollama_availability():
        return "ollama"
    
    # Check vLLM server
    if self._check_vllm_server():
        return "vllm"
    
    # Check local models
    if self._check_local_models():
        return "local"
    
    # Fallback to OpenAI
    return "openai"
```

### **2. Ollama Integration** (`src/inference/ollama_client.py`)

**Purpose**: Primary backend for local model deployment

**Design Features**:
- **Model Management**: Automatic model downloading and verification
- **Connection Pooling**: Efficient connection management
- **Error Recovery**: Automatic retry mechanisms
- **Configuration**: Flexible model parameter configuration

**Key Classes**:
```python
class OllamaManager:
    """Manages Ollama connections and model availability"""
    
class OllamaClient:
    """Client for interacting with Ollama models"""
    
def check_ollama_availability():
    """Check if Ollama server is available"""
```

### **3. Frontend Design** (`frontend/app.py`)

**Purpose**: User-friendly web interface with backend flexibility

**Design Features**:
- **Backend Detection**: Automatic detection of available backends
- **Fallback Mechanism**: Seamless switching between backends
- **Real-time Feedback**: Live status updates and error handling
- **Responsive Design**: Mobile-friendly interface

**Backend Integration**:
```python
def check_api_health():
    """Check if API server is running"""
    
def check_ollama_availability():
    """Check if Ollama is available"""
    
def score_transcript(transcript, hcp_name=None, session_date=None):
    """Score transcript with automatic backend selection"""
    # Try API first, fallback to direct Ollama
```

### **4. API Server Design** (`src/api/main.py`)

**Purpose**: RESTful API service with Ollama backend preference

**Design Features**:
- **Ollama-First**: Prioritizes Ollama backend initialization
- **Graceful Fallback**: Falls back to auto backend selection
- **Health Monitoring**: Comprehensive health checks
- **Error Handling**: Detailed error responses

**Initialization Logic**:
```python
# Initialize HCP Scorer with Ollama preference
try:
    hcp_scorer = HCPScorer(model_backend="ollama")
    logger.info("HCP Scorer initialized successfully with Ollama backend")
except Exception as e:
    logger.warning(f"Failed to initialize with Ollama backend: {e}")
    try:
        hcp_scorer = HCPScorer()  # Auto backend selection
        logger.info("HCP Scorer initialized successfully with auto backend selection")
    except Exception as e2:
        logger.error(f"Failed to initialize HCP Scorer: {e2}")
        hcp_scorer = None
```

## 🚀 Startup Scripts Design

### **1. Simple Startup** (`run.py`)

**Purpose**: One-command startup with comprehensive setup

**Design Features**:
- **Environment Management**: Automatic virtual environment creation
- **Dependency Installation**: Automatic package installation
- **Backend Verification**: Ollama availability checking
- **Integration Testing**: Automatic system testing
- **User Guidance**: Clear error messages and instructions

**Flow**:
```python
def main():
    # 1. Check virtual environment
    # 2. Check Ollama availability
    # 3. Install dependencies if needed
    # 4. Setup Ollama
    # 5. Test integration
    # 6. Start web interface
```

### **2. Advanced Startup** (`start.py`)

**Purpose**: Multi-option startup with granular control

**Design Features**:
- **Command-Based**: Multiple startup options
- **OS Detection**: Platform-specific instructions
- **Backend Management**: Ollama setup and testing
- **Service Isolation**: Individual service startup

**Commands**:
```bash
python start.py quick          # Quick start with Ollama
python start.py vllm           # Start vLLM server
python start.py api            # Start API server only
python start.py streamlit      # Start frontend only
python start.py ollama-setup   # Setup Ollama
python start.py ollama-test    # Test Ollama integration
```

### **3. API Server Startup** (`start_api.py`)

**Purpose**: Dedicated API server startup with Ollama backend

**Design Features**:
- **Ollama Verification**: Pre-flight Ollama checks
- **Development Mode**: Auto-reload for development
- **Health Monitoring**: Built-in health checks
- **Error Recovery**: Graceful error handling

## 📊 Data Flow Design

### **Single Transcript Scoring Flow**

```
User Input → Frontend → Backend Selection → Model → Scoring → Results → Visualization
     │           │              │              │        │         │         │
     │           │              │              │        │         │         └─ Charts & Reports
     │           │              │              │        │         └─ Validation & Processing
     │           │              │              │        └─ Multi-dimensional Analysis
     │           │              │              │
     │           │              │              │ Ollama/vLLM/Local
     │           │              │              └─ Auto-selection Logic
     │           │              └─ Backend Detection
     │           └─ Transcript + Metadata
```

### **Batch Processing Flow**

```
CSV Upload → Validation → Chunking → Parallel Scoring → Aggregation → Report Generation
     │           │           │           │               │              │
     │           │           │           │               │              └─ Excel/PDF Reports
     │           │           │           │               └─ Score Aggregation
     │           │           │           │
     │           │           │           └─ Concurrent Processing
     │           │           └─ Memory Management
     │           └─ Data Validation
     └─ File Upload
```

### **Backend Selection Flow**

```
Start → Check Ollama → Available? → Yes → Use Ollama
  │         │              │
  │         │              └─ No → Check vLLM → Available? → Yes → Use vLLM
```

## 🔁 Feedback Loop Design

To enable continuous model improvement, a feedback loop is implemented to collect corrections on inaccurate predictions and use them for retraining.

### 1. **Feedback Collection (Frontend)**
-   **Component**: A `st.expander` containing a `st.form` is added to the `display_scoring_results` function in `frontend/app.py`.
-   **Trigger**: This form is displayed on the results page after every single transcript evaluation.
-   **Functionality**:
    *   It is pre-populated with the model's original predictions.
    *   Users can override the 1-5 scores for each dimension using `st.number_input`.
    *   Users can edit the `reasoning`, `strengths`, and `areas_for_improvement` in `st.text_area` fields.

### 2. **Data Storage**
-   **File**: Corrected feedback is appended to `feedback/collected_feedback.jsonl`.
-   **Format**: The data is stored in JSONL format, which is identical to the training data format. This makes it seamless to use for retraining.
-   **Structure**: Each JSON object includes the original `transcript` and all the corrected fields provided by the user.
-   **Directory Creation**: The `feedback/` directory is created automatically if it does not exist.

### 3. **Retraining Workflow**
-   **Data Source**: The `train_model.py` script can be pointed directly to the `feedback/collected_feedback.jsonl` file using the `--data` argument.
-   **Process**:
    1.  Collect a sufficient amount of feedback data.
    2.  Run the training script, using the feedback file as the data source.
    3.  **Recommendation**: For optimal results, it's best to combine the feedback data with the original training dataset. This prevents the model from "catastrophic forgetting" and ensures it improves on its mistakes while retaining its general knowledge.
        ```bash
        # Example of combining datasets
        cat data/original_training_data.jsonl feedback/collected_feedback.jsonl > data/combined_training_data.jsonl
        
        # Retrain on the combined dataset
        python train_model.py --data data/combined_training_data.jsonl
        ```

This design creates a virtuous cycle where the model gets progressively better as more feedback is collected from real-world use.

## 🧪 Testing and Quality Assurance

### **Test Coverage and Automation**
- **Unit Tests**: Pytest for individual component testing
- **Integration Tests**: End-to-end workflow testing
- **API Tests**: HTTPX for REST API endpoint testing
- **Performance Tests**: Locust for load and stress testing

### **Code Quality and Standards**
- **Code Formatting**: Black for consistent formatting
- **Linting**: Ruff for high-performance code quality checks
- **Type Checking**: MyPy for static type safety
- **Pre-commit Hooks**: Automated checks before every commit

## 🔒 Security and Reliability

### **API Security**
- **CORS**: Configured for web access
- **Input Validation**: Pydantic models for request validation
- **Sanitization**: Standard libraries for input cleaning
- **Secrets Management**: `python-dotenv` for environment secrets

### **Data Privacy**
- **Local Processing**: Default operation is on-premise
- **No External Data Transfer**: No data sent to external services (unless OpenAI is explicitly configured)
- **Configurable Data Retention**: Policies can be set for data storage
- **Export Controls**: User-managed data exports 
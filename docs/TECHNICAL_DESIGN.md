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
     │           │              │              └─ Ollama/vLLM/Local
     │           │              └─ Auto-selection Logic
     │           └─ Backend Detection
     └─ Transcript + Metadata
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
  │         │                                 │
  │         │                                 └─ No → Check Local → Available? → Yes → Use Local
  │         │                                                                 │
  │         │                                                                 └─ No → Use OpenAI
  │         └─ Error → Show Installation Guide
  └─ Complete
```

## ⚙️ Configuration Design

### **Configuration Structure** (`configs/config.yaml`)

**Design Principles**:
- **Hierarchical**: Logical grouping of related settings
- **Environment-Aware**: Different settings for different environments
- **Validation**: Built-in configuration validation
- **Documentation**: Self-documenting configuration

**Key Sections**:
```yaml
# Model Configuration
model:
  name: "models/mistral-7b-instruct"
  max_length: 4096
  temperature: 0.1

# Ollama Configuration (Primary Backend)
ollama:
  base_url: "http://localhost:11434"
  model_name: "mistral"
  temperature: 0.1
  max_tokens: 2048
  timeout: 120

# vLLM Configuration (Secondary Backend)
server:
  host: "0.0.0.0"
  port: 8000
  tensor_parallel_size: 1
  max_model_len: 8192
  gpu_memory_utilization: 0.9

# Scoring Criteria
scoring:
  dimensions:
    empathy:
      weight: 0.25
      description: "Ability to understand and respond to patient emotions"
      criteria:
        - "Shows genuine concern for patient's feelings"
        - "Uses empathetic language and tone"
        - "Acknowledges patient's emotional state"
    # ... other dimensions

# API Configuration
api:
  title: "RD Rating API"
  description: "API for rating Registered Dietitians"
  version: "1.0.0"
  docs_url: "/docs"
  redoc_url: "/redoc"
```

## 🔒 Security Design

### **Data Privacy**

**Design Principles**:
- **Local Processing**: All data processed locally by default
- **No External Dependencies**: Optional external API usage
- **Configurable Privacy**: User-controlled data handling

**Implementation**:
```python
# Local processing with Ollama
def _get_ollama_response(self, prompt: str) -> str:
    """Get response from local Ollama instance"""
    # No data leaves the local machine
    
# Optional external processing
def _get_openai_response(self, prompt: str) -> str:
    """Get response from OpenAI API (optional)"""
    # Only used if explicitly configured
```

### **Input Validation**

**Design Features**:
- **Comprehensive Validation**: All inputs validated before processing
- **Sanitization**: Input sanitization to prevent injection attacks
- **Error Handling**: Graceful error handling with user-friendly messages

**Implementation**:
```python
class TranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=10000)
    rd_name: Optional[str] = Field(None, max_length=100)
    session_date: Optional[str] = Field(None, regex=r'^\d{4}-\d{2}-\d{2}$')
```

## 🧪 Testing Design

### **Test Strategy**

**Design Principles**:
- **Comprehensive Coverage**: All components tested
- **Integration Testing**: End-to-end workflow testing
- **Backend Testing**: Multiple backend testing
- **Error Testing**: Error condition testing

**Test Structure**:
```
tests/
├── test_hcp_scorer.py      # Core scoring logic tests
├── test_ollama.py         # Ollama integration tests
├── test_api.py           # API endpoint tests
└── test_integration.py   # End-to-end tests
```

### **Test Scripts**

**Ollama Test** (`test_ollama.py`):
```python
def test_ollama_integration():
    """Test complete Ollama integration"""
    # 1. Check Ollama availability
    # 2. Test model loading
    # 3. Test scoring functionality
    # 4. Validate results
```

**Quick Test** (`test_quick_scoring.py`):
```python
def test_basic_functionality():
    """Test basic system functionality"""
    # 1. Test backend detection
    # 2. Test scoring
    # 3. Test result validation
```

## 📈 Performance Design

### **Performance Optimization**

**Design Strategies**:
- **Backend Selection**: Optimal backend for use case
- **Caching**: Result caching for repeated requests
- **Batch Processing**: Efficient batch operations
- **Memory Management**: Optimized memory usage

**Performance Characteristics**:
| Backend | Startup Time | Memory Usage | Throughput | Setup Complexity |
|---------|-------------|--------------|------------|------------------|
| **Ollama** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| vLLM | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Local | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| OpenAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### **Scalability Design**

**Horizontal Scaling**:
- **API Server**: Multiple API server instances
- **Load Balancing**: Nginx load balancer
- **Database**: Optional database integration
- **Caching**: Redis caching layer

**Vertical Scaling**:
- **GPU Acceleration**: vLLM GPU optimization
- **Memory Optimization**: Efficient memory usage
- **CPU Optimization**: Multi-threading support

## 🔮 Future Design Considerations

### **Planned Enhancements**

**Multi-language Support**:
- **Internationalization**: Multi-language transcript support
- **Language Detection**: Automatic language detection
- **Localized Models**: Language-specific models

**Advanced Analytics**:
- **Trend Analysis**: Performance trend tracking
- **Comparative Analysis**: RD comparison features
- **Predictive Analytics**: Performance prediction

**Model Management**:
- **Model Versioning**: Version control for models
- **A/B Testing**: Model comparison framework
- **Automated Training**: Continuous model improvement

### **Architecture Evolution**

**Microservices Migration**:
- **Service Decomposition**: Component separation
- **API Gateway**: Centralized API management
- **Service Discovery**: Dynamic service discovery
- **Container Orchestration**: Kubernetes deployment

**Data Pipeline**:
- **Stream Processing**: Real-time data processing
- **Data Lake**: Centralized data storage
- **ETL Processes**: Automated data transformation
- **Analytics Platform**: Advanced analytics integration

This technical design provides a robust foundation for the RD Rating System while maintaining flexibility for future enhancements and scalability requirements. 
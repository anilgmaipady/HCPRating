# HCP Rating System - Training Guide

This guide explains how to train and fine-tune the HCP Rating System model for your specific use case.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Training Process](#training-process)
5. [Model Deployment](#model-deployment)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

## Overview

The HCP Rating System uses **LoRA (Low-Rank Adaptation)** fine-tuning to adapt the base Mistral-7B-Instruct model for healthcare provider evaluation. This approach:

- **Efficient**: Requires minimal computational resources
- **Fast**: Training completes in hours, not days
- **Effective**: Maintains base model capabilities while adding domain expertise
- **Flexible**: Easy to adapt for different healthcare domains

## Prerequisites

### Hardware Requirements
- **GPU**: 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space for model and data

### Software Requirements
```bash
# Install training dependencies
pip install torch transformers datasets peft bitsandbytes accelerate
pip install wandb  # Optional: for experiment tracking
```

### Model Requirements
- Base model: Mistral-7B-Instruct-v0.1
- Download via Ollama: `ollama pull mistral`

## Data Preparation

### Training Data Format

The system accepts two data formats:

#### 1. JSONL Format (Recommended)
```json
{
  "transcript": "RD: Hello, how are you feeling today? Patient: I'm struggling with my diet. RD: I understand this can be challenging. Let's work together to find solutions that work for you.",
  "empathy": 4,
  "clarity": 4,
  "accuracy": 4,
  "professionalism": 5,
  "reasoning": "The RD shows good empathy by acknowledging the patient's struggle and offering collaborative support.",
  "strengths": ["Shows empathy", "Offers collaborative approach", "Professional tone"],
  "areas_for_improvement": ["Could ask more specific questions about diet challenges"]
}
```

#### 2. CSV Format
```csv
transcript,empathy,clarity,accuracy,professionalism,reasoning,strengths,areas_for_improvement
"RD: Hello, how are you feeling today? Patient: I'm struggling...",4,4,4,5,"The RD shows good empathy...","Shows empathy;Offers collaborative approach","Could ask more specific questions"
```

### Data Quality Guidelines

#### Transcript Quality
- **Length**: 100-2000 words per transcript
- **Format**: Clear speaker identification (RD:, Patient:)
- **Content**: Realistic healthcare conversations
- **Diversity**: Various scenarios, patient types, and difficulty levels

#### Scoring Guidelines
- **Empathy (1-5)**: Understanding and compassion
- **Clarity (1-5)**: Communication effectiveness
- **Accuracy (1-5)**: Information quality and correctness
- **Professionalism (1-5)**: Professional standards and boundaries

#### Annotation Quality
- **Consistency**: Multiple annotators should agree
- **Reasoning**: Provide clear justification for scores
- **Balanced**: Include examples across all score ranges
- **Specific**: Detailed strengths and improvement areas

### Creating Sample Data

Use the built-in sample data generator:

```bash
python src/training/fine_tune.py --create_sample
```

This creates `data/sample_training_data.jsonl` with example training data.

## Training Process

### 1. Basic Training

```bash
# Train with your data
python src/training/fine_tune.py --data_path data/your_training_data.jsonl

# Train with sample data
python src/training/fine_tune.py --data_path data/sample_training_data.jsonl --output_dir models/my_finetuned_model
```

### 2. Training with Custom Configuration

```bash
# Use custom config file
python src/training/fine_tune.py \
  --data_path data/your_training_data.jsonl \
  --output_dir models/custom_model \
  --config_path configs/custom_training.yaml
```

### 3. Training Parameters

Key parameters in `configs/config.yaml`:

```yaml
training:
  learning_rate: 2e-4          # Learning rate for fine-tuning
  num_epochs: 3                # Number of training epochs
  batch_size: 2                # Batch size per device
  gradient_accumulation_steps: 4  # Gradient accumulation steps
  warmup_steps: 100            # Learning rate warmup steps
  save_steps: 500              # Save checkpoint every N steps
  eval_steps: 500              # Evaluate every N steps
  logging_steps: 10            # Log every N steps
  
  # LoRA Configuration
  lora:
    r: 8                       # LoRA rank
    lora_alpha: 16             # LoRA alpha parameter
    lora_dropout: 0.1          # LoRA dropout rate
    bias: "none"               # LoRA bias handling
```

### 4. Training Output

The training process creates:

```
models/finetuned_rd_model/
├── adapter_config.json        # LoRA configuration
├── adapter_model.bin          # LoRA weights
├── training_config.json       # Training metadata
├── pytorch_model.bin          # Model weights
├── config.json               # Model configuration
└── tokenizer.json            # Tokenizer files
```

## Model Deployment

### 1. Update Configuration

Edit `configs/config.yaml` to use your fine-tuned model:

```yaml
model:
  name: "models/finetuned_rd_model"  # Path to your fine-tuned model
  max_length: 4096
  temperature: 0.1
  load_in_4bit: false
  device_map: "auto"
```

### 2. Test the Fine-tuned Model

```bash
# Test with CLI
python src/cli.py score "RD: Hello, how are you feeling today? Patient: I'm struggling with my diet." --backend local

# Test with API
python src/deployment/start_server.py
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transcript": "RD: Hello, how are you feeling today? Patient: I\'m struggling with my diet."}'
```

### 3. Deploy to Production

```bash
# Start the API server with fine-tuned model
python src/deployment/start_server.py

# Or use the frontend
streamlit run frontend/app.py
```

## Advanced Configuration

### Custom Training Configuration

Create `configs/custom_training.yaml`:

```yaml
# Model Configuration
model:
  name: "mistralai/Mistral-7B-Instruct-v0.1"
  max_length: 4096
  load_in_4bit: true
  device_map: "auto"

# Training Configuration
training:
  learning_rate: 1e-4          # Lower learning rate for stability
  num_epochs: 5                # More epochs for better performance
  batch_size: 1                # Smaller batch size for memory constraints
  gradient_accumulation_steps: 8
  warmup_steps: 200
  save_steps: 250
  eval_steps: 250
  logging_steps: 5
  
  # LoRA Configuration
  lora:
    r: 16                      # Higher rank for more capacity
    lora_alpha: 32
    lora_dropout: 0.05
    bias: "none"
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Scoring Configuration
scoring:
  dimensions:
    empathy:
      weight: 0.3
      description: "Shows understanding and compassion"
    clarity:
      weight: 0.25
      description: "Communicates clearly and effectively"
    accuracy:
      weight: 0.25
      description: "Provides accurate information and advice"
    professionalism:
      weight: 0.2
      description: "Maintains professional standards and boundaries"
```

### Multi-GPU Training

For multi-GPU setups, modify the training script:

```python
# In fine_tune.py, update TrainingArguments
training_args = TrainingArguments(
    output_dir=str(output_path),
    per_device_train_batch_size=1,  # Smaller batch size per GPU
    gradient_accumulation_steps=8,   # Increase for effective batch size
    dataloader_num_workers=4,        # Parallel data loading
    ddp_find_unused_parameters=False,
    # ... other arguments
)
```

### Experiment Tracking

Enable Weights & Biases tracking:

```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login

# Update training script to enable wandb
# In fine_tune.py, set report_to=["wandb"]
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size and increase gradient accumulation
training:
  batch_size: 1
  gradient_accumulation_steps: 8

# Enable gradient checkpointing
model:
  gradient_checkpointing: true
```

#### 2. Slow Training
```bash
# Increase batch size if memory allows
training:
  batch_size: 4
  gradient_accumulation_steps: 2

# Use mixed precision training
training_args = TrainingArguments(
    fp16=True,
    dataloader_pin_memory=True,
    # ... other arguments
)
```

#### 3. Poor Model Performance
```bash
# Increase training data quality and quantity
# Adjust learning rate
training:
  learning_rate: 1e-4  # Lower for stability

# Increase LoRA rank
lora:
  r: 16
  lora_alpha: 32
```

#### 4. Model Not Loading
```bash
# Check model path
model:
  name: "models/finetuned_rd_model"  # Ensure path is correct

# Verify model files exist
ls -la models/finetuned_rd_model/
```

### Performance Optimization

#### 1. Data Loading
- Use SSD storage for faster data loading
- Increase `dataloader_num_workers`
- Enable `dataloader_pin_memory`

#### 2. Memory Management
- Use gradient checkpointing
- Enable 4-bit quantization
- Use gradient accumulation

#### 3. Training Speed
- Use mixed precision training (fp16)
- Optimize batch size for your hardware
- Use multiple GPUs if available

## Best Practices

### 1. Data Quality
- **Diverse Data**: Include various scenarios and difficulty levels
- **Consistent Annotation**: Use multiple annotators and measure agreement
- **Balanced Distribution**: Ensure examples across all score ranges
- **Realistic Transcripts**: Use authentic healthcare conversations

### 2. Training Strategy
- **Start Small**: Begin with sample data to verify setup
- **Gradual Scaling**: Increase data size and model complexity gradually
- **Regular Evaluation**: Monitor training metrics and model performance
- **Validation Set**: Reserve 20% of data for validation

### 3. Model Evaluation
- **Human Correlation**: Compare model scores with human evaluators
- **Inter-rater Reliability**: Measure consistency across multiple runs
- **Domain Testing**: Test on real-world scenarios
- **A/B Testing**: Compare with baseline models

### 4. Deployment
- **Gradual Rollout**: Deploy to small user groups first
- **Monitoring**: Track model performance in production
- **Feedback Loop**: Collect user feedback for model improvement
- **Version Control**: Maintain model versions and rollback capability

## Example Training Workflow

```bash
# 1. Prepare your data
python src/training/fine_tune.py --create_sample
# Edit data/sample_training_data.jsonl with your data

# 2. Train the model
python src/training/fine_tune.py \
  --data_path data/your_training_data.jsonl \
  --output_dir models/my_finetuned_model

# 3. Update configuration
# Edit configs/config.yaml to use your model

# 4. Test the model
python src/cli.py score "Your test transcript here"

# 5. Deploy
python src/deployment/start_server.py
```

## Conclusion

This training guide provides everything you need to fine-tune the HCP Rating System for your specific use case. The LoRA approach makes training efficient and accessible, while maintaining the quality of the base model.

For additional support or questions, refer to the main documentation or contact the development team. 
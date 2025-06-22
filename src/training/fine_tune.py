#!/usr/bin/env python3
"""
Fine-tuning script for RD Rating System using LoRA
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import yaml
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb
from tqdm import tqdm

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class RDFineTuner:
    """Fine-tuning class for RD Rating System."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the fine-tuner."""
        self.logger = setup_logging()
        self.config = load_config() if config_path is None else self._load_config(config_path)
        
        # Training configuration
        self.training_config = self.config.get('training', {})
        self.model_config = self.config.get('model', {})
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_model_and_tokenizer(self):
        """Prepare model and tokenizer for fine-tuning."""
        self.logger.info("Loading model and tokenizer...")
        
        model_name = self.model_config.get('name', 'mistralai/Mistral-7B-Instruct-v0.1')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with quantization
        load_in_4bit = self.model_config.get('load_in_4bit', True)
        device_map = self.model_config.get('device_map', 'auto')
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        if load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA configuration
        lora_config = self.training_config.get('lora', {})
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', 'none'),
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        self.logger.info("Model and tokenizer prepared successfully")
    
    def create_training_prompt(self, transcript: str, scores: Dict[str, int], 
                             reasoning: str, strengths: List[str], 
                             areas_for_improvement: List[str]) -> str:
        """Create training prompt in the format expected by the model."""
        
        # Create scoring criteria text
        scoring_criteria = self.config.get('scoring', {}).get('dimensions', {})
        criteria_text = ""
        for dimension, config in scoring_criteria.items():
            criteria_text += f"\n{dimension.title()}:\n"
            criteria_text += f"  Description: {config.get('description', '')}\n"
            criteria_text += "  Criteria:\n"
            for criterion in config.get('criteria', []):
                criteria_text += f"    - {criterion}\n"
        
        # Create the prompt
        prompt = f"""You are an expert evaluator of Registered Dietitians conducting telehealth sessions. 

Analyze the following transcript and rate the RD across four dimensions on a scale of 1-5:

{criteria_text}

Scoring Guidelines:
- 1: Poor - Significant issues, needs immediate improvement
- 2: Below Average - Several areas need improvement
- 3: Average - Meets basic standards, some room for improvement
- 4: Good - Above average performance, minor areas for improvement
- 5: Excellent - Outstanding performance, exemplary standards

Transcript:
{transcript}

Please provide your evaluation in the following JSON format:
{{
    "empathy": {scores['empathy']},
    "clarity": {scores['clarity']},
    "accuracy": {scores['accuracy']},
    "professionalism": {scores['professionalism']},
    "overall_score": {sum(scores.values()) / len(scores):.2f},
    "confidence": 0.85,
    "reasoning": "{reasoning}",
    "strengths": {json.dumps(strengths)},
    "areas_for_improvement": {json.dumps(areas_for_improvement)}
}}"""
        
        return prompt
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training."""
        self.logger.info(f"Preparing dataset from {data_path}")
        
        # Load data
        if data_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            dataset = Dataset.from_pandas(df)
        else:
            raise ValueError("Unsupported data format. Use .jsonl or .csv")
        
        # Process data into prompts
        def process_example(example):
            # Extract data from example
            transcript = example.get('transcript', '')
            scores = {
                'empathy': example.get('empathy', 3),
                'clarity': example.get('clarity', 3),
                'accuracy': example.get('accuracy', 3),
                'professionalism': example.get('professionalism', 3)
            }
            reasoning = example.get('reasoning', '')
            strengths = example.get('strengths', [])
            areas_for_improvement = example.get('areas_for_improvement', [])
            
            # Create prompt
            prompt = self.create_training_prompt(
                transcript, scores, reasoning, strengths, areas_for_improvement
            )
            
            # Tokenize
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.model_config.get('max_length', 4096),
                padding=False,
                return_tensors=None
            )
            
            return tokenized
        
        # Apply processing
        processed_dataset = dataset['train'].map(
            process_example,
            remove_columns=dataset['train'].column_names,
            desc="Processing dataset"
        )
        
        self.logger.info(f"Dataset prepared with {len(processed_dataset)} examples")
        return processed_dataset
    
    def train(self, data_path: str, output_dir: str = "models/finetuned_rd_model"):
        """Train the model."""
        self.logger.info("Starting fine-tuning process...")
        
        # Prepare model and tokenizer
        self.prepare_model_and_tokenizer()
        
        # Prepare dataset
        dataset = self.prepare_dataset(data_path)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            per_device_train_batch_size=self.training_config.get('batch_size', 2),
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
            num_train_epochs=self.training_config.get('num_epochs', 3),
            learning_rate=self.training_config.get('learning_rate', 2e-4),
            warmup_steps=self.training_config.get('warmup_steps', 100),
            save_steps=self.training_config.get('save_steps', 500),
            eval_steps=self.training_config.get('eval_steps', 500),
            logging_steps=self.training_config.get('logging_steps', 10),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        self.logger.info(f"Saving model to {output_path}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_path)
        
        # Save training configuration
        training_config = {
            'model_name': self.model_config.get('name'),
            'training_args': training_args.to_dict(),
            'lora_config': self.training_config.get('lora', {}),
            'data_path': data_path,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path / 'training_config.json', 'w') as f:
            json.dump(training_config, f, indent=2)
        
        self.logger.info("Training completed successfully!")
        return output_path

def create_sample_dataset(output_path: str = "data/sample_training_data.jsonl"):
    """Create a sample dataset for testing."""
    sample_data = [
        {
            "transcript": "RD: Hello, how are you feeling today? Patient: I'm really struggling with my diet. RD: I understand this can be challenging. Let's work together to find solutions that work for you. What specific difficulties are you facing?",
            "empathy": 4,
            "clarity": 4,
            "accuracy": 4,
            "professionalism": 5,
            "reasoning": "The RD shows good empathy by acknowledging the patient's struggle and offering collaborative support. Communication is clear and professional.",
            "strengths": ["Shows empathy", "Offers collaborative approach", "Professional tone"],
            "areas_for_improvement": ["Could ask more specific questions about diet challenges"]
        },
        {
            "transcript": "RD: You need to eat more vegetables. Patient: I don't like vegetables. RD: You have to eat them anyway. It's good for you.",
            "empathy": 2,
            "clarity": 3,
            "accuracy": 4,
            "professionalism": 2,
            "reasoning": "The RD lacks empathy and doesn't address the patient's concerns. Communication is directive rather than collaborative.",
            "strengths": ["Provides accurate nutritional information"],
            "areas_for_improvement": ["Lacks empathy", "Doesn't address patient preferences", "Too directive"]
        },
        {
            "transcript": "RD: Thank you for sharing that with me. I can see this has been really difficult for you. Let's explore some options together. What types of foods do you enjoy? Patient: I like pasta and bread. RD: Great! We can work with that. Let me show you some healthy pasta options and how to balance your meals.",
            "empathy": 5,
            "clarity": 5,
            "accuracy": 4,
            "professionalism": 5,
            "reasoning": "Excellent empathy shown through validation and collaborative approach. Clear communication and professional demeanor.",
            "strengths": ["High empathy", "Collaborative approach", "Clear communication", "Professional"],
            "areas_for_improvement": []
        }
    ]
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write to JSONL file
    with open(output_path, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample dataset created at {output_path}")
    return output_path

def main():
    """Main function for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune RD Rating Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_rd_model", help="Output directory")
    parser.add_argument("--config_path", type=str, help="Path to custom config file")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset")
    
    args = parser.parse_args()
    
    # Create sample dataset if requested
    if args.create_sample:
        data_path = create_sample_dataset()
        print(f"Using sample dataset: {data_path}")
    else:
        data_path = args.data_path
    
    # Initialize fine-tuner
    fine_tuner = RDFineTuner(args.config_path)
    
    # Train the model
    output_path = fine_tuner.train(data_path, args.output_dir)
    
    print(f"Training completed! Model saved to: {output_path}")

if __name__ == "__main__":
    main() 
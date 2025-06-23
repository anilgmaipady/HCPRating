#!/usr/bin/env python3
"""
Quick Training Script for HCP Rating System

This script provides a simple interface for training the HCP Rating System model.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.training.fine_tune import RDFineTuner, create_sample_dataset

def main():
    """Main training function with user-friendly interface."""
    parser = argparse.ArgumentParser(
        description="Train HCP Rating System Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample data and train
  python train_model.py --create-sample --train

  # Train with your own data
  python train_model.py --data data/my_training_data.jsonl --train

  # Quick training with sample data
  python train_model.py --quick-train

  # Full training with custom config
  python train_model.py --data data/my_data.jsonl --config configs/custom.yaml --train
        """
    )
    
    parser.add_argument(
        "--create-sample", 
        action="store_true", 
        help="Create sample training data"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        help="Path to training data file (.jsonl or .csv)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="models/finetuned_model",
        help="Output directory for trained model (default: models/finetuned_model)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Start training after data preparation"
    )
    
    parser.add_argument(
        "--quick-train", 
        action="store_true", 
        help="Quick training with sample data (creates sample data and trains)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=2,
        help="Training batch size (default: 2)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    
    args = parser.parse_args()
    
    print("🚀 HCP Rating System - Training Interface")
    print("=" * 50)
    
    # Quick train mode
    if args.quick_train:
        print("📝 Creating sample training data...")
        data_path = create_sample_dataset()
        print(f"✅ Sample data created: {data_path}")
        
        print("🏋️ Starting quick training...")
        fine_tuner = RDFineTuner(args.config)
        
        # Override config with command line arguments
        if args.epochs != 3:
            fine_tuner.training_config['num_epochs'] = args.epochs
        if args.batch_size != 2:
            fine_tuner.training_config['batch_size'] = args.batch_size
        if args.learning_rate != 2e-4:
            fine_tuner.training_config['learning_rate'] = args.learning_rate
        
        output_path = fine_tuner.train(data_path, args.output)
        print(f"✅ Training completed! Model saved to: {output_path}")
        return
    
    # Create sample data
    if args.create_sample:
        print("📝 Creating sample training data...")
        data_path = create_sample_dataset()
        print(f"✅ Sample data created: {data_path}")
        
        if args.train:
            print("🏋️ Starting training with sample data...")
            fine_tuner = RDFineTuner(args.config)
            output_path = fine_tuner.train(data_path, args.output)
            print(f"✅ Training completed! Model saved to: {output_path}")
        return
    
    # Train with provided data
    if args.train:
        if not args.data:
            print("❌ Error: Please provide training data with --data or use --create-sample")
            return
        
        if not Path(args.data).exists():
            print(f"❌ Error: Training data file not found: {args.data}")
            return
        
        print(f"🏋️ Starting training with data: {args.data}")
        fine_tuner = RDFineTuner(args.config)
        
        # Override config with command line arguments
        if args.epochs != 3:
            fine_tuner.training_config['num_epochs'] = args.epochs
        if args.batch_size != 2:
            fine_tuner.training_config['batch_size'] = args.batch_size
        if args.learning_rate != 2e-4:
            fine_tuner.training_config['learning_rate'] = args.learning_rate
        
        output_path = fine_tuner.train(args.data, args.output)
        print(f"✅ Training completed! Model saved to: {output_path}")
        return
    
    # No action specified
    print("ℹ️ No action specified. Use --help for usage information.")
    print("\nQuick start:")
    print("  python train_model.py --quick-train")

if __name__ == "__main__":
    main() 
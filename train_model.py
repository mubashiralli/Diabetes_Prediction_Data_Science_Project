#!/usr/bin/env python3
"""
Diabetes Prediction Model Training Script

This script provides a command-line interface for training and evaluating
the diabetes prediction model using the configured parameters.
"""

import pandas as pd
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import local modules
try:
    from utils import preprocess_diabetes_data, evaluate_model, print_model_summary
    from config import (
        DATA_FILE, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN,
        MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Make sure utils.py and config.py are in the same directory.")
    sys.exit(1)


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the diabetes dataset.
    
    Args:
        file_path (str): Path to the CSV data file
        
    Returns:
        tuple: Preprocessed features (X) and target (y)
    """
    try:
        # Load data
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} features.")
        
        # Preprocess data
        print("Preprocessing data...")
        df_processed = preprocess_diabetes_data(df)
        
        # Separate features and target
        X = df_processed.drop(columns=[TARGET_COLUMN])
        y = df_processed[TARGET_COLUMN]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
        
    except FileNotFoundError:
        print(f"Error: Data file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def train_model(X_train, y_train, **kwargs):
    """
    Train the decision tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional parameters for the classifier
        
    Returns:
        Trained DecisionTreeClassifier
    """
    print("Training Decision Tree Classifier...")
    
    # Create and configure the model
    model = DecisionTreeClassifier(
        max_depth=kwargs.get('max_depth', MAX_DEPTH),
        min_samples_split=kwargs.get('min_samples_split', MIN_SAMPLES_SPLIT),
        min_samples_leaf=kwargs.get('min_samples_leaf', MIN_SAMPLES_LEAF),
        random_state=RANDOM_STATE
    )
    
    # Train the model
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    return model


def main():
    """Main function to orchestrate the training pipeline."""
    parser = argparse.ArgumentParser(description='Train diabetes prediction model')
    parser.add_argument('--data-file', default=DATA_FILE,
                       help='Path to the diabetes dataset CSV file')
    parser.add_argument('--test-size', type=float, default=TEST_SIZE,
                       help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE,
                       help='Random state for reproducibility')
    parser.add_argument('--max-depth', type=int, default=MAX_DEPTH,
                       help='Maximum depth of the decision tree')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIABETES PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(args.data_file)
    
    # Split data
    print(f"Splitting data (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train, max_depth=args.max_depth)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Print results
    print_model_summary(results)
    
    print("\nTraining pipeline completed successfully!")
    return model, results


if __name__ == "__main__":
    trained_model, evaluation_results = main()

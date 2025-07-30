"""
Diabetes Prediction Model Utilities

This module contains utility functions for the diabetes prediction project,
including data preprocessing, model evaluation, and visualization helpers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def parse_categorical_data(data):
    """
    Convert categorical variables to numerical format.
    
    Args:
        data (str): Categorical value to convert
        
    Returns:
        int or str: 1 for positive indicators, 0 for negative indicators
    """
    if data in ['Yes', 'Positive', 'Female']:
        return 1
    elif data in ['No', 'Negative', 'Male']:
        return 0
    else:
        return data


def preprocess_diabetes_data(df):
    """
    Preprocess the diabetes dataset by cleaning and converting categorical variables.
    
    Args:
        df (pandas.DataFrame): Raw diabetes dataset
        
    Returns:
        pandas.DataFrame: Preprocessed dataset with numerical variables
    """
    # Remove duplicates
    df_clean = df.drop_duplicates()
    
    # Convert categorical to numerical
    df_numeric = df_clean.copy()
    for column in df_numeric.columns:
        df_numeric[column] = df_numeric[column].apply(parse_categorical_data)
    
    return df_numeric


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and return comprehensive metrics.
    
    Args:
        model: Trained machine learning model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create evaluation results
    results = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return results


def create_age_groups(age):
    """
    Create age groups for demographic analysis.
    
    Args:
        age (int): Patient age
        
    Returns:
        str: Age group label (e.g., 'G6-20', 'G21-35')
    """
    n = 20
    while True:
        if age <= n:
            return f'G{n-14}-{n}'
        n += 15


def plot_correlation_heatmap(df, figsize=(13, 10)):
    """
    Create and display a correlation heatmap for the dataset.
    
    Args:
        df (pandas.DataFrame): Numerical dataset
        figsize (tuple): Figure size for the plot
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap="viridis", fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def analyze_demographics(df, target_col='class'):
    """
    Analyze demographic patterns in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset with demographic information
        target_col (str): Name of the target variable column
    """
    # Gender analysis
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='Gender', hue=target_col)
    plt.title('Diabetes Distribution by Gender')
    
    # Age group analysis
    df_age = df.copy()
    df_age['Age_Group'] = df_age['Age'].apply(create_age_groups)
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df_age, x='Age_Group', hue=target_col)
    plt.title('Diabetes Distribution by Age Group')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def print_model_summary(results):
    """
    Print a formatted summary of model evaluation results.
    
    Args:
        results (dict): Results from evaluate_model function
    """
    print("=" * 50)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("=" * 50)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic Regression Model Training Script for Legal Document Classification
This script trains a Logistic Regression model on legal documents for classification.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_logistic_regression_model(data_path, output_dir, test_size=0.2, max_features=10000, 
                   ngram_range=(1, 2), C=1.0, solver='liblinear', max_iter=1000, 
                   class_weight='balanced', random_state=42, perform_grid_search=False):
    """
    Train a Logistic Regression model for text classification.
    
    Args:
        data_path (str): Path to the CSV file containing the dataset
        output_dir (str): Directory to save the output model and results
        test_size (float): Proportion of data to use for testing
        max_features (int): Maximum number of features for TF-IDF vectorizer
        ngram_range (tuple): Range of n-grams to consider for TF-IDF
        C (float): Regularization parameter
        solver (str): Algorithm to use in optimization problem
        max_iter (int): Maximum number of iterations
        class_weight (str): Weighting strategy for classes
        random_state (int): Random seed for reproducibility
        perform_grid_search (bool): Whether to perform grid search for hyperparameters
        
    Returns:
        tuple: (trained model, accuracy score)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Check and display class distribution
    print(f"Number of classes: {df['label'].nunique()}")
    print("Class distribution:")
    print(df['label'].value_counts())
    
    # Split data into train and test sets
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=test_size, random_state=random_state, stratify=df['label']
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create pipeline with TF-IDF and Logistic Regression
    print("Creating TF-IDF vectorizer and Logistic Regression pipeline...")
    print(f"Max features: {max_features}")
    print(f"N-gram range: {ngram_range}")
    print(f"C parameter: {C}")
    print(f"Solver: {solver}")
    print(f"Max iterations: {max_iter}")
    print(f"Class weight: {class_weight}")
    
    # Define the pipeline
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ('clf', LogisticRegression(
            C=C, 
            solver=solver, 
            max_iter=max_iter, 
            class_weight=class_weight, 
            random_state=random_state,
            multi_class='auto',
            n_jobs=-1
        ))
    ])
    
    # Perform grid search if requested
    if perform_grid_search:
        print("Performing grid search for hyperparameter tuning...")
        param_grid = {
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__C': [0.1, 1.0, 10.0],
            'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'clf__class_weight': ['balanced', None],
        }
        
        grid_search = GridSearchCV(text_clf, param_grid, cv=3, n_jobs=-1, verbose=1)
        
        # Train with grid search
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get best model
        text_clf = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Save grid search results
        pd.DataFrame(grid_search.cv_results_).to_csv(
            os.path.join(output_dir, 'grid_search_results.csv'), index=False
        )
    else:
        # Train the model
        print("Training the model...")
        start_time = time.time()
        text_clf.fit(X_train, y_train)
        training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    start_time = time.time()
    y_pred = text_clf.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate inference time per document
    avg_inference_time = (inference_time / len(X_test)) * 1000  # in milliseconds
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Average inference time per document: {avg_inference_time:.2f} ms")
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Save classification report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nTraining time: {training_time:.2f} seconds\n")
        f.write(f"Average inference time per document: {avg_inference_time:.2f} ms\n")
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(df['label'].unique()),
                yticklabels=sorted(df['label'].unique()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save confusion matrix
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    
    # Analyze feature importance - in logistic regression, coefficients represent feature importance
    print("\nAnalyzing feature importance...")
    vectorizer = text_clf.named_steps['tfidf']
    classifier = text_clf.named_steps['clf']
    
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    
    # Create a DataFrame with feature names and coefficients for each class
    classes = classifier.classes_
    coefficients = classifier.coef_
    
    # Plot top features with highest absolute coefficients
    plt.figure(figsize=(12, 8))
    
    # For multiclass, we plot the average absolute coefficient across all classes
    if len(classes) > 2:
        # Average absolute coefficient
        mean_abs_coef = np.abs(coefficients).mean(axis=0)
        top_indices = np.argsort(mean_abs_coef)[-20:]  # Top 20 features
        
        plt.barh(range(len(top_indices)), 
                mean_abs_coef[top_indices], 
                align='center')
        plt.yticks(range(len(top_indices)), 
                [feature_names[i] for i in top_indices])
        plt.gca().invert_yaxis()
        plt.title('Top 20 Average Feature Importances Across All Classes')
        plt.xlabel('Mean Absolute Coefficient')
    else:
        # For binary classification
        top_indices = np.argsort(coefficients[0])[-20:]  # Top 20 features
        plt.barh(range(len(top_indices)), 
                coefficients[0][top_indices], 
                align='center')
        plt.yticks(range(len(top_indices)), 
                [feature_names[i] for i in top_indices])
        plt.gca().invert_yaxis()
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Coefficient')
    
    feature_importance_path = os.path.join(output_dir, 'feature_importance.png')
    plt.tight_layout()
    plt.savefig(feature_importance_path)
    print(f"Feature importance plot saved to {feature_importance_path}")
    
    # Analyze top features per class
    top_features_per_class = []
    
    for i, class_name in enumerate(classes):
        if len(classes) <= 2 and i > 0:
            # For binary classification, we only need to analyze one class
            continue
        
        # Get top features for the current class
        class_coefficients = coefficients[i]
        top_feature_indices = np.argsort(class_coefficients)[-10:]  # Top 10 features
        
        top_features = []
        for idx in top_feature_indices[::-1]:  # Reverse to get highest first
            top_features.append({
                'feature': feature_names[idx],
                'coefficient': float(class_coefficients[idx])
            })
        
        # Print top features
        print(f"\nTop features for class '{class_name}':")
        for j, feature in enumerate(top_features, 1):
            print(f"  {j}. {feature['feature']} (coefficient: {feature['coefficient']:.1f})")
        
        # Save to collection
        top_features_per_class.append({
            'class': class_name,
            'features': top_features
        })
    
    # Save top features to CSV
    top_features_df = []
    for class_data in top_features_per_class:
        class_name = class_data['class']
        for feature in class_data['features']:
            top_features_df.append({
                'class': class_name,
                'feature': feature['feature'],
                'coefficient': feature['coefficient']
            })
    
    pd.DataFrame(top_features_df).to_csv(
        os.path.join(output_dir, 'top_features.csv'), index=False
    )
    
    # Save the trained model
    model_path = os.path.join(output_dir, 'logistic_regression_model.joblib')
    joblib.dump(text_clf, model_path)
    print(f"Model saved to {model_path}")
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to {output_dir}")
    
    return text_clf, accuracy

def main():
    parser = argparse.ArgumentParser(description='Train a Logistic Regression model for text classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./model_output', help='Directory to save the output model and results')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--max_features', type=int, default=10000, help='Maximum number of features for TF-IDF vectorizer')
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=2, help='Maximum n-gram size')
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--solver', type=str, default='liblinear', 
                        choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
                        help='Algorithm to use in optimization problem')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--class_weight', type=str, default='balanced', help='Weighting strategy for classes (balanced or None)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--grid_search', action='store_true', help='Perform grid search for hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Set class_weight to None if specified as 'none'
    if args.class_weight.lower() == 'none':
        args.class_weight = None
    
    # Train the model
    train_logistic_regression_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        random_state=args.random_state,
        perform_grid_search=args.grid_search
    )

if __name__ == '__main__':
    main()

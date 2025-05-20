#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SVM Model Inference Script for Legal Document Classification
This script loads a trained SVM model and allows predictions on new documents.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib

def load_model(model_path):
    """
    Load the trained SVM model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Pipeline: The loaded model pipeline
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model

def extract_text_from_json(json_file):
    """
    Extract text content from a JSON file containing a legal document.
    
    Args:
        json_file (str): Path to the JSON file
        
    Returns:
        str: Extracted document text
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract header and recitals which are the key parts for classification
    document_text = ""
    
    # Extract title/header
    if "title" in data and data["title"]:
        document_text += data["title"] + " "
    
    # Extract recitals if available
    if "recitals" in data and data["recitals"]:
        for recital in data["recitals"]:
            if "text" in recital:
                document_text += recital["text"] + " "
    
    # If no recitals, try to get content from body
    if not document_text and "body" in data:
        # This is a simplified extraction - adapt based on your JSON structure
        if isinstance(data["body"], str):
            document_text += data["body"]
        elif isinstance(data["body"], dict) and "text" in data["body"]:
            document_text += data["body"]["text"]
        elif isinstance(data["body"], list):
            for item in data["body"][:5]:  # Limit to first few items
                if isinstance(item, dict) and "text" in item:
                    document_text += item["text"] + " "
    
    return document_text.strip()

def predict_single_document(model, text, analyze=False):
    """
    Predict the class of a single document.
    
    Args:
        model (Pipeline): The trained model pipeline
        text (str): The document text
        analyze (bool): Whether to analyze feature importance
        
    Returns:
        dict: Prediction results including class and probabilities
    """
    # Start time
    start_time = time.time()
    
    # Make prediction
    prediction = model.predict([text])[0]
    
    # Get decision values (distances to hyperplanes)
    decision_values = model.decision_function([text])
    
    # Get classes
    classes = model.classes_
    
    # Convert distances to pseudo-probabilities using softmax
    exp_values = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    # Calculate inference time
    inference_time = (time.time() - start_time) * 1000  # in milliseconds
    
    # Create result dictionary
    result = {
        "prediction": prediction,
        "probabilities": {
            class_name: float(f"{prob:.4f}") 
            for class_name, prob in zip(classes, probabilities[0])
        },
        "inference_time_ms": inference_time
    }
    
    # Analyze feature importance if requested
    if analyze:
        # Get the vectorizer and classifier from the pipeline
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        # Vectorize the input text
        feature_vector = vectorizer.transform([text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get non-zero features
        non_zero_indices = feature_vector.nonzero()[1]
        
        # Get the corresponding feature names
        document_features = {feature_names[i]: feature_vector[0, i] for i in non_zero_indices}
        
        # Get coefficients for the predicted class
        predicted_class_idx = list(classes).index(prediction)
        coefficients = classifier.coef_[predicted_class_idx]
        
        # Calculate feature contributions
        feature_contributions = {}
        for feature_name, tfidf_value in document_features.items():
            try:
                feature_idx = list(feature_names).index(feature_name)
                contribution = coefficients[feature_idx] * tfidf_value
                feature_contributions[feature_name] = float(contribution)
            except ValueError:
                # Skip features not found in the trained model
                continue
        
        # Get top contributing features
        top_features = {k: v for k, v in sorted(
            feature_contributions.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )[:10]}
        
        result["feature_importance"] = top_features
    
    return result

def batch_predict(model, texts, batch_size=100):
    """
    Make predictions for a batch of texts to optimize memory usage.
    
    Args:
        model (Pipeline): The trained model pipeline
        texts (list): List of document texts
        batch_size (int): Batch size for processing
        
    Returns:
        list: List of prediction results
    """
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Make predictions for the batch
        start_time = time.time()
        batch_predictions = model.predict(batch_texts)
        batch_decisions = model.decision_function(batch_texts)
        
        # Get classes
        classes = model.classes_
        
        # Convert to probabilities
        exp_values = np.exp(batch_decisions - np.max(batch_decisions, axis=1, keepdims=True))
        batch_probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # Calculate inference time
        batch_time = time.time() - start_time
        avg_inference_time = (batch_time / len(batch_texts)) * 1000  # in milliseconds
        
        # Create result dictionaries for the batch
        for j, (prediction, probs) in enumerate(zip(batch_predictions, batch_probabilities)):
            result = {
                "index": i + j,
                "prediction": prediction,
                "probabilities": {
                    class_name: float(f"{prob:.2f}") 
                    for class_name, prob in zip(classes, probs)
                }
            }
            results.append(result)
    
    return results, avg_inference_time

def main():
    parser = argparse.ArgumentParser(description='Make predictions with a trained SVM model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    
    # Input options (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help='Text to classify')
    group.add_argument('--json_file', type=str, help='Path to a JSON file containing a document')
    group.add_argument('--text_file', type=str, help='Path to a text file containing a document')
    group.add_argument('--csv_file', type=str, help='Path to a CSV file containing documents to classify')
    
    # Additional options
    parser.add_argument('--output_path', type=str, help='Path to save the predictions')
    parser.add_argument('--analyze', action='store_true', help='Analyze feature importance for the prediction')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing multiple documents')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Make predictions based on input type
    if args.text:
        # Single text prediction
        result = predict_single_document(model, args.text, args.analyze)
        print(f"\nPrediction: {result['prediction']}")
        print("Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
        if args.analyze and 'feature_importance' in result:
            print("\nFeature importance analysis:")
            for feature, importance in result['feature_importance'].items():
                print(f"  {feature}: {importance:.4f}")
        
        # Save to file if requested
        if args.output_path:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output_path}")
    
    elif args.json_file:
        # Prediction for a JSON document
        text = extract_text_from_json(args.json_file)
        result = predict_single_document(model, text, args.analyze)
        
        print(f"\nPrediction: {result['prediction']}")
        print("Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
            
        # Try to get the true label from the JSON file
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'category' in data:
                    print(f"True label: {data['category']}")
        except:
            pass
            
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
        if args.analyze and 'feature_importance' in result:
            print("\nFeature importance analysis:")
            for feature, importance in result['feature_importance'].items():
                print(f"  {feature}: {importance:.4f}")
        
        # Save to file if requested
        if args.output_path:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output_path}")
    
    elif args.text_file:
        # Prediction for a text file
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = predict_single_document(model, text, args.analyze)
        
        print(f"\nPrediction: {result['prediction']}")
        print("Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
        if args.analyze and 'feature_importance' in result:
            print("\nFeature importance analysis:")
            for feature, importance in result['feature_importance'].items():
                print(f"  {feature}: {importance:.4f}")
        
        # Save to file if requested
        if args.output_path:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output_path}")
    
    elif args.csv_file:
        # Predictions for multiple documents from a CSV file
        df = pd.read_csv(args.csv_file)
        print(f"Loading documents from {args.csv_file}...")
        
        # Check if the dataframe has the required columns
        if 'text' not in df.columns:
            print("Error: CSV file must contain a 'text' column")
            return
        
        has_labels = 'label' in df.columns
        
        # Make predictions
        texts = df['text'].tolist()
        print(f"Making predictions for {len(texts)} documents...")
        
        # Process in batches
        results, avg_inference_time = batch_predict(model, texts, args.batch_size)
        print(f"Predictions completed in {len(texts) * avg_inference_time / 1000:.2f} seconds")
        print(f"Average inference time: {avg_inference_time:.2f} ms per document")
        
        # Add true labels if available
        if has_labels:
            true_labels = df['label'].tolist()
            for i, result in enumerate(results):
                result['true_label'] = true_labels[i]
            
            # Calculate accuracy
            correct = sum(1 for res in results if res['prediction'] == res['true_label'])
            accuracy = correct / len(results)
            print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
        
        # Save to file
        if args.output_path:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_path}")
        else:
            # Print first few results
            for result in results[:5]:
                print(f"Document {result['index']}: {result['prediction']}")

if __name__ == '__main__':
    main()

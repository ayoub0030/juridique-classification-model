import joblib
import pandas as pd
import numpy as np
import argparse
import os
import time
import json
from tqdm import tqdm

def load_model(model_path):
    """Load the trained Random Forest model"""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model

def predict_single_document(model, text):
    """Predict the class for a single document"""
    start_time = time.time()
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    
    # Get class names
    classes = model.named_steps['clf'].classes_
    
    # Create dictionary of class probabilities
    prob_dict = {classes[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    inference_time = time.time() - start_time
    
    result = {
        'prediction': prediction,
        'probabilities': prob_dict,
        'inference_time_ms': inference_time * 1000
    }
    
    return result

def predict_from_text_file(model, file_path):
    """Predict the class for a document from a text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return predict_single_document(model, text)

def predict_from_json_file(model, file_path):
    """Predict the class for a document from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # For EU legal documents, combine header, recitals, and main body
    text_parts = []
    if 'header' in data:
        text_parts.append(data['header'])
    if 'recitals' in data:
        text_parts.append(data['recitals'])
    if 'main_body' in data and isinstance(data['main_body'], list):
        text_parts.append(' '.join(data['main_body']))
    
    text = '\n\n'.join(text_parts)
    
    result = predict_single_document(model, text)
    
    # Add document ID if available
    if 'celex_id' in data:
        result['document_id'] = data['celex_id']
        
    # Add true label if available
    if 'type' in data:
        result['true_label'] = data['type']
    
    return result

def predict_from_csv(model, file_path, text_column='text', output_path=None):
    """Predict classes for documents in a CSV file"""
    print(f"Loading documents from {file_path}...")
    df = pd.read_csv(file_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV file")
    
    results = []
    
    print(f"Making predictions for {len(df)} documents...")
    start_time = time.time()
    
    # Process in batches for memory efficiency
    batch_size = 100
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        texts = batch[text_column].tolist()
        
        # Get predictions and probabilities
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        
        # Get class names
        classes = model.named_steps['clf'].classes_
        
        # Create result for each document
        for j, (pred, probs) in enumerate(zip(predictions, probabilities)):
            idx = i + j
            result = {
                'index': idx,
                'prediction': pred,
                'probabilities': {classes[k]: float(prob) for k, prob in enumerate(probs)}
            }
            
            # Add true label if available
            if 'label' in df.columns:
                result['true_label'] = df.iloc[idx]['label']
            
            # Add document ID if available
            if 'celex_id' in df.columns:
                result['document_id'] = df.iloc[idx]['celex_id']
                
            results.append(result)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(df) * 1000  # in milliseconds
    
    print(f"Predictions completed in {total_time:.2f} seconds")
    print(f"Average inference time: {avg_time:.2f} ms per document")
    
    # Calculate accuracy if true labels are available
    if 'label' in df.columns:
        correct = sum(1 for r in results if r['prediction'] == r['true_label'])
        accuracy = correct / len(results)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    
    # Save results if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results

def analyze_feature_importance(model, text, top_n=10):
    """Analyze which features contributed most to the prediction"""
    # Get the prediction
    prediction = model.predict([text])[0]
    
    # Get the vectorizer and feature names
    vectorizer = model.named_steps['tfidf']
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the document vector
    X = vectorizer.transform([text])
    
    # Get the feature importance from the random forest
    clf = model.named_steps['clf']
    feature_importances = clf.feature_importances_
    
    # Get the features present in this document
    non_zero_features = X.nonzero()[1]
    
    # Get the importance of features present in this document
    document_features = [(feature_names[i], feature_importances[i], X[0, i]) 
                         for i in non_zero_features]
    
    # Sort by importance
    document_features.sort(key=lambda x: x[1] * x[2], reverse=True)
    
    # Take top N features
    top_features = document_features[:top_n]
    
    # Create a result dictionary
    result = {
        'prediction': prediction,
        'top_features': [{'feature': f, 'importance': float(i), 'value': float(v)} 
                         for f, i, v in top_features]
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Make predictions with a trained Random Forest model")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.joblib)')
    
    # Input options (choose one)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str,
                      help='Text string to classify')
    group.add_argument('--text_file', type=str,
                      help='Path to a text file to classify')
    group.add_argument('--json_file', type=str,
                      help='Path to a JSON file to classify')
    group.add_argument('--csv_file', type=str,
                      help='Path to a CSV file with documents to classify')
    
    # Additional options
    parser.add_argument('--text_column', type=str, default='text',
                        help='Column name for text in CSV file')
    parser.add_argument('--output_path', type=str,
                        help='Path to save results (for CSV mode)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze feature importance for prediction')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Make predictions based on input type
    if args.text:
        result = predict_single_document(model, args.text)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {prob:.4f}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
        if args.analyze:
            analysis = analyze_feature_importance(model, args.text)
            print("\nFeature importance analysis:")
            for feature in analysis['top_features']:
                print(f"  {feature['feature']}: {feature['importance']:.4f}")
        
    elif args.text_file:
        result = predict_from_text_file(model, args.text_file)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {prob:.4f}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
        if args.analyze:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            analysis = analyze_feature_importance(model, text)
            print("\nFeature importance analysis:")
            for feature in analysis['top_features']:
                print(f"  {feature['feature']}: {feature['importance']:.4f}")
        
    elif args.json_file:
        result = predict_from_json_file(model, args.json_file)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {prob:.4f}")
        if 'true_label' in result:
            print(f"True label: {result['true_label']}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
        if args.analyze:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text_parts = []
            if 'header' in data:
                text_parts.append(data['header'])
            if 'recitals' in data:
                text_parts.append(data['recitals'])
            if 'main_body' in data and isinstance(data['main_body'], list):
                text_parts.append(' '.join(data['main_body']))
            
            text = '\n\n'.join(text_parts)
            
            analysis = analyze_feature_importance(model, text)
            print("\nFeature importance analysis:")
            for feature in analysis['top_features']:
                print(f"  {feature['feature']}: {feature['importance']:.4f}")
        
    elif args.csv_file:
        results = predict_from_csv(
            model, 
            args.csv_file, 
            text_column=args.text_column, 
            output_path=args.output_path
        )

if __name__ == "__main__":
    main()

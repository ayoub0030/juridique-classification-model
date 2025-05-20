import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

def train_random_forest_model(data_path, output_dir, test_size=0.2, max_features=10000, 
                             ngram_range=(1, 2), random_state=42, n_estimators=100,
                             max_depth=None, min_samples_split=2, n_jobs=-1):
    """
    Train a Random Forest model on the legal document dataset
    
    Args:
        data_path: Path to CSV file containing 'text' and 'label' columns
        output_dir: Directory to save model and results
        test_size: Proportion of data to use for testing
        max_features: Maximum number of features for TF-IDF
        ngram_range: Range of n-grams to use
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        min_samples_split: Minimum samples required to split an internal node
        n_jobs: Number of jobs to run in parallel (-1 means using all processors)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Print dataset information
    print(f"Dataset shape: {data.shape}")
    print(f"Number of classes: {data['label'].nunique()}")
    print(f"Class distribution:\n{data['label'].value_counts()}")
    
    # Split the data
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=test_size, 
        random_state=random_state, stratify=data['label']
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create a pipeline with vectorizer and classifier
    print("Creating TF-IDF vectorizer and Random Forest pipeline...")
    print(f"Max features: {max_features}")
    print(f"N-gram range: {ngram_range}")
    print(f"Number of trees: {n_estimators}")
    if max_depth:
        print(f"Max depth: {max_depth}")
    else:
        print("Max depth: None (unlimited)")
    print(f"Min samples split: {min_samples_split}")
    
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=5  # Ignore terms that appear in fewer than 5 documents
        )),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1
        ))
    ])
    
    # Train the model
    print("Training the model...")
    start_time = time.time()
    text_clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    print("Evaluating on test set...")
    start_time = time.time()
    y_pred = text_clf.predict(X_test)
    inference_time = time.time() - start_time
    average_inference_time = inference_time / len(X_test)
    
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Average inference time per document: {average_inference_time*1000:.2f} ms")
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Detailed classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    labels = sorted(data['label'].unique())
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_names = text_clf.named_steps['tfidf'].get_feature_names_out()
    
    # Extract classifier from pipeline
    rf_classifier = text_clf.named_steps['clf']
    
    # Calculate feature importance
    importances = rf_classifier.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Get top features for each class
    class_names = rf_classifier.classes_
    print("Top 10 most important features:")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.title("Feature importance")
    plt.bar(range(20), importances[indices[:20]], align="center")
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    
    # Save feature importance plot
    feature_importance_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(feature_importance_path)
    print(f"Feature importance plot saved to {feature_importance_path}")
    
    # Print top features for each class
    top_features = []
    for i, class_name in enumerate(class_names):
        print(f"\nTop features for class '{class_name}':")
        # For each tree, collect the feature importances for this class
        class_feature_importance = np.zeros(len(feature_names))
        
        # Count the number of times each feature appears in a decision
        # that leads to this class across all trees
        for tree in rf_classifier.estimators_:
            # Get node indices that predict this class
            tree_leaf_values = tree.tree_.value  # Shape: [n_nodes, 1, n_classes]
            tree_class_distribution = tree_leaf_values.reshape(-1, len(class_names))
            # Find nodes where this class has highest probability
            class_dominant_nodes = np.argmax(tree_class_distribution, axis=1) == i
            
            # Get feature used at each node
            feature_indices = tree.tree_.feature
            
            # Count feature usage in paths to this class
            for feature_idx, is_dominant in zip(feature_indices, class_dominant_nodes):
                if feature_idx >= 0 and is_dominant:  # -1 indicates leaf node
                    class_feature_importance[feature_idx] += 1
        
        # Get indices of top features for this class
        class_top_indices = np.argsort(class_feature_importance)[::-1][:10]
        
        # Print top 10 features for this class
        for j, idx in enumerate(class_top_indices):
            if class_feature_importance[idx] > 0:
                print(f"  {j+1}. {feature_names[idx]} (importance: {class_feature_importance[idx]})")
                top_features.append((class_name, feature_names[idx], float(class_feature_importance[idx])))
    
    # Save the model
    model_path = os.path.join(output_dir, 'random_forest_model.joblib')
    joblib.dump(text_clf, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names
    feature_path = os.path.join(output_dir, 'feature_names.joblib')
    joblib.dump(feature_names, feature_path)
    
    # Save top features
    top_features_df = pd.DataFrame(top_features, columns=['class', 'feature', 'importance'])
    top_features_path = os.path.join(output_dir, 'top_features.csv')
    top_features_df.to_csv(top_features_path, index=False)
    print(f"Top features saved to {top_features_path}")
    
    # Save metadata
    metadata = {
        'accuracy': accuracy,
        'training_time': train_time,
        'average_inference_time': average_inference_time,
        'parameters': {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'n_jobs': n_jobs
        },
        'classes': list(rf_classifier.classes_)
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df = pd.DataFrame(class_report).transpose()
    report_df.to_csv(report_path)
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to {output_dir}")
    
    return text_clf, accuracy

def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest model for legal document classification")
    parser.add_argument('--data_path', type=str, default='../text-mining-v2/full_bert_dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='./model_output',
                        help='Directory to save model and results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Maximum number of features for TF-IDF')
    parser.add_argument('--ngram_min', type=int, default=1,
                        help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=2,
                        help='Maximum n-gram size')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum depth of the trees (None for unlimited)')
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='Minimum number of samples required to split an internal node')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of jobs to run in parallel (-1 means using all processors)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create ngram_range tuple
    ngram_range = (args.ngram_min, args.ngram_max)
    
    # Train model
    train_random_forest_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        max_features=args.max_features,
        ngram_range=ngram_range,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        n_jobs=args.n_jobs
    )

if __name__ == "__main__":
    main()

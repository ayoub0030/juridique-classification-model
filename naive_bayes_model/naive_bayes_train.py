import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

def train_naive_bayes_model(data_path, output_dir, test_size=0.2, max_features=20000, 
                            ngram_range=(1, 2), random_state=42, alpha=0.1):
    """
    Train a Naive Bayes model on the legal document dataset
    
    Args:
        data_path: Path to CSV file containing 'text' and 'label' columns
        output_dir: Directory to save model and results
        test_size: Proportion of data to use for testing
        max_features: Maximum number of features for TF-IDF
        ngram_range: Range of n-grams to use
        random_state: Random seed for reproducibility
        alpha: Smoothing parameter for Naive Bayes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
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
    print("Creating TF-IDF vectorizer and Naive Bayes pipeline...")
    print(f"Max features: {max_features}")
    print(f"N-gram range: {ngram_range}")
    print(f"Alpha: {alpha}")
    
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=5  # Ignore terms that appear in fewer than 5 documents
        )),
        ('clf', MultinomialNB(alpha=alpha))
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
    
    # Save the model
    model_path = os.path.join(output_dir, 'naive_bayes_model.joblib')
    joblib.dump(text_clf, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names (for model explanation)
    feature_names = text_clf.named_steps['tfidf'].get_feature_names_out()
    feature_path = os.path.join(output_dir, 'feature_names.joblib')
    joblib.dump(feature_names, feature_path)
    
    # Save metadata
    metadata = {
        'accuracy': accuracy,
        'training_time': train_time,
        'average_inference_time': average_inference_time,
        'parameters': {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'alpha': alpha
        },
        'classes': list(text_clf.named_steps['clf'].classes_)
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
    parser = argparse.ArgumentParser(description="Train a Naive Bayes model for legal document classification")
    parser.add_argument('--data_path', type=str, default='../text-mining-v2/full_bert_dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='./model_output',
                        help='Directory to save model and results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--max_features', type=int, default=20000,
                        help='Maximum number of features for TF-IDF')
    parser.add_argument('--ngram_min', type=int, default=1,
                        help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=2,
                        help='Maximum n-gram size')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Smoothing parameter for Naive Bayes')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create ngram_range tuple
    ngram_range = (args.ngram_min, args.ngram_max)
    
    # Train model
    train_naive_bayes_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        max_features=args.max_features,
        ngram_range=ngram_range,
        random_state=args.random_state,
        alpha=args.alpha
    )

if __name__ == "__main__":
    main()

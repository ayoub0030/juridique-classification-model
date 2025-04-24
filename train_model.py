import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(excel_file_path):
    print(f"Loading data from {excel_file_path}...")
    df = pd.read_excel(excel_file_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    return df

# Preprocess the data
def preprocess_data(df):
    print("Preprocessing data...")
    
    # Check for missing values in key columns
    missing_concepts = df['concepts'].isna().sum()
    missing_type = df['type'].isna().sum()
    
    if missing_concepts > 0:
        print(f"Warning: {missing_concepts} rows have missing 'concepts' values.")
    if missing_type > 0:
        print(f"Warning: {missing_type} rows have missing 'type' values.")
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['concepts', 'type'])
    
    # Encode the target variable (type)
    label_encoder = LabelEncoder()
    df['type_encoded'] = label_encoder.fit_transform(df['type'])
    
    # Map labels for later reference
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping (type -> encoded value):")
    for label, code in label_mapping.items():
        print(f"  {label} -> {code}")
    
    return df, label_encoder

# Split the data into training and testing sets
def split_data(df, test_size=0.2, random_state=42):
    X = df['concepts']
    y = df['type_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

# Feature extraction
def extract_features(X_train, X_test):
    # Convert concepts to bag-of-words features
    # We'll treat each concept number as a "word"
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    print(f"Number of features (unique concept numbers): {len(vectorizer.get_feature_names_out())}")
    print(f"Feature names (first 10): {vectorizer.get_feature_names_out()[:10]}")
    
    return X_train_features, X_test_features, vectorizer

# Train the model
def train_model(X_train_features, y_train):
    print("Training the model...")
    model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
    model.fit(X_train_features, y_train)
    print("Model training completed.")
    return model

# Evaluate the model
def evaluate_model(model, X_test_features, y_test, label_encoder):
    print("Evaluating the model...")
    
    # Make predictions
    y_pred = model.predict(X_test_features)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    return accuracy, report

# Save the model and vectorizer for later use
def save_model(model, vectorizer, label_encoder):
    print("Saving the model...")
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': label_encoder
    }
    
    with open('concept_classifier_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'concept_classifier_model.pkl'")

# Function to make predictions with the trained model
def predict_type(concept_numbers, model, vectorizer, label_encoder):
    """
    Predict the document type based on concept numbers.
    
    Args:
        concept_numbers (str): Comma-separated list of concept numbers.
        model: Trained classifier.
        vectorizer: Fitted CountVectorizer.
        label_encoder: Fitted LabelEncoder.
    
    Returns:
        str: Predicted document type.
    """
    # Convert concept numbers to features
    concept_features = vectorizer.transform([concept_numbers])
    
    # Make prediction
    predicted_encoded = model.predict(concept_features)[0]
    
    # Decode prediction
    predicted_type = label_encoder.inverse_transform([predicted_encoded])[0]
    
    return predicted_type

# Main function
def main():
    # Load data from Excel file
    excel_file_path = 'dataset_output.xlsx'
    df = load_data(excel_file_path)
    
    # Preprocess data
    df, label_encoder = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Extract features
    X_train_features, X_test_features, vectorizer = extract_features(X_train, X_test)
    
    # Train model
    model = train_model(X_train_features, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test_features, y_test, label_encoder)
    
    # Save model
    save_model(model, vectorizer, label_encoder)
    
    # Test prediction functionality
    print("\nTesting prediction functionality with sample concepts:")
    
    # Try a few examples from the dataset
    examples = df.head(3)[['concepts', 'type']].values
    for concepts, true_type in examples:
        predicted_type = predict_type(concepts, model, vectorizer, label_encoder)
        print(f"Concepts: {concepts}")
        print(f"True type: {true_type}")
        print(f"Predicted type: {predicted_type}")
        print("---")
    
    # Let user try a custom prediction
    print("\nYou can now use the following code to predict document type from concept numbers:")
    print("Example usage:\n")
    print("from train_model import predict_type")
    print("import pickle")
    print("")
    print("# Load the saved model")
    print("with open('concept_classifier_model.pkl', 'rb') as f:")
    print("    model_data = pickle.load(f)")
    print("")
    print("# Extract model components")
    print("model = model_data['model']")
    print("vectorizer = model_data['vectorizer']")
    print("label_encoder = model_data['label_encoder']")
    print("")
    print("# Make a prediction")
    print("concepts = '1086, 1196, 2002'  # Example concepts")
    print("predicted_type = predict_type(concepts, model, vectorizer, label_encoder)")
    print("print(f'Predicted document type: {predicted_type}')")

if __name__ == "__main__":
    main()

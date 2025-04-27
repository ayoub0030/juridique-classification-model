import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Define a tokenizer function at the global scope for pickling
def concept_tokenizer(text):
    return text.split(', ')

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

# Feature extraction
def extract_features(X):
    # Convert concepts to bag-of-words features
    # We'll treat each concept number as a "word"
    vectorizer = CountVectorizer(tokenizer=concept_tokenizer)
    X_features = vectorizer.fit_transform(X)
    
    print(f"Number of features (unique concept numbers): {len(vectorizer.get_feature_names_out())}")
    print(f"Feature names (first 10): {vectorizer.get_feature_names_out()[:10]}")
    
    return X_features, vectorizer

# Train the model
def train_model(X_features, y):
    print("Training the model...")
    model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
    model.fit(X_features, y)
    print("Model training completed.")
    return model

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
    excel_file_path = 'expanded_dataset.xlsx'
    df = load_data(excel_file_path)
    
    # Preprocess data
    df, label_encoder = preprocess_data(df)
    
    # Use all data for training (no train/test split)
    X = df['concepts']
    y = df['type_encoded']
    print(f"Using all {len(X)} examples for training")
    
    # Extract features
    X_features, vectorizer = extract_features(X)
    
    # Train model using all data
    model = train_model(X_features, y)
    
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

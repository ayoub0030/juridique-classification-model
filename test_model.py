import pickle
import sys
from train_model import concept_tokenizer, predict_type  # We need to import the tokenizer for pickle to work

def load_model(model_path="concept_classifier_model.pkl"):
    """Load the trained model from a file."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model components
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        label_encoder = model_data['label_encoder']
            
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def classify_concepts(concepts, model, vectorizer, label_encoder):
    """Classify a set of concepts and return the predicted document type."""
    try:
        predicted_type = predict_type(concepts, model, vectorizer, label_encoder)
        
        # Get probabilities for each class
        concept_features = vectorizer.transform([concepts])
        probabilities = model.predict_proba(concept_features)[0]
        
        # Find the highest probability and its corresponding class
        max_prob = 0
        max_class = None
        
        for i, class_name in enumerate(label_encoder.classes_):
            class_idx = list(model.classes_).index(i) if i in model.classes_ else -1
            
            if class_idx >= 0 and probabilities[class_idx] > max_prob:
                max_prob = probabilities[class_idx]
                max_class = class_name
        
        return predicted_type, max_prob * 100
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0

def main():
    # Check if concepts are provided as command line arguments
    if len(sys.argv) < 2:
        print("Please provide concept numbers as arguments.")
        print("Example: python test_model.py 1086 1196 2002")
        return
    
    # Extract concepts from command line arguments
    concepts = ", ".join(sys.argv[1:])
    
    # Load the model
    model, vectorizer, label_encoder = load_model()
    
    if model is None:
        return
    
    # Classify the concepts
    document_type, confidence = classify_concepts(concepts, model, vectorizer, label_encoder)
    
    # Print the result
    print(f"Concepts: {concepts}")
    print(f"Document Type: {document_type}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()

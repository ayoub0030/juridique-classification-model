import joblib
import time
import os

class NaiveBayesLegalClassifier:
    """
    Helper class for legal document classification using Naive Bayes
    """
    def __init__(self, model_dir='./model'):
        """
        Initialize the classifier with a trained model
        
        Args:
            model_dir: Directory containing the model files
        """
        model_path = os.path.join(model_dir, 'naive_bayes_model.joblib')
        metadata_path = os.path.join(model_dir, 'model_metadata.joblib')
        
        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path)
        self.classes = self.metadata['classes']
    
    def predict(self, text):
        """
        Predict the class for a single document
        
        Args:
            text: Document text to classify
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Make prediction
        prediction = self.model.predict([text])[0]
        
        # Get probabilities
        probabilities = self.model.predict_proba([text])[0]
        prob_dict = {self.classes[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        inference_time = time.time() - start_time
        
        return {
            'prediction': prediction,
            'probabilities': prob_dict,
            'inference_time_ms': inference_time * 1000
        }
    
    def get_model_info(self):
        """
        Get information about the model
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'accuracy': self.metadata.get('accuracy', 'Unknown'),
            'classes': self.classes,
            'parameters': self.metadata.get('parameters', {}),
            'average_inference_time': self.metadata.get('average_inference_time', 0) * 1000
        }

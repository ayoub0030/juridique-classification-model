import joblib
import os
import json
import shutil
import time
import argparse

def prepare_model_for_web_deployment(model_dir, output_dir):
    """
    Prepares the trained Naive Bayes model for web deployment by:
    1. Creating a deployment-ready package with necessary files
    2. Generating a helper function for easy loading and prediction
    3. Creating a simple prediction API format example
    
    Args:
        model_dir: Directory containing the trained model and metadata
        output_dir: Directory to save the deployment package
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    model_path = os.path.join(model_dir, 'naive_bayes_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load model to verify it works
    print(f"Loading model from {model_path} to verify...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Create model directory in the output directory
    model_output_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Copy model files
    print(f"Copying model files to {model_output_dir}...")
    files_to_copy = [
        'naive_bayes_model.joblib',
        'model_metadata.joblib',
        'feature_names.joblib',
        'classification_report.csv'
    ]
    
    for file in files_to_copy:
        src_path = os.path.join(model_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(model_output_dir, file))
            print(f"Copied {file}")
        else:
            print(f"Warning: {file} not found in {model_dir}, skipping")
    
    # Copy confusion matrix (optional)
    confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        shutil.copy2(confusion_matrix_path, os.path.join(output_dir, 'confusion_matrix.png'))
        print("Copied confusion matrix image")
    
    # Create model helper module
    helper_path = os.path.join(output_dir, 'model_helper.py')
    with open(helper_path, 'w', encoding='utf-8') as f:
        f.write("""import joblib
import time
import os

class NaiveBayesLegalClassifier:
    \"\"\"
    Helper class for legal document classification using Naive Bayes
    \"\"\"
    def __init__(self, model_dir='./model'):
        \"\"\"
        Initialize the classifier with a trained model
        
        Args:
            model_dir: Directory containing the model files
        \"\"\"
        model_path = os.path.join(model_dir, 'naive_bayes_model.joblib')
        metadata_path = os.path.join(model_dir, 'model_metadata.joblib')
        
        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path)
        self.classes = self.metadata['classes']
    
    def predict(self, text):
        \"\"\"
        Predict the class for a single document
        
        Args:
            text: Document text to classify
            
        Returns:
            Dictionary with prediction results
        \"\"\"
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
        \"\"\"
        Get information about the model
        
        Returns:
            Dictionary with model metadata
        \"\"\"
        return {
            'accuracy': self.metadata.get('accuracy', 'Unknown'),
            'classes': self.classes,
            'parameters': self.metadata.get('parameters', {}),
            'average_inference_time': self.metadata.get('average_inference_time', 0) * 1000
        }
""")
    print(f"Created model helper module at {helper_path}")
    
    # Create minimal usage example
    example_path = os.path.join(output_dir, 'example_usage.py')
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write("""from model_helper import NaiveBayesLegalClassifier

# Initialize the classifier
classifier = NaiveBayesLegalClassifier(model_dir='./model')

# Get model info
model_info = classifier.get_model_info()
print("Model Information:")
print(f"  Accuracy: {model_info['accuracy']:.4f}")
print(f"  Classes: {model_info['classes']}")
print(f"  Average inference time: {model_info['average_inference_time']:.2f} ms")

# Example document text
example_text = \"\"\"
REGULATION (EU) No 1025/2012 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
of 25 October 2012
on European standardisation, amending Council Directives 89/686/EEC and 93/15/EEC and Directives 94/9/EC,
94/25/EC, 95/16/EC, 97/23/EC, 98/34/EC, 2004/22/EC, 2007/23/EC, 2009/23/EC and 2009/105/EC of the
European Parliament and of the Council and repealing Council Decision 87/95/EEC and Decision
No 1673/2006/EC of the European Parliament and of the Council
\"\"\"

# Make a prediction
result = classifier.predict(example_text)

print("\\nPrediction Results:")
print(f"  Predicted class: {result['prediction']}")
print("  Class probabilities:")
for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"    {cls}: {prob:.4f}")
print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
""")
    print(f"Created usage example at {example_path}")
    
    # Create Flask API example for web deployment
    flask_example_path = os.path.join(output_dir, 'flask_api_example.py')
    with open(flask_example_path, 'w', encoding='utf-8') as f:
        f.write("""from flask import Flask, request, jsonify
from model_helper import NaiveBayesLegalClassifier
import os

app = Flask(__name__)

# Initialize the classifier
MODEL_DIR = os.environ.get('MODEL_DIR', './model')
classifier = NaiveBayesLegalClassifier(model_dir=MODEL_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()
    
    # Check if text is provided
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get the text
    text = data['text']
    
    # Make prediction
    result = classifier.predict(text)
    
    return jsonify(result)

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify(classifier.get_model_info())

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Legal Document Classifier API',
        'status': 'active',
        'endpoints': [
            {'path': '/', 'method': 'GET', 'description': 'Service information'},
            {'path': '/predict', 'method': 'POST', 'description': 'Classify a document', 'body': {'text': 'string'}},
            {'path': '/model-info', 'method': 'GET', 'description': 'Get model information'}
        ]
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
""")
    print(f"Created Flask API example at {flask_example_path}")
    
    # Create requirements.txt
    requirements_path = os.path.join(output_dir, 'requirements.txt')
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write("""scikit-learn==1.2.2
pandas==2.0.3
numpy==1.24.3
joblib==1.3.1
flask==2.3.3
gunicorn==21.2.0
""")
    print(f"Created requirements.txt at {requirements_path}")
    
    # Create README.md
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("""# Legal Document Classifier - Web Deployment

This package contains everything needed to deploy the Naive Bayes legal document classifier as a web service.

## Files Included

- `model/` - Directory containing the trained model and metadata
- `model_helper.py` - Helper class for using the model
- `example_usage.py` - Example script for using the model
- `flask_api_example.py` - Example Flask API for web deployment
- `requirements.txt` - Required packages for deployment

## Quick Start

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the example:
   ```
   python example_usage.py
   ```

3. Start the API server:
   ```
   python flask_api_example.py
   ```

## API Endpoints

- `GET /` - Service information
- `GET /model-info` - Get model information (accuracy, classes, etc.)
- `POST /predict` - Classify a document
  - Request body: `{"text": "document text here"}`
  - Response: `{"prediction": "class", "probabilities": {"class1": 0.8, "class2": 0.2}, "inference_time_ms": 2.5}`

## Deployment Options

### Local Server
Run the Flask API locally:
```
python flask_api_example.py
```

### Docker
Create a Dockerfile:
```
FROM python:3.10-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "flask_api_example:app"]
```

Build and run:
```
docker build -t legal-classifier .
docker run -p 5000:5000 legal-classifier
```

### Cloud Deployment
This package is ready to deploy to services like:
- Heroku
- Google Cloud Run
- AWS Elastic Beanstalk

## Performance

- Accuracy: 93.92%
- Average inference time: ~3.3ms per document
- Memory usage: Low (suitable for small instances)
""")
    print(f"Created README.md at {readme_path}")
    
    print("\nDeployment package created successfully!")
    print(f"All files are saved to {output_dir}")
    print("\nTo use in a web application:")
    print("1. Copy the files to your web app project")
    print("2. Import NaiveBayesLegalClassifier from model_helper.py")
    print("3. Use the provided Flask API example or integrate with your own web framework")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Prepare Naive Bayes model for web deployment")
    parser.add_argument('--model_dir', type=str, default='./model_output',
                        help='Directory containing the trained model and metadata')
    parser.add_argument('--output_dir', type=str, default='./web_deployment',
                        help='Directory to save the deployment package')
    
    args = parser.parse_args()
    
    prepare_model_for_web_deployment(args.model_dir, args.output_dir)

if __name__ == "__main__":
    main()

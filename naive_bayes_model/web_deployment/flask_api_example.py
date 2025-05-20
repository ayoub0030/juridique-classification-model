from flask import Flask, request, jsonify
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

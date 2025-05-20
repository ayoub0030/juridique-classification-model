# Legal Document Classifier - Web Deployment

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

# Legal Document Classification Project

This project implements a machine learning model to classify legal documents based on their concept numbers. The model predicts the type of document (e.g., Decision, Regulation, Directive) using the associated concept codes.

## Project Structure

- `demo.py`: Script to convert JSON legal documents to Excel format
- `train_model.py`: Script to train and evaluate the classification model
- `dataset_folder/`: Contains JSON legal document files
- `dataset_output.xlsx`: Processed dataset in Excel format
- `classification_dataset_300.csv`: Expanded dataset for model training
- `concept_classifier_model.pkl`: Trained model saved for future use

## Features

- Data conversion from JSON to structured Excel format
- Text classification using scikit-learn
- Feature extraction from concept numbers
- Model evaluation with accuracy metrics and confusion matrix
- Prediction functionality for new documents

## How to Use

### Converting JSON to Excel

```python
python demo.py
```

This will process all JSON files in the `dataset_folder` and create an Excel file with structured data.

### Training the Model

```python
python train_model.py
```

This will:
1. Load the data from the Excel file
2. Preprocess the data
3. Split it into training and testing sets
4. Extract features from concept numbers
5. Train a OneVsRest classifier with LogisticRegression
6. Evaluate the model and save it for future use

### Making Predictions

```python
from train_model import predict_type
import pickle

# Load the saved model
with open('concept_classifier_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract model components
model = model_data['model']
vectorizer = model_data['vectorizer']
label_encoder = model_data['label_encoder']

# Make a prediction
concepts = '1086, 1196, 2002'  # Example concepts
predicted_type = predict_type(concepts, model, vectorizer, label_encoder)
print(f'Predicted document type: {predicted_type}')
```

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl (for Excel file operations)

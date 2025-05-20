# Naive Bayes Legal Document Classifier

This is a lightweight implementation of a Naive Bayes classifier for legal documents. It's designed to run efficiently on a laptop without requiring advanced hardware.

## Features

- Fast training on large datasets (processes 45K documents in minutes)
- Low memory usage - suitable for standard laptops
- Simple but effective text classification
- Includes both training and inference scripts
- Comprehensive evaluation metrics

## Files in this Project

- `naive_bayes_train.py` - Script to train the model
- `naive_bayes_inference.py` - Script to make predictions with the trained model
- `model_output/` - Directory where trained models and results are saved

## How to Train the Model

```bash
python naive_bayes_train.py --data_path ../text-mining-v2/full_bert_dataset.csv --output_dir ./model_output
```

### Training Options

- `--data_path` - Path to your dataset CSV file (should have 'text' and 'label' columns)
- `--output_dir` - Directory to save the model and results
- `--test_size` - Proportion of data to use for testing (default: 0.2)
- `--max_features` - Maximum number of features for TF-IDF (default: 20000)
- `--ngram_min` - Minimum n-gram size (default: 1)
- `--ngram_max` - Maximum n-gram size (default: 2)
- `--alpha` - Smoothing parameter for Naive Bayes (default: 0.1)

## How to Use the Trained Model

Once you've trained the model, you can use it to classify new documents:

### Classify a Single Text String

```bash
python naive_bayes_inference.py --model_path ./model_output/naive_bayes_model.joblib --text "REGULATION (EU) No 1025/2012 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL..."
```

### Classify a Text File

```bash
python naive_bayes_inference.py --model_path ./model_output/naive_bayes_model.joblib --text_file path/to/document.txt
```

### Classify a JSON File

```bash
python naive_bayes_inference.py --model_path ./model_output/naive_bayes_model.joblib --json_file path/to/document.json
```

### Classify Multiple Documents from a CSV

```bash
python naive_bayes_inference.py --model_path ./model_output/naive_bayes_model.joblib --csv_file path/to/documents.csv --text_column text --output_path results.json
```

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- tqdm

Install requirements with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib tqdm
```

## Performance Comparison

Naive Bayes vs. BERT for legal document classification:

| Aspect | Naive Bayes | BERT |
|--------|------------|------|
| Training Time | Minutes | Hours/Days |
| Memory Usage | ~1-2 GB | 8+ GB |
| Accuracy | Good (typically 90%+) | Excellent (near 100%) |
| Inference Speed | Very Fast (milliseconds) | Moderate (seconds) |
| Hardware Requirements | Standard Laptop | GPU Recommended |

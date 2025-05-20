# Logistic Regression Legal Document Classifier

This implementation uses Logistic Regression for classifying legal documents (EU legislation). Logistic Regression is a powerful linear model that offers an excellent balance between performance, interpretability, and computational efficiency.

## Key Features

- **Probabilistic Output**: Provides class probabilities rather than just predictions
- **Highly Interpretable**: Clear coefficient weights for understanding feature importance
- **Computationally Efficient**: Fast training and inference times
- **Memory Friendly**: Lower memory footprint than tree-based models
- **Regularization Options**: L1 and L2 regularization options to prevent overfitting

## Files in this Project

- `logistic_regression_train.py` - Script to train the Logistic Regression model
- `logistic_regression_inference.py` - Script to make predictions with the trained model
- `model_output/` - Directory where the trained model and results are saved

## How to Train the Model

```bash
python logistic_regression_train.py --data_path ../text-mining-v2/full_bert_dataset.csv --output_dir ./model_output
```

### Training Options

- `--data_path` - Path to your dataset CSV file (requires 'text' and 'label' columns)
- `--output_dir` - Directory to save the model and results
- `--test_size` - Proportion of data to use for testing (default: 0.2)
- `--max_features` - Maximum number of features for TF-IDF (default: 10000)
- `--ngram_min` - Minimum n-gram size (default: 1)
- `--ngram_max` - Maximum n-gram size (default: 2)
- `--C` - Regularization parameter (default: 1.0, higher values = less regularization)
- `--solver` - Algorithm to use for optimization (default: 'liblinear')
- `--max_iter` - Maximum number of iterations for convergence (default: 1000)
- `--class_weight` - Weighting strategy for classes (default: 'balanced')
- `--grid_search` - Use this flag to perform hyperparameter tuning

## How to Use the Trained Model

### Classify a Single Text String

```bash
python logistic_regression_inference.py --model_path ./model_output/logistic_regression_model.joblib --text "REGULATION (EU) No 1025/2012 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL..."
```

### Classify a JSON File

```bash
python logistic_regression_inference.py --model_path ./model_output/logistic_regression_model.joblib --json_file ../test-dataset/31967R0142.json
```

### Classify Multiple Documents from a CSV

```bash
python logistic_regression_inference.py --model_path ./model_output/logistic_regression_model.joblib --csv_file ../text-mining-v2/full_bert_dataset.csv --output_path results.json
```

### Analyze Feature Importance

Add the `--analyze` flag to see which features influenced a prediction most:

```bash
python logistic_regression_inference.py --model_path ./model_output/logistic_regression_model.joblib --json_file ../test-dataset/31967R0142.json --analyze
```

## Model Comparison

| Aspect | Logistic Regression | SVM | Naive Bayes | Random Forest | BERT |
|--------|---------------------|-----|-------------|--------------|------|
| Training Time | Minutes | Minutes | Minutes | Minutes-Hours | Hours-Days |
| Memory Usage | ~0.5-1 GB | ~1-2 GB | ~1 GB | 2-4 GB | 8+ GB |
| Accuracy | Very Good (~98%) | Very Good (~99%) | Good (~93%) | Very Good (~99%) | Excellent (near 100%) |
| Inference Speed | Very Fast (~2-5ms) | Fast (~3-5ms) | Very Fast (~3ms) | Fast (~3-10ms) | Moderate (~250-500ms) |
| Interpretability | Very High | High | Moderate | High | Low |
| Hyperparameter Sensitivity | Low-Moderate | Moderate | Low | High | Very High |

## Key Advantages of Logistic Regression for Legal Text Classification

1. **Direct Probability Estimates**: Output includes probability scores for each class
2. **Linear Decision Boundaries**: Effective for text data with clear class separations
3. **Feature Coefficient Interpretation**: Coefficients directly indicate importance of words/phrases
4. **Efficient Training**: Fast even on large datasets
5. **Robust to Irrelevant Features**: Less prone to overfitting with proper regularization
6. **Low Memory Requirements**: Smaller model size than tree ensembles or neural networks

## Requirements

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- tqdm

Install requirements with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
```

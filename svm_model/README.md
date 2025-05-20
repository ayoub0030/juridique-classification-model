# SVM Legal Document Classifier

This implementation uses Support Vector Machines (SVM) with linear kernels for classifying legal documents. SVM provides an excellent balance between accuracy and computational efficiency, making it suitable for production environments where both performance and speed are important.

## Key Features

- **High Accuracy**: Linear SVM typically achieves excellent performance for text classification tasks
- **Memory Efficient**: More memory-efficient than Random Forest while maintaining high accuracy
- **Interpretable**: Provides coefficient analysis to understand important features
- **Fast Inference**: Quick prediction times suitable for web applications
- **Regularization Control**: C parameter allows fine-tuning the trade-off between accuracy and generalization

## Files in this Project

- `svm_train.py` - Script to train the SVM model
- `svm_inference.py` - Script to make predictions with the trained model
- `model_output/` - Directory where the trained model and results are saved

## How to Train the Model

```bash
python svm_train.py --data_path ../text-mining-v2/full_bert_dataset.csv --output_dir ./model_output
```

### Training Options

- `--data_path` - Path to your dataset CSV file (requires 'text' and 'label' columns)
- `--output_dir` - Directory to save the model and results
- `--test_size` - Proportion of data to use for testing (default: 0.2)
- `--max_features` - Maximum number of features for TF-IDF (default: 10000)
- `--ngram_min` - Minimum n-gram size (default: 1)
- `--ngram_max` - Maximum n-gram size (default: 2)
- `--C` - Regularization parameter (default: 1.0, higher values = less regularization)
- `--class_weight` - Weighting strategy for classes (default: 'balanced')
- `--grid_search` - Use this flag to perform hyperparameter tuning

## How to Use the Trained Model

### Classify a Single Text String

```bash
python svm_inference.py --model_path ./model_output/svm_model.joblib --text "REGULATION (EU) No 1025/2012 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL..."
```

### Classify a JSON File

```bash
python svm_inference.py --model_path ./model_output/svm_model.joblib --json_file ../test-dataset/31967R0142.json
```

### Classify Multiple Documents from a CSV

```bash
python svm_inference.py --model_path ./model_output/svm_model.joblib --csv_file ../text-mining-v2/full_bert_dataset.csv --output_path results.json
```

### Analyze Feature Importance

Add the `--analyze` flag to see which features influenced a prediction most:

```bash
python svm_inference.py --model_path ./model_output/svm_model.joblib --json_file ../test-dataset/31967R0142.json --analyze
```

## Model Comparison

| Aspect | SVM | Naive Bayes | Random Forest | BERT |
|--------|-----|-------------|--------------|------|
| Training Time | Minutes | Minutes | Minutes-Hours | Hours-Days |
| Memory Usage | ~1-2 GB | ~1 GB | 2-4 GB | 8+ GB |
| Accuracy | Very Good (~97%) | Good (~93%) | Very Good (~95%) | Excellent (near 100%) |
| Inference Speed | Fast (~5-10ms) | Very Fast (~3ms) | Fast (~10-20ms) | Moderate (~250-500ms) |
| Interpretability | High | Moderate | High | Low |
| Hyperparameter Sensitivity | Moderate | Low | High | Very High |

## Key Advantages of SVM for Legal Text Classification

1. **Effective with High-Dimensional Data**: Works well with the large feature spaces from text data
2. **Maximizes Margin**: Creates the widest possible separation between classes
3. **Regularization**: C parameter provides control over model complexity and overfitting
4. **Memory Efficient**: The linear kernel implementation is memory-efficient compared to tree-based models
5. **Fast Training**: Relatively quick training times for medium-sized datasets

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

import os
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import functions from other files
from train_model import concept_tokenizer, predict_type
from convert_json2table import json_folder_to_excel, fields_to_extract

def convert_test_data():
    """Convert the test dataset JSON files to Excel format."""
    test_dir = 'test-dataset'
    test_excel = 'test_dataset.xlsx'
    
    print("Converting test JSON files to Excel...")
    json_folder_to_excel(test_dir, test_excel, fields_to_extract)
    
    return test_excel

def load_model(model_path="concept_classifier_model.pkl"):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        label_encoder = model_data['label_encoder']
        
        print("Model loaded successfully.")
        print(f"Document type classes: {label_encoder.classes_}")
        
        return model, vectorizer, label_encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def evaluate_model(test_data_path, model, vectorizer, label_encoder):
    """Evaluate the model performance on the test dataset."""
    print(f"Loading test data from {test_data_path}...")
    try:
        df_test = pd.read_excel(test_data_path)
        print(f"Test dataset loaded. Shape: {df_test.shape}")
        
        # Drop rows with missing values in key columns
        df_test = df_test.dropna(subset=['concepts', 'type'])
        print(f"After dropping rows with missing values: {df_test.shape}")
        
        # Prepare data for evaluation
        X_test = df_test['concepts']
        y_true = df_test['type']
        
        # Encode true labels
        y_true_encoded = label_encoder.transform(y_true)
        
        # Transform concepts to features
        X_test_features = vectorizer.transform(X_test)
        
        # Make predictions
        print("Making predictions on test data...")
        y_pred_encoded = model.predict(X_test_features)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        report = classification_report(
            y_true_encoded, 
            y_pred_encoded, 
            target_names=label_encoder.classes_,
            zero_division=0
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'test_data': df_test
        }
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

def visualize_results(results):
    """Create visualizations for the evaluation results."""
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        xticklabels=results['y_true'].unique(),
        yticklabels=results['y_true'].unique()
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on Test Data')
    plt.savefig('test_confusion_matrix.png')
    print("Confusion matrix saved as 'test_confusion_matrix.png'")
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 6))
    
    # Count occurrences of each document type
    true_counts = results['y_true'].value_counts()
    pred_counts = pd.Series(results['y_pred']).value_counts()
    
    # Combine into a DataFrame
    count_df = pd.DataFrame({
        'True': true_counts,
        'Predicted': pred_counts
    }).fillna(0)
    
    # Plot as a grouped bar chart
    count_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Document Type Distribution: True vs Predicted')
    plt.xlabel('Document Type')
    plt.ylabel('Count')
    plt.savefig('document_type_distribution.png')
    print("Document type distribution saved as 'document_type_distribution.png'")
    
    # Calculate per-class performance
    report_dict = classification_report(
        results['y_true'], 
        results['y_pred'], 
        output_dict=True
    )
    
    # Extract per-class metrics
    class_metrics = {class_name: report_dict[class_name] 
                    for class_name in results['y_true'].unique() 
                    if class_name in report_dict}
    
    # Create a DataFrame for plotting
    metrics_df = pd.DataFrame(class_metrics).T
    
    # Plot metrics for each class
    plt.figure(figsize=(12, 6))
    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
    plt.title('Performance Metrics by Document Type')
    plt.xlabel('Document Type')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig('class_performance_metrics.png')
    print("Class performance metrics saved as 'class_performance_metrics.png'")

def analyze_errors(results):
    """Analyze misclassified examples."""
    # Create a DataFrame with the true and predicted labels
    comparison_df = pd.DataFrame({
        'concepts': results['test_data']['concepts'],
        'true_type': results['y_true'],
        'predicted_type': results['y_pred'],
        'correct': results['y_true'] == results['y_pred']
    })
    
    # Find misclassified examples
    errors = comparison_df[~comparison_df['correct']]
    
    # Save errors to a CSV file for further analysis
    errors.to_csv('misclassified_examples.csv', index=False)
    print(f"Found {len(errors)} misclassified examples out of {len(comparison_df)} total examples.")
    print("Misclassified examples saved to 'misclassified_examples.csv'")
    
    # Analyze error patterns
    if len(errors) > 0:
        print("\nError patterns analysis:")
        error_patterns = errors.groupby(['true_type', 'predicted_type']).size().reset_index()
        error_patterns.columns = ['True Type', 'Predicted Type', 'Count']
        error_patterns = error_patterns.sort_values('Count', ascending=False)
        print(error_patterns)
        
        # Look at some example errors
        print("\nExample misclassifications:")
        for _, row in errors.head(5).iterrows():
            print(f"True type: {row['true_type']}, Predicted: {row['predicted_type']}")
            print(f"Concepts: {row['concepts']}")
            print("---")

def main():
    # Step 1: Convert test data to Excel
    test_excel = convert_test_data()
    
    # Step 2: Load the trained model
    model, vectorizer, label_encoder = load_model()
    if model is None:
        return
    
    # Step 3: Evaluate the model on test data
    try:
        results = evaluate_model(test_excel, model, vectorizer, label_encoder)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return
    
    # Step 4: Display and visualize results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    
    # Step 5: Create visualizations
    visualize_results(results)
    
    # Step 6: Analyze errors
    analyze_errors(results)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print("Summary of findings:")
    print(f"- Model accuracy on test data: {results['accuracy']:.4f}")
    
    # Calculate overall metrics
    report_dict = classification_report(results['y_true'], results['y_pred'], output_dict=True)
    print(f"- Overall precision: {report_dict['macro avg']['precision']:.4f}")
    print(f"- Overall recall: {report_dict['macro avg']['recall']:.4f}")
    print(f"- Overall F1-score: {report_dict['macro avg']['f1-score']:.4f}")
    
    # Compare to training performance
    print("\nCompared to training performance (previous accuracy: 0.8638):")
    diff = results['accuracy'] - 0.8638
    if diff > 0.05:
        print("- The model performs significantly BETTER on the test data!")
    elif diff < -0.05:
        print("- The model performs significantly WORSE on the test data.")
    else:
        print("- The model performs SIMILARLY on the test data.")
    
    # Recommendations
    print("\nRecommendations:")
    if results['accuracy'] < 0.8:
        print("- Consider improving the model by adding more training data")
        print("- Try different classification algorithms or parameter tuning")
    else:
        print("- The model is performing well and can be used for document classification")
        print("- For further improvements, consider enriching the features or using more advanced models")

if __name__ == "__main__":
    # Fix import error with convert-json2table.py (dash in name)
    try:
        import convert_json2table
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "convert_json2table", 
            "convert-json2table.py"
        )
        convert_json2table = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert_json2table)
        
        # Access the required functions/variables
        fields_to_extract = convert_json2table.fields_to_extract
        json_folder_to_excel = convert_json2table.json_folder_to_excel
    
    main()

import pandas as pd
import os

def analyze_results():
    """Analyze evaluation results from generated files"""
    # Check if misclassified examples file exists
    if os.path.exists('misclassified_examples.csv'):
        # Read misclassified examples
        errors_df = pd.read_csv('misclassified_examples.csv')
        error_count = len(errors_df)
        
        # Read test dataset to get total size
        if os.path.exists('test_dataset.xlsx'):
            test_df = pd.read_excel('test_dataset.xlsx')
            total_count = len(test_df)
            accuracy = 1 - (error_count / total_count)
            
            print(f"Total examples in test set: {total_count}")
            print(f"Misclassified examples: {error_count}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Analyze error patterns
            error_patterns = errors_df.groupby(['true_type', 'predicted_type']).size()
            print("\nError patterns:")
            print(error_patterns)
            
            # Distribution of document types in test set
            type_dist = test_df['type'].value_counts()
            print("\nDocument type distribution in test set:")
            print(type_dist)
            
            # Calculate per-class accuracy
            class_accuracy = {}
            for doc_type in test_df['type'].unique():
                type_total = len(test_df[test_df['type'] == doc_type])
                type_errors = len(errors_df[errors_df['true_type'] == doc_type])
                type_accuracy = 1 - (type_errors / type_total)
                class_accuracy[doc_type] = type_accuracy
            
            print("\nPer-class accuracy:")
            for doc_type, acc in class_accuracy.items():
                print(f"{doc_type}: {acc:.4f} ({acc*100:.2f}%)")
        else:
            print("Test dataset file (test_dataset.xlsx) not found.")
    else:
        print("Misclassified examples file (misclassified_examples.csv) not found.")

if __name__ == "__main__":
    analyze_results()

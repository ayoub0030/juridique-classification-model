import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

def prepare_text_dataset(json_folder_path, output_file_path, min_text_length=5):
    """
    Extracts main_body text and document type from JSON files and saves to CSV.
    
    Args:
        json_folder_path: Path to folder containing JSON files
        output_file_path: Path to save the output CSV file
        min_text_length: Minimum character length for main_body text to be included
    """
    print(f"Extracting text and labels from JSON files in {json_folder_path}...")
    
    # Check if the input directory exists
    if not os.path.isdir(json_folder_path):
        print(f"Error: Folder not found: {json_folder_path}")
        return
    
    # Get list of JSON files
    try:
        json_files = [f for f in os.listdir(json_folder_path) if f.lower().endswith('.json')]
    except OSError as e:
        print(f"Error accessing folder {json_folder_path}: {e}")
        return
    
    if not json_files:
        print(f"No JSON files found in '{json_folder_path}'.")
        return
    
    print(f"Found {len(json_files)} JSON files. Processing...")
    
    # Prepare data storage
    data = []
    skipped_count = 0
    empty_text_count = 0
    
    # Process each file
    for filename in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(json_folder_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)
            
            # Extract document type (label)
            doc_type = article.get('type', None)
            
            # Extract and process main_body text
            main_body = article.get('main_body', [])
            
            # Skip if missing essential data
            if not doc_type or main_body is None:
                skipped_count += 1
                continue
            
            # Convert main_body list to a single string
            if isinstance(main_body, list):
                text = ' '.join(main_body)
            else:
                text = str(main_body)
            
            # Skip documents with very short text
            if len(text) < min_text_length:
                empty_text_count += 1
                continue
            
            # Add to dataset
            data.append({
                'text': text.strip(),
                'label': doc_type,
                'celex_id': article.get('celex_id', '')  # Keep ID for reference
            })
                
        except json.JSONDecodeError:
            print(f"\nWarning: Could not decode JSON from file: {filename}. Skipping.")
        except Exception as e:
            print(f"\nWarning: Error processing file {filename}: {e}. Skipping.")
    
    # Create DataFrame
    if not data:
        print("No valid data extracted. Exiting.")
        return
    
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"\nExtracted {len(df)} documents with valid text and labels.")
    print(f"Skipped {skipped_count} documents missing type or main_body.")
    print(f"Skipped {empty_text_count} documents with text shorter than {min_text_length} characters.")
    
    # Check label distribution
    label_counts = df['label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")
    
    # Save to CSV
    print(f"\nSaving to {output_file_path}...")
    df.to_csv(output_file_path, index=False)
    print(f"Data successfully saved to {output_file_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Prepare text dataset for BERT classification')
    parser.add_argument('--input_folder', type=str, default='dataset_folder', 
                        help='Folder containing JSON files')
    parser.add_argument('--output_file', type=str, default='text_classification_dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--min_length', type=int, default=5,
                        help='Minimum text length to include')
    
    args = parser.parse_args()
    
    prepare_text_dataset(args.input_folder, args.output_file, args.min_length)

if __name__ == "__main__":
    main()

import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
import time
from pathlib import Path

def prepare_bert_dataset(json_folder_path, output_file_path, min_text_length=10, batch_size=5000, resume=True):
    """
    Extracts header + recitals text and document type from JSON files and saves to CSV.
    Processes files in batches and supports resuming from interruptions.
    
    Args:
        json_folder_path: Path to folder containing JSON files
        output_file_path: Path to save the output CSV file
        min_text_length: Minimum character length for text to be included
        batch_size: Number of files to process in each batch
        resume: Whether to resume from previous progress
    """
    print(f"Extracting header + recitals and labels from JSON files in {json_folder_path}...")
    
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
    
    print(f"Found {len(json_files)} JSON files.")
    
    # Check for progress file
    progress_file = 'bert_extraction_progress.json'
    progress_data = {'processed_files': [], 'data': []}
    
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                
            print(f"Resuming from previous run. Already processed {len(progress_data['processed_files'])} files.")
            json_files = [f for f in json_files if f not in progress_data['processed_files']]
            print(f"Remaining files to process: {len(json_files)}")
        except Exception as e:
            print(f"Error loading progress file: {e}. Starting from beginning.")
            progress_data = {'processed_files': [], 'data': []}
    
    # Prepare data storage
    data = progress_data['data']
    processed_files = progress_data['processed_files']
    skipped_count = 0
    empty_text_count = 0
    
    # Process files in batches
    total_batches = (len(json_files) + batch_size - 1) // batch_size
    
    try:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(json_files))
            batch_files = json_files[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_idx+1}/{total_batches} ({len(batch_files)} files)...")
            
            # Process each file in the batch
            for filename in tqdm(batch_files, desc=f"Batch {batch_idx+1}"):
                file_path = os.path.join(json_folder_path, filename)
                
                try:
                    # Try different encodings if utf-8 fails
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            article = json.load(f)
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            article = json.load(f)
                    
                    # Extract document type (label)
                    doc_type = article.get('type', None)
                    
                    # Extract header and recitals
                    header = article.get('header', '')
                    recitals = article.get('recitals', '')
                    
                    # Skip if missing essential data
                    if not doc_type or (not header and not recitals):
                        skipped_count += 1
                        processed_files.append(filename)  # Mark as processed even if skipped
                        continue
                    
                    # Combine header and recitals
                    if header and recitals:
                        text = f"{header}\n{recitals}"
                    elif header:
                        text = header
                    else:
                        text = recitals
                    
                    # Skip documents with very short text
                    if len(text) < min_text_length:
                        empty_text_count += 1
                        processed_files.append(filename)  # Mark as processed even if skipped
                        continue
                    
                    # Add to dataset
                    data.append({
                        'text': text.strip(),
                        'label': doc_type,
                        'celex_id': article.get('celex_id', '')  # Keep ID for reference
                    })
                    
                    # Mark as processed
                    processed_files.append(filename)
                        
                except json.JSONDecodeError:
                    print(f"\nWarning: Could not decode JSON from file: {filename}. Skipping.")
                    processed_files.append(filename)  # Mark as processed even if error
                except Exception as e:
                    print(f"\nWarning: Error processing file {filename}: {e}. Skipping.")
                    processed_files.append(filename)  # Mark as processed even if error
            
            # Save progress after each batch
            progress_data = {'processed_files': processed_files, 'data': data}
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f)
            
            print(f"Progress saved. Processed {len(processed_files)}/{len(processed_files) + len(json_files)} files.")
            print(f"Valid documents extracted so far: {len(data)}")
            
            # Save intermediate CSV after each batch
            temp_df = pd.DataFrame(data)
            temp_output_path = f"{Path(output_file_path).stem}_partial_{batch_idx+1}.csv"
            temp_df.to_csv(temp_output_path, index=False)
            print(f"Intermediate data saved to {temp_output_path}")
            
            # Optional: Brief pause between batches
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        print(f"Processed {len(processed_files)}/{len(processed_files) + len(json_files)} files so far.")
    
    # Create final DataFrame
    if not data:
        print("No valid data extracted. Exiting.")
        return
    
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"\nExtracted {len(df)} documents with valid text and labels.")
    print(f"Skipped {skipped_count} documents missing type or text.")
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
    
    # Clean up progress file after successful completion
    if os.path.exists(progress_file) and len(json_files) == 0:
        os.remove(progress_file)
        print("Removed progress file since all processing is complete.")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Prepare text dataset for BERT classification')
    parser.add_argument('--input_folder', type=str, default='dataset_folder', 
                        help='Folder containing JSON files')
    parser.add_argument('--output_file', type=str, default='bert_classification_dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--min_length', type=int, default=10,
                        help='Minimum text length to include')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Number of files to process in each batch')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not resume from previous progress')
    
    args = parser.parse_args()
    
    prepare_bert_dataset(
        args.input_folder, 
        args.output_file, 
        args.min_length,
        args.batch_size,
        not args.no_resume
    )

if __name__ == "__main__":
    main()

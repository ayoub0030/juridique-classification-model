import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import time
import logging
from pathlib import Path
import sys
import re
import string
import unicodedata

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def clean_text_content(text, remove_punctuation=True, remove_numbers=False, lowercase=True):
    """
    Clean text by removing unwanted characters, normalizing whitespace, etc.
    
    Args:
        text: The text to clean
        remove_punctuation: Whether to remove punctuation
        remove_numbers: Whether to remove numeric characters
        lowercase: Whether to convert text to lowercase
    
    Returns:
        Cleaned text string
    """
    # Skip cleaning if text is None or empty
    if not text:
        return text
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Remove punctuation if requested
    if remove_punctuation:
        # Define punctuation to remove - keep some useful ones like '-' (hyphen)
        punctuation_to_remove = string.punctuation.replace('-', '')
        translator = str.maketrans('', '', punctuation_to_remove)
        text = text.translate(translator)
    
    # Remove numbers if requested
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_full_dataset(json_folder_path, output_file_path, min_text_length=10, batch_size=5000, 
                       clean_text=False, remove_punctuation=True, remove_numbers=False, lowercase=True):
    """
    Processes all JSON files in the folder and creates a full dataset without 
    splitting into train/validation sets.
    
    Args:
        json_folder_path: Path to folder containing JSON files
        output_file_path: Path to save the output CSV file
        min_text_length: Minimum character length for text to be included
        batch_size: Number of files to process in each batch (for progress tracking)
    """
    logging.info(f"Processing full dataset from {json_folder_path}")
    
    # Check if the input directory exists
    if not os.path.isdir(json_folder_path):
        logging.error(f"Error: Folder not found: {json_folder_path}")
        return
    
    # Get list of JSON files
    try:
        json_files = [f for f in os.listdir(json_folder_path) if f.lower().endswith('.json')]
    except OSError as e:
        logging.error(f"Error accessing folder {json_folder_path}: {e}")
        return
    
    if not json_files:
        logging.error(f"No JSON files found in '{json_folder_path}'.")
        return
    
    total_files = len(json_files)
    logging.info(f"Found {total_files} JSON files to process.")
    
    # Prepare data storage
    data = []
    skipped_count = 0
    empty_text_count = 0
    error_count = 0
    start_time = time.time()
    
    # Process files in batches with progress bar
    for i, filename in enumerate(tqdm(json_files, desc="Processing files", total=total_files)):
        try:
            file_path = os.path.join(json_folder_path, filename)
            
            # Try different encodings if utf-8 fails
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    article = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"JSON decode error in file {filename}, skipping")
                error_count += 1
                continue
            
            # Extract document type (label)
            doc_type = article.get('type', None)
            
            # Extract header, recitals, and main body
            header = article.get('header', '')
            recitals = article.get('recitals', '')
            main_body = article.get('main_body', [])
            
            # Skip if missing essential data
            if not doc_type:
                skipped_count += 1
                continue
            
            # Process main_body - it's usually a list of strings that need to be joined
            main_body_text = ''
            if main_body:
                if isinstance(main_body, list):
                    main_body_text = '\n'.join(main_body)
                else:
                    main_body_text = str(main_body)
            
            # Combine header, recitals, and main_body
            text_parts = []
            if header:
                text_parts.append(header)
            if recitals:
                text_parts.append(recitals)
            if main_body_text:
                text_parts.append(main_body_text)
                
            # Skip if no text content is available
            if not text_parts:
                empty_text_count += 1
                continue
                
            # Join all text parts
            text = '\n\n'.join(text_parts)
            
            # Clean the text if cleaning is enabled
            if clean_text:
                text = clean_text_content(text, remove_punctuation, remove_numbers, lowercase)
            
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
            
            # Save intermediate results and report progress periodically
            if (i + 1) % batch_size == 0 or (i + 1) == total_files:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total_files - (i + 1)) / rate if rate > 0 else 0
                
                logging.info(f"Processed {i + 1}/{total_files} files ({(i + 1)/total_files*100:.1f}%)")
                logging.info(f"Elapsed time: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
                logging.info(f"Current dataset size: {len(data)} valid documents")
                
                # Save interim DataFrame to avoid losing progress
                if len(data) > 0 and (i + 1) % (batch_size * 5) == 0:
                    interim_df = pd.DataFrame(data)
                    interim_path = output_file_path.replace('.csv', f'_interim_{i+1}.csv')
                    interim_df.to_csv(interim_path, index=False)
                    logging.info(f"Saved interim results to {interim_path}")
        
        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
            error_count += 1
            continue
    
    # Create DataFrame
    if not data:
        logging.error("No valid data extracted. Exiting.")
        return
    
    df = pd.DataFrame(data)
    
    # Print statistics
    logging.info(f"\nExtracted {len(df)} documents with valid text and labels.")
    logging.info(f"Skipped {skipped_count} documents missing type or text.")
    logging.info(f"Skipped {empty_text_count} documents with text shorter than {min_text_length} characters.")
    logging.info(f"Encountered {error_count} errors during processing.")
    
    # Check label distribution
    label_counts = df['label'].value_counts()
    logging.info("\nLabel distribution:")
    for label, count in label_counts.items():
        logging.info(f"  {label}: {count} ({count/len(df)*100:.2f}%)")
    
    # Check text length statistics
    text_lengths = df['text'].apply(len)
    logging.info("\nText length statistics (characters):")
    logging.info(f"  Min: {text_lengths.min()}")
    logging.info(f"  Max: {text_lengths.max()}")
    logging.info(f"  Mean: {text_lengths.mean():.1f}")
    logging.info(f"  Median: {text_lengths.median():.1f}")
    
    # Save to CSV
    logging.info(f"\nSaving to {output_file_path}...")
    
    # Log text cleaning status
    if clean_text:
        logging.info("Text cleaning was applied with the following settings:")
        logging.info(f"  - Remove punctuation: {remove_punctuation}")
        logging.info(f"  - Remove numbers: {remove_numbers}")
        logging.info(f"  - Lowercase: {lowercase}")
    else:
        logging.info("No text cleaning was applied.")
    
    # Create simplified DataFrame with just the text and label columns
    simplified_df = df[['text', 'label']]
    
    # Save as CSV
    simplified_df.to_csv(output_file_path, index=False)
    logging.info(f"CSV file saved to {output_file_path}")
    
    # Also save as Excel
    excel_path = output_file_path.replace('.csv', '.xlsx')
    try:
        simplified_df.to_excel(excel_path, index=False)
        logging.info(f"Excel file saved to {excel_path}")
    except Exception as e:
        logging.warning(f"Could not save Excel file: {e}")
    
    # Create a sample file with examples of each class
    try:
        sample_df = pd.DataFrame()
        for label in df['label'].unique():
            sample_df = pd.concat([sample_df, df[df['label'] == label].head(5)])
        
        sample_path = output_file_path.replace('.csv', '_samples.csv')
        sample_df[['text', 'label']].to_csv(sample_path, index=False)
        logging.info(f"Sample examples saved to {sample_path}")
    except Exception as e:
        logging.warning(f"Could not create sample file: {e}")
    
    # Calculate processing speed
    total_time = time.time() - start_time
    logging.info(f"\nTotal processing time: {total_time:.1f}s")
    logging.info(f"Average processing speed: {total_files/total_time:.1f} documents per second")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Process full dataset for BERT legal document classification')
    parser.add_argument('--input_folder', type=str, default='../dataset_folder',
                        help='Folder containing JSON files')
    parser.add_argument('--output_file', type=str, default='full_bert_dataset.csv',
                        help='Output file path')
    parser.add_argument('--min_length', type=int, default=10,
                        help='Minimum text length to include')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Number of files to process in each batch for progress reporting')
    # Text cleaning arguments
    parser.add_argument('--clean_text', action='store_true',
                        help='Clean the text by removing unwanted characters')
    parser.add_argument('--remove_punctuation', action='store_true', 
                        help='Remove punctuation from the text')
    parser.add_argument('--remove_numbers', action='store_true',
                        help='Remove numerical digits from the text')
    parser.add_argument('--lowercase', action='store_true', 
                        help='Convert all text to lowercase')
    
    args = parser.parse_args()
    
    process_full_dataset(
        args.input_folder,
        args.output_file,
        args.min_length,
        args.batch_size,
        args.clean_text,
        args.remove_punctuation,
        args.remove_numbers,
        args.lowercase
    )

if __name__ == "__main__":
    main()

import os
import json
import pandas as pd
from tqdm import tqdm # Optional: for a nice progress bar
import sys # To check Python version for potential encoding issues

def process_article_json(data, filename, fields_to_extract):
    """
    Extracts data for one article, handling specific list fields.
    """
    article_info = {}
    for field in fields_to_extract:
        value = data.get(field, None) # Use .get() for safety

        # --- Special Handling for list fields ---
        if field == 'main_body' and isinstance(value, list):
            # Join the list of strings into a single text block with newlines
            article_info[field] = "\n".join(value)
        elif field == 'concepts' and isinstance(value, list):
            # Join the list of concepts into a comma-separated string
            article_info[field] = ", ".join(value)
        else:
            # For other fields, just take the value as is
             article_info[field] = value
        # --- End Special Handling ---

    # Optional: Add filename as a column for traceability
    article_info['source_filename'] = filename
    return article_info

def json_folder_to_excel(json_folder_path, output_excel_path, fields_to_extract):
    """
    Reads JSON files from a folder, extracts specified fields (handling lists),
    and saves the data to an Excel file.

    Args:
        json_folder_path (str): Path to the folder containing JSON files.
        output_excel_path (str): Path where the output Excel file will be saved.
        fields_to_extract (list): A list of strings representing the keys
                                   to extract from each JSON file. These will
                                   become the columns in the Excel file.
    """
    all_article_data = []
    
    # Check if the input directory exists
    if not os.path.isdir(json_folder_path):
        print(f"Error: Folder not found: {json_folder_path}")
        return

    try:
        json_files = [f for f in os.listdir(json_folder_path) if f.lower().endswith('.json')]
    except OSError as e:
        print(f"Error accessing folder {json_folder_path}: {e}")
        return
        
    if not json_files:
        print(f"No JSON files found in '{json_folder_path}'.")
        return

    print(f"Found {len(json_files)} JSON files in '{json_folder_path}'. Processing...")

    # Determine appropriate encoding - utf-8 is usually safe
    file_encoding = 'utf-8'
    # Optional: More robust encoding detection could be added if needed

    # Use tqdm for a progress bar (optional, remove if not installed/needed)
    for filename in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(json_folder_path, filename)
        try:
            # Explicitly use utf-8 encoding, common for JSON
            with open(file_path, 'r', encoding=file_encoding) as f:
                data = json.load(f)

            # Process the loaded JSON data using the helper function
            extracted_data = process_article_json(data, filename, fields_to_extract)
            all_article_data.append(extracted_data)

        except json.JSONDecodeError:
            print(f"\nWarning: Could not decode JSON from file: {filename}. Check encoding or structure. Skipping.")
        except FileNotFoundError:
            # This shouldn't happen if os.listdir worked, but good practice
            print(f"\nWarning: File not found during processing: {filename}. Skipping.")
        except Exception as e:
            print(f"\nWarning: An unexpected error occurred processing file {filename}: {e}. Skipping.")

    if not all_article_data:
        print("No data was successfully extracted. Exiting.")
        return

    # Create pandas DataFrame
    print("\nCreating DataFrame...")
    df = pd.DataFrame(all_article_data)

    # Define final column order (fields + source_filename)
    column_order = fields_to_extract + ['source_filename']
    # Reindex to ensure all desired columns are present and in order
    # This handles cases where some files might have missed certain fields entirely
    df = df.reindex(columns=column_order)

    # Save to Excel
    print(f"Saving DataFrame to '{output_excel_path}'...")
    try:
        # index=False prevents pandas from writing the DataFrame index as a column
        # engine='openpyxl' is needed for .xlsx format
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"Successfully saved data to Excel: {output_excel_path}")
    except ImportError:
         print("\nError: The 'openpyxl' library is required to write Excel (.xlsx) files.")
         print("Please install it using: pip install openpyxl")
         print("Attempting to save as CSV instead...")
         save_as_csv(df, output_excel_path)
    except Exception as e:
        print(f"\nError saving to Excel: {e}")
        print("Attempting to save as CSV instead...")
        save_as_csv(df, output_excel_path)

def save_as_csv(dataframe, original_excel_path):
    """Helper function to save DataFrame as CSV."""
    try:
        csv_path = original_excel_path.rsplit('.', 1)[0] + '.csv'
        dataframe.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig helps Excel open CSVs correctly
        print(f"Successfully saved data to CSV: {csv_path}")
    except Exception as e_csv:
        print(f"Error saving to CSV: {e_csv}")


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- CONFIGURATION --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# 1. Specify the folder containing your JSON article files
#    *** IMPORTANT: Replace this with the actual path to your folder ***
json_directory = 'dataset_folder'  # <--- CHANGE THIS

# 2. Specify the desired output Excel file path
#    The script will create this file.
output_file = 'expanded_dataset.xlsx'         # <--- CHANGE THIS (if needed)

# 3. Specify the fields (keys) you want to extract from your JSON files.
#    These are based EXACTLY on your example JSON.
fields_to_extract = [
    "celex_id",
    "uri",
    "type",         # Likely your target variable for classification
    "concepts",     # Will be converted to comma-separated string
    "title",
    "header",
    "recitals",
    "main_body",    # Will be converted to a single string with newlines
    "attachments"
]

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- RUN THE CONVERSION --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

if __name__ == "__main__":
    json_folder_to_excel(json_directory, output_file, fields_to_extract)
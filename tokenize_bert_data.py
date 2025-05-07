import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import pickle
import os
from tqdm import tqdm

class LegalDocumentDataset(Dataset):
    """Legal document dataset for BERT classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension (since we're tokenizing one by one)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def tokenize_and_prepare_data(dataset_path, output_dir, test_size=0.1, max_length=512, batch_size=16):
    """
    Tokenize the text data using BERT tokenizer and prepare datasets for training.
    
    Args:
        dataset_path: Path to the CSV file with text and labels
        output_dir: Directory to save processed data
        test_size: Proportion of data to use for validation
        max_length: Maximum sequence length for BERT
        batch_size: Batch size for DataLoader
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Check data
    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique labels: {df['label'].nunique()}")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['label'])
    
    # Save label encoder for inference
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Label encoding mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")
    
    # Load BERT tokenizer
    print("Loading BERT tokenizer (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Split into train and validation sets
    texts = df['text'].values
    labels = encoded_labels
    
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create datasets
    print("Creating PyTorch datasets...")
    train_dataset = LegalDocumentDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = LegalDocumentDataset(X_val, y_val, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving processed data to {output_dir}...")
    
    # Save dataloaders
    torch.save(train_loader, os.path.join(output_dir, 'train_loader.pth'))
    torch.save(val_loader, os.path.join(output_dir, 'val_loader.pth'))
    
    # Save some metadata
    metadata = {
        'num_labels': len(label_encoder.classes_),
        'label_names': label_encoder.classes_,
        'max_length': max_length,
        'train_size': len(X_train),
        'val_size': len(X_val)
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Data preparation complete!")
    
    # Return a single batch for inspection
    sample_batch = next(iter(train_loader))
    return sample_batch

def inspect_tokenized_data(batch, tokenizer):
    """Inspect a batch of tokenized data."""
    print("\nSample batch inspection:")
    print(f"Batch size: {len(batch['input_ids'])}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # Decode a sample
    sample_idx = 0
    print("\nSample document:")
    decoded_text = tokenizer.decode(batch['input_ids'][sample_idx])
    print(f"Decoded text (truncated): {decoded_text[:100]}...")
    print(f"Label: {batch['labels'][sample_idx].item()}")
    
    # Check sequence lengths
    seq_lengths = batch['attention_mask'].sum(dim=1)
    print(f"\nSequence lengths in this batch:")
    print(f"  Min: {seq_lengths.min().item()}")
    print(f"  Max: {seq_lengths.max().item()}")
    print(f"  Mean: {seq_lengths.float().mean().item():.2f}")
    
    # Check for truncation
    max_length = batch['input_ids'].shape[1]
    num_truncated = (seq_lengths == max_length).sum().item()
    print(f"Number of possibly truncated sequences: {num_truncated}/{len(batch['input_ids'])}")

def main():
    parser = argparse.ArgumentParser(description='Tokenize text data for BERT')
    parser.add_argument('--input_file', type=str, default='bert_classification_dataset.csv',
                        help='Input CSV file with text and labels')
    parser.add_argument('--output_dir', type=str, default='bert_data',
                        help='Output directory for processed data')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of data to use for validation')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for BERT')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for DataLoader')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tokenize and prepare data
    sample_batch = tokenize_and_prepare_data(
        args.input_file,
        args.output_dir,
        args.test_size,
        args.max_length,
        args.batch_size
    )
    
    # Inspect tokenized data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inspect_tokenized_data(sample_batch, tokenizer)

if __name__ == "__main__":
    main()

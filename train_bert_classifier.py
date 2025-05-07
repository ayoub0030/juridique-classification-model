import os
import argparse
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Dataset class for legal documents
class LegalDocumentDataset(Dataset):
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
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=2):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", position=0, leave=True)
    
    # For gradient accumulation
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        total_loss += loss.item()
        
        # Update weights after accumulating gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_description(f"Training (loss: {loss.item():.4f})")
    
    # Final step if dataset size is not divisible by gradient_accumulation_steps
    if len(dataloader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, zero_division=0)
    
    return val_loss / len(dataloader), accuracy, report, predictions, true_labels

# Main training function
def train_bert_classifier(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Subsample the dataset to 20k examples
    if len(df) > args.max_samples:
        print(f"Subsampling dataset from {len(df)} to {args.max_samples} examples...")
        # Stratified sampling to maintain class distribution
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(args.max_samples * len(x) / len(df))))
        )
        print(f"New dataset size: {len(df)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])
    
    # Save label encoder
    with open(os.path.join(args.output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Display label mapping
    print("Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df['encoded_label']
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = LegalDocumentDataset(
        train_df['text'].values,
        train_df['encoded_label'].values,
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = LegalDocumentDataset(
        val_df['text'].values,
        val_df['encoded_label'].values,
        tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=0  # No additional workers to save memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        num_workers=0  # No additional workers to save memory
    )
    
    # Load pre-trained model
    print(f"Loading pre-trained model: {args.model_name}")
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_encoder.classes_),
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Move model to device
    model.to(device)
    
    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    
    best_val_accuracy = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.gradient_accumulation_steps
        )
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_accuracy, val_report, _, _ = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(val_report)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {val_accuracy:.4f}")
        
        # Early stopping
        if args.early_stopping > 0 and epoch > args.early_stopping:
            last_accuracies = val_accuracies[-(args.early_stopping+1):]
            if all(last_accuracies[i] <= last_accuracies[i-1] for i in range(1, len(last_accuracies))):
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation
    _, final_accuracy, final_report, predictions, true_labels = evaluate(model, val_loader, device)
    
    print("\nFinal Evaluation:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print("\nClassification Report:")
    print(final_report)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    
    with open(os.path.join(args.output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Save model config
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'num_labels': len(label_encoder.classes_),
        'label_mapping': dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    }
    
    with open(os.path.join(args.output_dir, 'model_config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
    # Save the model (can be loaded with BertForSequenceClassification.from_pretrained)
    model_save_path = os.path.join(args.output_dir, "model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Model and training artifacts saved to {args.output_dir}")
    
    return final_accuracy

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT for legal document classification')
    
    # Data parameters
    parser.add_argument('--input_file', type=str, default='bert_classification_dataset.csv',
                        help='Input CSV file with text and labels')
    parser.add_argument('--output_dir', type=str, default='bert_model',
                        help='Output directory for trained model and artifacts')
    parser.add_argument('--max_samples', type=int, default=20000,
                        help='Maximum number of samples to use for training and validation')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data to use for validation')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pre-trained model name')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and validation')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of steps to accumulate gradients before updating weights')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='Epsilon for Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--early_stopping', type=int, default=2,
                        help='Number of epochs with no improvement for early stopping (0 to disable)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    train_bert_classifier(args)

if __name__ == '__main__':
    main()

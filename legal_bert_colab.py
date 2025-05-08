"""
LEGAL DOCUMENT CLASSIFICATION WITH BERT
Google Colab Notebook Template

This file contains all the code sections for your Colab notebook.
Just copy each section into a separate cell in Colab.

Instructions:
1. Create a new Colab notebook
2. Copy each section (marked with #### SECTION X ####) into a new cell
3. Modify paths as needed
4. Run the cells sequentially
"""

#### SECTION 1: SETUP AND INSTALLATION ####
# Run this cell first to install all necessary packages
!pip install transformers datasets torch scikit-learn tqdm matplotlib seaborn pandas

# Mount Google Drive for saving models and data
from google.colab import drive
drive.mount('/content/drive')

# Create a folder for the project
!mkdir -p /content/drive/MyDrive/legal_bert_classification

# Check if GPU is available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print GPU info if available
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


#### SECTION 2: DATA UPLOAD ####
# Option 1: Upload the dataset directly
from google.colab import files

# Use this for small files
print("Upload your bert_classification_dataset.csv file:")
uploaded = files.upload()  # Will prompt for file upload

# Option 2: If you've already uploaded to Drive, set the path
# dataset_path = '/content/drive/MyDrive/legal_bert_classification/bert_classification_dataset.csv'


#### SECTION 3: DATA PREPARATION ####
# If you need to prepare data from JSON files in Colab
import os
import json
import pandas as pd
from tqdm import tqdm

def prepare_bert_dataset(json_folder_path, output_file_path, min_text_length=10):
    """Extracts header + recitals and document type from JSON files and saves to CSV."""
    
    print(f"Extracting header + recitals and labels from JSON files...")
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(json_folder_path) if f.lower().endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found.")
        return
    
    print(f"Found {len(json_files)} JSON files. Processing...")
    
    # Prepare data storage
    data = []
    skipped_count = 0
    empty_text_count = 0
    
    # Process files with progress bar
    for filename in tqdm(json_files):
        file_path = os.path.join(json_folder_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article = json.load(f)
            
            # Extract document type (label)
            doc_type = article.get('type', None)
            
            # Extract header and recitals
            header = article.get('header', '')
            recitals = article.get('recitals', '')
            
            # Skip if missing essential data
            if not doc_type or (not header and not recitals):
                skipped_count += 1
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
                continue
            
            # Add to dataset
            data.append({
                'text': text.strip(),
                'label': doc_type,
                'celex_id': article.get('celex_id', '')  # Keep ID for reference
            })
                
        except Exception as e:
            print(f"\nWarning: Error processing file {filename}: {e}. Skipping.")
    
    # Create DataFrame
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
    
    return df

# If you have JSON files uploaded to a folder in Drive, uncomment and run:
# json_folder_path = '/content/drive/MyDrive/legal_bert_classification/dataset_folder'
# output_file_path = '/content/drive/MyDrive/legal_bert_classification/bert_classification_dataset.csv'
# df = prepare_bert_dataset(json_folder_path, output_file_path)


#### SECTION 4: DATA EXPLORATION ####
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (adjust path as needed)
# If you uploaded the file in Section 2, use:
df = pd.read_csv('bert_classification_dataset.csv')

# Or if you saved it to Drive, use:
# df = pd.read_csv('/content/drive/MyDrive/legal_bert_classification/bert_classification_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Number of unique labels: {df['label'].nunique()}")

# Display label distribution
print("Label distribution:")
label_counts = df['label'].value_counts()
print(label_counts)

# Plot label distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title('Document Type Distribution')
plt.xlabel('Document Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Text length analysis
df['text_length'] = df['text'].apply(len)
print("\nText length statistics:")
print(df['text_length'].describe())

# Plot text length distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='text_length', hue='label', bins=50, element='step')
plt.title('Text Length Distribution by Document Type')
plt.xlabel('Text Length (characters)')
plt.xlim(0, df['text_length'].quantile(0.99))  # Limit x-axis to 99th percentile
plt.legend(title='Document Type')
plt.tight_layout()
plt.show()

# Sample document display
print("\nSample document from each class:")
for label in df['label'].unique():
    sample = df[df['label'] == label].iloc[0]
    print(f"\n--- {label} Example ---")
    print(f"Text (first 300 chars): {sample['text'][:300]}...")


#### SECTION 5: DATA PREPROCESSING ####
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, AutoTokenizer
import pickle

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define dataset class
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

# Load the dataset (adjust path if needed)
# df = pd.read_csv('bert_classification_dataset.csv')

# Subsample the dataset to reduce training time (optional)
# Comment out if you want to use the full dataset
# sample_size = 20000
# if len(df) > sample_size:
#     df = df.groupby('label', group_keys=False).apply(
#         lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))))
#     )
#     print(f"Subsampled dataset to {len(df)} examples")

# Encode labels
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])

# Display label mapping
print("Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} -> {i}")

# Save label encoder
with open('/content/drive/MyDrive/legal_bert_classification/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split dataset
train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df['encoded_label']
)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Load tokenizer
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Alternative option: use a legal domain BERT model
# tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

# Create datasets
train_dataset = LegalDocumentDataset(
    train_df['text'].values,
    train_df['encoded_label'].values,
    tokenizer,
    max_length=512
)

val_dataset = LegalDocumentDataset(
    val_df['text'].values,
    val_df['encoded_label'].values,
    tokenizer,
    max_length=512
)

# Create data loaders
batch_size = 16  # Adjust based on your GPU memory

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=RandomSampler(train_dataset),
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=SequentialSampler(val_dataset),
    num_workers=2
)

# Inspect a batch
batch = next(iter(train_loader))
print("\nSample batch inspection:")
print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Attention mask shape: {batch['attention_mask'].shape}")
print(f"Labels shape: {batch['labels'].shape}")

# Sample text decoding
sample_idx = 0
print("\nDecoded sample text (truncated):")
print(tokenizer.decode(batch['input_ids'][sample_idx])[:100] + "...")


#### SECTION 6: MODEL TRAINING ####
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os

# Define training function
def train_model(model, train_loader, val_loader, device, epochs=4, 
                learning_rate=2e-5, warmup_steps=0, weight_decay=0.01,
                save_dir='/content/drive/MyDrive/legal_bert_classification'):
    
    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Calculate total training steps
    total_steps = len(train_loader) * epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Track training metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    best_model_path = os.path.join(save_dir, "best_model.pt")
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update metrics
            total_train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        # Progress bar for validation
        progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch in progress_bar:
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
                total_val_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate accuracy
        accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_preds)
        val_accuracies.append(accuracy)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Save checkpoint to Drive
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': accuracy
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {accuracy:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    
    return model, history

# Load pre-trained model
print("Loading pre-trained BERT model...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',  # Change to 'nlpaueb/legal-bert-base-uncased' for legal BERT
    num_labels=len(label_encoder.classes_),
    output_attentions=False,
    output_hidden_states=False
)

# Move model to device
model.to(device)

# Set training parameters
epochs = 4
learning_rate = 2e-5
warmup_steps = 0
weight_decay = 0.01

# Train the model
trained_model, history = train_model(
    model, 
    train_loader, 
    val_loader, 
    device, 
    epochs=epochs, 
    learning_rate=learning_rate, 
    warmup_steps=warmup_steps, 
    weight_decay=weight_decay
)

# Save the final model
model_save_path = '/content/drive/MyDrive/legal_bert_classification/final_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Final model saved to {model_save_path}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_losses'], label='Train Loss')
plt.plot(history['val_losses'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_accuracies'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/legal_bert_classification/training_history.png')
plt.show()


#### SECTION 7: MODEL EVALUATION ####
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluate final model
def evaluate_model(model, dataloader, device, label_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return all_preds, all_labels, report, cm

# Load best model for evaluation
model.load_state_dict(torch.load('/content/drive/MyDrive/legal_bert_classification/best_model.pt'))
model.to(device)

# Run evaluation
predictions, true_labels, report, cm = evaluate_model(
    model, val_loader, device, label_encoder.classes_
)

# Print classification report
print("Classification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/legal_bert_classification/confusion_matrix.png')
plt.show()

# Calculate per-class accuracy
class_accuracy = {}
for i, label in enumerate(label_encoder.classes_):
    class_mask = [tl == i for tl in true_labels]
    class_true = sum(class_mask)
    class_correct = sum(1 for p, tl in zip(predictions, true_labels) if p == tl and tl == i)
    class_accuracy[label] = class_correct / class_true if class_true > 0 else 0

# Plot per-class accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_accuracy.keys()), y=list(class_accuracy.values()))
plt.title('Accuracy by Document Type')
plt.xlabel('Document Type')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
for i, val in enumerate(class_accuracy.values()):
    plt.text(i, val + 0.01, f'{val:.4f}', ha='center')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/legal_bert_classification/class_accuracy.png')
plt.show()


#### SECTION 8: INFERENCE WITH NEW DATA ####
def predict_document_type(text, model, tokenizer, label_encoder, device, max_length=512):
    """Predict document type for a new text."""
    # Prepare the text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get predicted class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    
    # Convert to label
    predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
    
    return {
        'predicted_label': predicted_label,
        'confidence': confidence.item(),
        'probabilities': {
            label: prob.item() 
            for label, prob in zip(label_encoder.classes_, probabilities[0].cpu().numpy())
        }
    }

# Example usage
sample_text = """REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
of 27 April 2016
on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation)
(Text with EEA relevance)
THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,
Having regard to the Treaty on the Functioning of the European Union, and in particular Article 16 thereof,
Having regard to the proposal from the European Commission,
After transmission of the draft legislative act to the national parliaments,
Having regard to the opinion of the European Economic and Social Committee,
Having regard to the opinion of the Committee of the Regions,
Acting in accordance with the ordinary legislative procedure,"""

# Make prediction
prediction = predict_document_type(sample_text, model, tokenizer, label_encoder, device)

# Print results
print(f"Predicted document type: {prediction['predicted_label']}")
print(f"Confidence: {prediction['confidence']:.4f}")
print("\nProbability for each class:")
for label, prob in prediction['probabilities'].items():
    print(f"  {label}: {prob:.4f}")


#### SECTION 9: SAVE AND LOAD THE MODEL ####
# Save the model and tokenizer (already done in training section)
# model.save_pretrained('/content/drive/MyDrive/legal_bert_classification/final_model')
# tokenizer.save_pretrained('/content/drive/MyDrive/legal_bert_classification/final_model')

# Example of how to load the model later
from transformers import BertForSequenceClassification, BertTokenizer
import pickle

def load_model_for_inference():
    # Load label encoder
    with open('/content/drive/MyDrive/legal_bert_classification/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load model and tokenizer
    model_path = '/content/drive/MyDrive/legal_bert_classification/final_model'
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, label_encoder, device

# Example of loading and using the model
# model, tokenizer, label_encoder, device = load_model_for_inference()
# prediction = predict_document_type("Your text here", model, tokenizer, label_encoder, device)

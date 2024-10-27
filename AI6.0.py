import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
MAX_LEN = 32
MODEL_SAVE_PATH = 'tense_classifier.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)

class TenseDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=32):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index] if index < len(self.labels) else -1
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def save_model(model, optimizer, filepath=MODEL_SAVE_PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Model saved to {filepath}.")

def load_model(model, optimizer, filepath=MODEL_SAVE_PATH):
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}.")
    except FileNotFoundError:
        print(f"No checkpoint found at {filepath}. Starting from scratch.")

def load_data_from_csv(filepath):
    df = pd.read_csv(filepath)
    sentences = df['sentence'].tolist()
    labels = df['label'].tolist()
    return sentences, labels

model = bert_model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

csv_filepath = 'tense_dataset.csv'
sentences, labels = load_data_from_csv(csv_filepath)

train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sentences, 
    labels + [-1] * (len(sentences) - len(labels)),  # Fill missing labels with -1
    test_size=0.2, 
    random_state=42
)

train_dataset = TenseDataset(train_sentences, train_labels, tokenizer, max_length=MAX_LEN)
val_dataset = TenseDataset(val_sentences, val_labels, tokenizer, max_length=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

total_steps = EPOCHS * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0,
    num_training_steps=total_steps
)

load_model(model, optimizer)

def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, scheduler):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        num_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()

                mask = labels != -1  # Ignore -1 labels
                correct_predictions += torch.sum((torch.max(outputs.logits, dim=1).indices == labels) * mask)
                num_samples += torch.sum(mask)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions.double() / num_samples
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    save_model(model, optimizer)

train_model(model, train_loader, val_loader, optimizer, loss_fn, EPOCHS, scheduler)

label_map = {"Past": 0, "Present": 1, "Future": 2}
label_map_reverse = {0: "Past", 1: "Present", 2: "Future"}

def run_bot():
    print("Welcome to the TenseSense!")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("Enter a sentence: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            print("MADE BY SUHANI VERMA")
            print("MADE BY TAPASYA")
            break

        encoding = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
        
        predicted_tense = label_map_reverse[prediction.item()]
        print(f"The sentence is in the {predicted_tense} tense.")
        
        correct_tense = input("Is this correct? (yes/no): ").strip().lower()
        if correct_tense == 'no':
            correct_label = input("Please provide the correct tense (Past/Present/Future): ").strip().title()
            
            if correct_label not in label_map:
                print("Invalid tense. Please enter one of the following: Past, Present, Future.")
                continue
            
            correct_label_idx = label_map[correct_label]
            correct_label_tensor = torch.tensor([correct_label_idx]).to(DEVICE)
            # Train on the mistake
            loss = train_on_mistake(model, optimizer, loss_fn, input_ids, attention_mask, correct_label_tensor)
            print(f"Model updated with loss: {loss:.4f}")
            # Save the model state
            save_model(model, optimizer)
        print()

def train_on_mistake(model, optimizer, loss_fn, input_ids, attention_mask, label):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = loss_fn(outputs.logits, label)
    loss.backward()
    optimizer.step()
    return loss.item()

run_bot()

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split

# Define the GPT2Dataset class
class GPT2Dataset(Dataset):
    def __init__(self, data_points, tokenizer, max_length=1024):
        self.data_points = data_points
        self.tokenizer = tokenizer
        self.max_length = max_length

        
    
        # Add a padding token to the tokenizer
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'

    
        
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, index):
        item = self.data_points[index]
        encoding = self.tokenizer.encode_plus(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()}



def tokenize_data_points(data_points, tokenizer, max_length):
    tokenized_data_points = []
    for point in data_points:
        text = point.strip()
        if text:
            encoding = tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tokenized_data_points.append({'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']})
    return tokenized_data_points

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data and split into training and validation sets
    with open('/fs/classhomes/spring2023/cmsc742/c7420004/gpt3_responses.txt', 'r') as f:
        data_points = f.readlines()
        print(data_points)
    train_data, val_data = train_test_split(data_points, test_size=0.1, random_state=42)
    
    # Initialize tokenizer and tokenize data points
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 1024
    train_tokenized = tokenize_data_points(train_data, tokenizer, max_length)
    val_tokenized = tokenize_data_points(val_data, tokenizer, max_length)
    
    # Initialize datasets and dataloaders
    train_dataset = GPT2Dataset(train_tokenized, tokenizer, max_length)
    val_dataset = GPT2Dataset(val_tokenized, tokenizer, max_length)
    batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda data: {'input_ids': torch.stack([item['input_ids'] for item in data]),
                                                                                                             'attention_mask': torch.stack([item['attention_mask'] for item in data])})
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda data: {'input_ids': torch.stack([item['input_ids'] for item in data]),
                                                                                                           'attention_mask': torch.stack([item['attention_mask'] for item in data])})

    # Initialize model
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.to(device)
    
    # Initialize training parameters
    num_epochs = 3
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(**batch, labels=batch["input_ids"])

            # Calculate the loss
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Clip the gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update the weights
            optimizer.step()

            # Update the training loss
            train_loss += loss.item()
    
    # Update the learning rate
    scheduler.step()

    # Calculate the average training loss for the epoch
    train_loss /= len(train_dataloader)

    # Print the training loss every 100 batches
    if (batch_idx + 1) % 100 == 0:
        print(f"Batch {batch_idx+1}/{len(train_dataloader)} - loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])

            # Compute loss
            loss = outputs.loss

            # Update evaluation loss
            eval_loss += loss.item()

        # Calculate the average evaluation loss for the epoch
        eval_loss /= len(val_dataloader)

        # Print the average evaluation loss for the epoch
        print(f"Epoch {epoch+1} - evaluation loss: {eval_loss:.4f}")

        # Save the model checkpoint
        checkpoint_path = f"gpt2-medium_epoch{epoch+1}_loss{eval_loss:.4f}.pt"
        torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    main()


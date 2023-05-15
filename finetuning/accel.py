import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import os
from accelerate import Accelerator

accelerator = Accelerator()



df = pd.read_csv("data/crawl-data-2023-jan-feb-1-5.csv")
i = 0
dropping = []
print(len(df['Text']))
for row in df['Text']:
  try:
    if len(row.split(' ')) > 350:
      dropping.append(i)
  except:
    dropping.append(i)
  i += 1
df = df.drop(dropping)
print(len(df['Text']))
print(df.head())

test_set = df.sample(n=10)
df = df.loc[~df.index.isin(test_set.index)]
test_set['Ground_truth'] = test_set['Text'].str.split().str[-20:].apply(' '.join)
test_set['Text'] = test_set['Text'].str.split().str[:-20].apply(' '.join)

df = df.drop(columns=['Unnamed: 0'])
print(df.head())

class CommonCrawl(Dataset):  
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.text = []

        for row in df['Text']:
          tokens = self.tokenizer.encode(f"<|{control_code}|>{row}", truncation=True, max_length=max_length)
          self.text.append(torch.tensor(tokens))

        if truncate:
            self.text = self.text[:20000]
        self.text_count = len(self.text)
        
    def __len__(self):
        return self.text_count

    def __getitem__(self, item):
        return self.text[item] 

gpt2_size = "gpt2"
#Create dataset
dataset = CommonCrawl(df['Text'], truncate=True, gpt2_type=gpt2_size)

#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_size)
model = GPT2LMHeadModel.from_pretrained(gpt2_size)

# Optimizer
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
train_dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
    dataset, model, tokenizer, optimizer, train_dataloader,
    batch_size=256, epochs=3, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir="checkpoints/", output_prefix="pynonedidit",
    test_mode=False,save_model_on_epoch=False,
):
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    model.train()
    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            accelerator.backward(loss)

            if (idx+1) % batch_size == 0 or idx == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Save a checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
        }
        torch.save(checkpoint, os.path.join(output_dir, f"{output_prefix}-checkpoint-{epoch}.pt"))

        if save_model_on_epoch:
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model

model = train(dataset, model, tokenizer, optimizer=optimizer, train_dataloader=train_dataloader)


# RESUME CHECKPOINTS

# checkpoint = torch.load("path_to_your_checkpoint.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# model.train()
# # Continue training...

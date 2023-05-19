from transformers import GPT2LMHeadModel, GPT2Tokenizer
from accelerate import Accelerator
import torch
import os
import pandas as pd

# Initialize accelerator
accelerator = Accelerator()

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Path to the directory containing the checkpoint
checkpoint_dir = "/fs/class-projects/spring2023/cmsc742/c7420g00/epoch_8"

# Load model weights
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{checkpoint_dir}/pytorch_model.bin")))

# Prepare the model and the tokenizer for acceleration
model, tokenizer = accelerator.prepare(model, tokenizer)

# Prepare a prompt
# prompt = "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor. The statue was a gift from the people of France \
#     and is of a female figure representing Libertas, the Roman goddess of freedom.It was dedicated on October 28, 1886. The statue has become an icon of \
#     freedom and democracy and is one of the most famous landmarks in the United States."
# outputs = []
prompts_file = "crawl-data-2023-5-10_new.csv"
df = pd.read_csv(prompts_file)
df = df.drop(columns=['Unnamed: 0'])
df_new = df.sample(n=100)
prompts = df_new["Text"].to_list()
full_outputs = []
generated_parts = []

# Tokenize the input
# input_ids = tokenizer(prompt, max_length=500, padding=True, truncation=True)
for prompt in prompts:
    if prompt[0] == '"':
         prompt = prompt[1:]
    if prompt[-1] == '"':
         prompt = prompt[:-1]
    # input_ids = tokenizer.encode(prompt, max_length=200, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
    tokens = tokenizer(prompt, max_length=200, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    # Generate a sequence
    max_length = 400
    temperature = 0.95
    output_sequence = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=model.config.pad_token_id
    )

    output_sequence = output_sequence.cpu().tolist()
    for generated_sequence in output_sequence:
        # generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # print(text)
        full_outputs.append(text)
        gp = text[len(prompt):]
        generated_parts.append(gp)

df_new["Generated_Text"] = generated_parts
df_new["Full_Text"] = full_outputs
print(len(full_outputs))
df_new.to_csv(f"attention_generated-{prompts_file[:-4]}.csv")

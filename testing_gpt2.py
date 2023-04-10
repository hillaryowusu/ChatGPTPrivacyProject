from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline


model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)




generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device= -1)  # Use device=0 for GPU

text = "Once upon a time"
generated_text = generator(text, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

print(generated_text[0]['generated_text'])




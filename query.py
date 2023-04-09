import openai

with open('wiki.train.tokens', 'r') as f:
    text = f.read()
    text = text.replace('\n', '<eos>')

# Split the text into sentences
sentences = text.split('.')


# Select the first 100 sentences as data points
data_points = sentences[:100]

# Set your API key here
openai.api_key = "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"

for data_point in data_points:
    # Construct prompt
    prompt = f"Is the following text part of the training set?\n\n{data_point}\n\nAnswer:"

    # Generate completion
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Analyze the response to infer membership
    response_text = response.choices[0].text.strip()
    if "yes" in response_text.lower() or "likely" in response_text.lower():
        print(f"The data point '{data_point}' was likely in the training set.")
    else:
        print(f"The data point '{data_point}' may not have been in the training set.")

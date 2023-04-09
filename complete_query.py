import openai
import io
import sys

with open('wiki.train.tokens', 'r') as f:
    text = f.read()
    text = text.replace('\n', '<eos>')

# Split the text into sentences
sentences = text.split('.')
responses = []

# Select the first 100 sentences as data points
data_points = sentences[:100]

# Set your API key here
openai.api_key = "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"

# Function to capture printed statements
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

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

    # Capture the printed statements and append them to the responses list
    with Capturing() as output:
        if "yes" in response_text.lower() or "likely" in response_text.lower():
            print(f"The data point '{data_point}' was likely in the training set.")
        else:
            print(f"The data point '{data_point}' may not have been in the training set.")
    responses.extend(output)

# Save the responses to a text file
with open("gpt3_responses.txt", "w") as output_file:
    for response in responses:
        output_file.write(response + "\n")

print("Process complete. Check 'gpt3_responses.txt' for results.")

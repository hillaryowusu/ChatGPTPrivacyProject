#getting started. ChatGPT is using the model architecture of GPT-3 and hence, performing tests on GPT-2 is not really helpful. You need an API access key for GPT-3

#pip install openai
# i cretaed a virtual environment because of permission issues
#this is just a test of how we can perform the membership inference. (that's my personal API Key).

import openai
>>>
>>> openai.api_key = "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"
>>>
>>> model_engine = "text-davinci-002"
>>>
>>> data_point = "To be, or not to be, that is the question."
>>>
>>> prompt = f"Was the following data point in the training set of GPT-3? Data point: '{data_point}'"
>>>
>>> response = openai.Completion.create(
...     engine=model_engine,
...     prompt=prompt,
...     max_tokens=50,
...     n=1,
...     stop=None,
...     temperature=0.5,
... )
>>>
>>> response_text = response.choices[0].text.strip()
>>>
>>> # Analyze the response to infer membership
... if "yes" in response_text.lower() or "likely" in response_text.lower():
...     print(f"The data point '{data_point}' was likely in the training set.")
... else:
...     print(f"The data point '{data_point}' may not have been in the training set.")

# ChatGPTPrivacyProject

This code is for our project: Assessing the vulnearibilies of ChatGPT against membership inference attacks & proposing defective strategies. For the course timeline, we focused on the phase one which was assessing the vulnetabilities of ChatGPT.

Due to inaccessibility og gpt-3 model. We used a fine-tuned gpt-2 model. Find info regarding the finetuning in the folder "finetuning".
Datasets used for training the attack model can be accessed through this link: https://drive.google.com/drive/folders/1_H3M7wKD0EbeSZOTeevD5WIcqnL4396X?usp=share_link

Weights for attack model can also be accessed via: https://drive.google.com/drive/folders/1eGVTtXWagyvgKaItRS0C02biLTNfWopp?usp=share_link

Make sure to call the checkpoint when loading the finetuned model.
Also include the saved weights for attack model, vectorizer and scaler when performing an inference attack

General steps:
1. Synthesize your data for finetuning gpt-2 model as well as training attack model.
2. Finetune the gpt-2 model
3. Develop and train an attacker model using known positive and negative examples
4. Use attack model to perform inference on finetuned gpt-2 model.
5. Improve quality of dataset to enhance quality of results.


NB: Our inference results are skewed due to biased data and complexity of GPT-3 architecture. Further study and tests will be ran to improve upon our results.

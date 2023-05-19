# GPT-2 Fine-tuning

See `requirements.txt` for necessary libraries for GPT-2 Fine-tuning.

Datasets used for fine-tuning: https://drive.google.com/drive/u/1/folders/1lQlSOpfjGXdHiblm6flNu3zBAve6uJRc

Model Weights: https://drive.google.com/drive/u/2/folders/1Df-UiLJJZwDxQWuIAfr-y2QZN1sVKIhA

To generate prompts with the fine-tuned model check `generate.py`

To replicate the fine-tuning process or to fine-tune further with more Common Crawl Data:
1. Process data according to `preprocess.py`. Note: For data before 2021, use V1 and for data after 2021, use v2.
2. Install venv with `requirements.txt`. Make sure you have GPU.
3. Run `sbatch run.sh run_train.sh` 

To download data from Common Crawl: run `downloader.sh -i <path-to-wet-paths> -o <path-to-save-outputs> -s <start-file-from-wet-paths> -e <end-file-from-wet-paths>`
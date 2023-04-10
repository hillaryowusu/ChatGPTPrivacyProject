import requests
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import re
import nltk
from nltk.stem import WordNetLemmatizer
import shutil


nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def clean_text(text):
    # Remove URLs, special characters, and extra whitespace
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    # Tokenize, lowercase, and lemmatize words
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

import warc
import tempfile
import gzip

from warcio.archiveiterator import ArchiveIterator

def process_warc_file(warc_url):
    extracted_data = []

    response = requests.get(warc_url, stream=True)

    for record in ArchiveIterator(response.raw, arc2warc=True):
        if record.rec_type == 'response':
            content = record.content_stream().read()
            extracted_data.append(content.decode('utf-8', errors='ignore'))

    return extracted_data







warc_file_url = "https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2021-39/segments/1632487483149.16/warc/CC-MAIN-20210927122325-20210927142325-00727.warc.gz"

data = process_warc_file(warc_file_url)

print("Extracted data:", data)

for record in data:
    print(record)




output_file = "extracted_data.txt"

with open(output_file, "w") as f:
    for item in data:
        f.write(f"{item}\n")

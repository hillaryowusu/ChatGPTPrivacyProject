import os
import csv
import pandas as pd
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
def filter_common_crawl_v1(year, idx):
    prefix = f'./data/{year}'
    months = os.listdir(prefix)
    if '.DS_Store' in months:
        months.remove(".DS_Store")
    out_path = f'./data/crawl-data/crawl-data-{year}-{idx}.csv'
    final_eng = []
    nlp = spacy.load("en_core_web_sm")
    # nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)
    data = {'Text': []}
    print(f"Compiling crawl data from {year}...")
    print(f"Output path: {out_path}")

    for m in months:
        # open wet file and read lines
        print(f"Month: {m}")
        filename = f'{year}_{m}_wet.paths'
        file_paths = f'./data/wet-paths/{filename}'
        print(f"wet paths: {file_paths}")
        fopen = open(file_paths, 'r')
        lines = fopen.readlines()

        # for i in range(start - 1, end):
        # open crawl file
        fname = lines[idx].split('/')[-1]
        file_path = f'{prefix}/{m}/{fname[:-4]}'
        print(f"crawl file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        blocks = content.split("WARC/1.0")
        for i in range(len(blocks)):
            print(i)
            block = blocks[i]
            _block = block.strip().split('\n')
            if len(_block) < 10:
                continue
            text_block = block.split(_block[9])[-1].strip()
            doc = nlp(text_block)
            detect_language = doc._.language
            if detect_language['language'] == 'en' and detect_language['score'] > 0.98:
                block_lines = text_block.split("\n")
                new_block = ''
                for bl in block_lines:
                    if len(bl.split()) >= 20:
                        new_block += bl + '\n'
                if len(new_block) > 0:
                    final_eng.append(new_block)
    text_len = 300
    for bl in final_eng:
        words = bl.strip().split()
        if len(words) <= text_len:
            data['Text'].append(bl.strip())
        else:
            breaker = len(words) // text_len
            start = 0
            end = text_len
            for i in range(breaker):
                text = ' '.join(words[start + (i * text_len): end + (i * text_len)])
                data['Text'].append(text)
    print(f'data points: {len(data["Text"])}')
    df = pd.DataFrame.from_dict(data)
    df.to_csv(out_path)

def filter_common_crawl_v2(year, idx):
    prefix = f'/content/gdrive/MyDrive/CommonCrawl/{year}'
    months = os.listdir(prefix)
    if '.DS_Store' in months:
        months.remove(".DS_Store")
    out_path = f'/content/gdrive/MyDrive/CommonCrawl/crawl-data/negative-crawl-data-{year}.csv'#{idx}.csv'
    final_eng = []
    data = {'Text': []}
    print(f"Compiling crawl data from {year}...")
    print(f"Output path: {out_path}")

    for m in months:
        # open wet file and read lines
        print(f"Month: {m}")
        filename = f'{year}_{m}_wet.paths'
        file_paths = f'/content/gdrive/MyDrive/CommonCrawl/wet-paths/{filename}'
        print(f"wet paths: {file_paths}")

        fopen = open(file_paths, 'r')
        lines = fopen.readlines()

        for i in range(idx):
          fname = lines[i].split('/')[-1]
          # fname = lines[idx].split('/')[-1]
          file_path = f'{prefix}/{m}/{fname[:-4]}'
          print(f"crawl file: {file_path}")
          with open(file_path, "r", encoding="utf-8") as file:
              content = file.read()
          blocks = content.split("WARC/1.0")
          for i in range(len(blocks)):
              # print(f"\nBlock {i}:")
              block = blocks[i]
              _block = block.strip().split('\n')
              if len(_block) < 8 :
                  continue
              if _block[6].strip() == "WARC-Identified-Content-Language: eng":
                  block = block.split(_block[8])[-1].strip()
                  block_lines = block.split("\n")
                  new_block = ''
                  for bl in block_lines:
                      if len(bl.split()) >= 50:
                          new_block += bl #+ '\n'
                  if len(new_block) > 0:
                      final_eng.append(new_block)

    text_len = 100
    for bl in final_eng:
        words = bl.strip().split()
        if len(words) <= text_len:
            data['Text'].append(bl.strip())
        else:
            breaker = len(words) // text_len
            start = 0
            end = text_len
            for i in range(breaker):
                text = ' '.join(words[start + (i * text_len): end + (i * text_len)])
                if text[0] != '"':
                  text = '"' + text.strip()
                if text[-1] != '"':
                  text = text + '"'
                data['Text'].append(text.strip())
    
    df = pd.DataFrame.from_dict(data)
    df['Non Alphanumeric'] = df['Text'].str.findall(r'[^a-zA-Z0-9 ]').str.len()
    # return df
    df = df[df['Non Alphanumeric'] <= 200]
    with open(out_path, 'w', encoding='UTF8') as f:
      # create the csv writer
      writer = csv.writer(f)

      # write a row to the csv file
      writer.writerow(['Text'])
      for t in data['Text']:
        try:
          writer.writerow([t])
        except:
          pass
    df.to_csv(out_path)
import requests
from bs4 import BeautifulSoup
import re

def search_google(query, num_results=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
    }
    query = query.replace(' ', '+')
    url = f'https://www.google.com/search?q={query}&num={num_results}'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.find_all('div', class_='tF2Cxc')
    return search_results

def extract_links(search_results):
    links = []
    for result in search_results:
        link = result.find('a')['href']
        links.append(link)
    return links

def main():
    query = "GPT-3 training data dataset sources"
    search_results = search_google(query)
    links = extract_links(search_results)

    for link in links:
        print(link)

if __name__ == "__main__":
    main()

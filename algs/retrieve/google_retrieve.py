import requests
import json
import re
from bs4 import BeautifulSoup

API_KEY = ""
SEARCH_ENGINE_ID_ENTIREWEB = ""
SEARCH_ENGINE_ID_ACADEMIA = ""

def build_payload_entireweb(query, start=1, num=10):
    payload = {
        'key': API_KEY,
        'q': query,
        'cx': SEARCH_ENGINE_ID_ENTIREWEB,
        'start': start,
        'num': num,
        'lr': 'lang_en',
        'hl': 'en'
    }
    return payload

def build_payload_academic(query, start=1, num=10):
    payload = {
        'key': API_KEY,
        'q': query,
        'cx': SEARCH_ENGINE_ID_ACADEMIA,
        'start': start,
        'num': num,
        'lr': 'lang_en',
        'hl': 'en'
    }
    return payload

def make_request(payload):
    response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
    if response.status_code != 200:
        raise Exception('Request failed')
    return response.json()

def process_text(text):
    text = re.sub(r'[\n\t]+', ' ', text)
    text = text.replace("â", "'").replace(" ", " ")
    return text.strip()

def combine_adjacent_paragraphs(soup, tag):
    paragraphs = soup.find_all(tag)
    combined_paragraphs = []
    current_paragraph = ""

    # Iterate over all tag elements
    for i in range(len(paragraphs)):
        if i == 0:
            current_paragraph = paragraphs[i].get_text()
        else:
            # Check if current <p> tag is directly following the tag
            if paragraphs[i].previous_sibling == paragraphs[i - 1] :
                current_paragraph += " " + paragraphs[i].get_text()
            else:
                combined_paragraphs.append(process_text(current_paragraph))
                current_paragraph = paragraphs[i].get_text()

    # Append the last accumulated paragraph
    if current_paragraph:
        combined_paragraphs.append(process_text(current_paragraph))

    # remove duplicate elements, empty string, and too short strong
    min_word_num = 3
    seen = set()
    cleaned_paragraphs = [s for s in combined_paragraphs if s and len(s.split()) >= min_word_num and not (s in seen or seen.add(s))]

    return cleaned_paragraphs

def count_occurrence(query):
    print(query)
    payload = build_payload_academic(query)
    response = make_request(payload)
    # print(response)
    # print(f"total results: {response['searchInformation']['totalResults']}", type(response["searchInformation"]["totalResults"]))
    return int(response["searchInformation"]["totalResults"])


def deep_retrieve_by_google_academic(query: str, extracted_docs: dict, pages_per_query: int) -> dict:
    print(query)
    all_retrieved_results = []
    for start_idx in range(1, pages_per_query*10, 10):
        payload = build_payload_academic(query, start=start_idx, num=10)
        response = make_request(payload)
        # print(response)
        print(f"total results: {response['searchInformation']['totalResults']}")
        if response["searchInformation"]["totalResults"] == "0":
            break
        all_retrieved_results += response["items"]


    # deeper extract documents
    tags_of_interest = ['p']  # , 'li', 'h3', 'h2', 'h1']
    for item in all_retrieved_results:
        url = item['link']
        if ".pdf" in url:
            continue
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"}
        print(url)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in tags_of_interest:
                combined_paragraphs = combine_adjacent_paragraphs(soup, tag)
                if url not in extracted_docs and len(combined_paragraphs) > 0:
                    extracted_docs[url] = combined_paragraphs

        except requests.exceptions.HTTPError as http_err:
            # Specific errors for HTTP issues
            print(f'HTTP error occurred: {http_err}')
        except requests.exceptions.ConnectionError as conn_err:
            # Handle connection-related issues, e.g., DNS failure, refused connection.
            print(f'Connection error occurred: {conn_err}')
        except Exception as e:
            # Handle other unforeseen errors
            print(f'An unexpected error occurred: {e}')

    return extracted_docs

def deep_retrieve_by_google(query: str, extracted_docs: dict, pages_per_query: int) -> dict:
    print(query)
    all_retrieved_results = []
    for start_idx in range(1, pages_per_query*10, 10):
        payload = build_payload_entireweb(query, start=start_idx, num=10)
        response = make_request(payload)
        # print(response)
        totoal_results = int(response['searchInformation']['totalResults'])
        print(f"total results: {totoal_results}")
        if totoal_results == 0:
            break
        all_retrieved_results += response["items"]
        if totoal_results < start_idx+10:
            break

    # deeper extract documents
    tags_of_interest = ['p']  # , 'li', 'h3', 'h2', 'h1']
    for item in all_retrieved_results:
        url = item['link']
        if ".pdf" in url:
            continue
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"}
        print(url)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in tags_of_interest:
                combined_paragraphs = combine_adjacent_paragraphs(soup, tag)
                if url not in extracted_docs and len(combined_paragraphs) > 0:
                    extracted_docs[url] = combined_paragraphs

        except requests.exceptions.HTTPError as http_err:
            # Specific errors for HTTP issues
            print(f'HTTP error occurred: {http_err}')
        except requests.exceptions.ConnectionError as conn_err:
            # Handle connection-related issues, e.g., DNS failure, refused connection.
            print(f'Connection error occurred: {conn_err}')
        except Exception as e:
            # Handle other unforeseen errors
            print(f'An unexpected error occurred: {e}')

    return extracted_docs
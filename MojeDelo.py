import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def fetch_content(url):
    """Fetch content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def get_pages(curr_page, dct={}):
    content = fetch_content(curr_page)
    if content is None:
        return dct
    soup = BeautifulSoup(content, "html.parser")
    dct[curr_page] = soup
    parent_li = soup.find("li", {"class": "PagedList-skipToNext"})
    if parent_li is None or "PagedList-skipToNext" in parent_li.get("class"):
        return dct
    a_elem = parent_li.find("a")
    next_page_url = urljoin(curr_page, a_elem.get("href"))
    return get_pages(next_page_url, dct)


def get_jobs(main_page):
    dct = {}
    for url, page in get_pages(main_page).items():
        for x in page.find_all("a", {"class": "details overlayOnHover1"}):
            new_url = urljoin(url, x.get("href"))
            content = fetch_content(new_url)
            if content is None:
                continue
            dct[new_url] = content
    return dct


def find_similar_sublink(primary_url, target_text):
    urls = []
    documents = [target_text]
    for url, sublink_content in get_jobs(primary_url).items():
        sub_soup = BeautifulSoup(sublink_content, 'html.parser')
        text = ' '.join(sub_soup.stripped_strings)
        documents.append(text)
        urls.append(url)

    if len(documents) > 1:
        # Compute TF-IDF vectors
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        # Compute cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        max_similarity_index = np.argmax(cosine_similarities)

        if cosine_similarities[0, max_similarity_index] > 0:
            return f"Most similar sublink: {urls[max_similarity_index]}"
        else:
            return "No similar text found in any sublink."
    else:
        return "No sublinks with valid content found."


# Example usage
primary_url = ''
target_text = """
V Rochu zaposlujemo prek 100.000 ljudi, ki skupaj delujejo v več kot 150 državah. Smo vodilno biotehnološko podjetje na svetu, ki se osredotoča na raziskave. Naš uspeh temelji na inovativnosti, radovednosti in raznolikosti.

Iščemo novega sodelavca/ko za organizacijo poslovanja in dogodkov, ki bo skupaj s kolegi ustvarjal odlične izkušnje za naše stranke.
"""
# result = find_similar_sublink(primary_url, target_text)
# print(result)
print(find_similar_sublink("https://www.mojedelo.com/prosta-delovna-mesta/administracija/gorenjska", target_text))
# print(find_similar_sublink(primary_url, target_text))

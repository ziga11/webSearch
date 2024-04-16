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
    global next_page_url
    content = fetch_content(curr_page)
    if content is None:
        return dct  # Stop if the page content could not be fetched
    soup = BeautifulSoup(content, "html.parser")
    dct[curr_page] = soup
    found = False
    for x in reversed(soup.find_all("a")):
        if x.get_text().strip() == "Naslednja stran":  # Ensure to strip any surrounding whitespace
            next_page_url = urljoin(curr_page, x.get('href'))
            found = True
            break
    if not found:
        return dct
    return get_pages(next_page_url, dct)  # Recursive call to fetch the next page


def get_article_contents(main_page):
    page_info_dct = get_pages(main_page)
    dct = {}
    for url, content in page_info_dct.items():
        for article in content.find_all("article"):
            for a in article.find_all("a"):
                a_url = urljoin(url, a.get('href'))
                content = fetch_content(a_url)
                if a_url is None or not content:
                    continue
                dct[a_url] = content
    return dct


def find_similar_sublink(primary_url, target_text):
    urls = []
    documents = [target_text]
    for url, sublink_content in get_article_contents(primary_url).items():
        sub_soup = BeautifulSoup(sublink_content, 'html.parser')
        print(sub_soup)
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
primary_url = 'https://www.careerjet.si/delovna-mesta-komunala.html?p='
target_text = """
V Rochu zaposlujemo prek 100.000 ljudi, ki skupaj delujejo v več kot 150 državah. Smo vodilno biotehnološko podjetje na svetu, ki se osredotoča na raziskave. Naš uspeh temelji na inovativnosti, radovednosti in raznolikosti.

Iščemo novega sodelavca/ko za organizacijo poslovanja in dogodkov, ki bo skupaj s kolegi ustvarjal odlične izkušnje za naše stranke.
"""
# result = find_similar_sublink(primary_url, target_text)
# print(result)
print(find_similar_sublink(primary_url, target_text))

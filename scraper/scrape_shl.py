# scraper/scrape_shl.py
import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/?start=48&type=1&type=1"

results = []

def get_catalog_links():
    r = requests.get(CATALOG_URL, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if href and "/products/" in href and "job-solutions" not in href:
            if href.startswith("/"):
                href = BASE_URL + href
            links.append(href)

    return list(set(links))

def scrape_product(url):
    r = requests.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    name = soup.find("h1")
    desc = soup.find("meta", {"name": "description"})
    test_type = soup.find("span", {"class": "product-catalogue__key"})
    adaptive_support = soup.find("span", {"class": "adaptive_support"})
    duration = soup.find("p", {"class": "duration"})
    remote_support = soup.find("span", {"class": "remote_support"})
    
    if adaptive_support:
        adaptive_support = adaptive_support.text.strip()
    if duration:
        duration = duration.text.strip()
    if remote_support:
        remote_support = remote_support.text.strip()


    return {
        "name": name.text.strip() if name else "",
        "url": url,
        "description": desc["content"] if desc else "",
        "test_type": test_type.text.strip() if test_type else "",
        "adaptive_support": adaptive_support,
        "duration": duration,
        "remote_support": remote_support
    }

def main():
    links = get_catalog_links()
    print(f"Found {len(links)} links")

    for i, link in enumerate(links):
        try:
            data = scrape_product(link)
            results.append(data)
            print(f"[{i+1}/{len(links)}] {data['name']}")
            time.sleep(0.5)
        except Exception as e:
            print("Error:", link)

    with open("../data/catalog.json", "a") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

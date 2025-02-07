import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from urllib.parse import urljoin
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

# load env file
load_dotenv()

# initialize the clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def clean_url(url: str) -> str:
    """Remove fragment identifier from URL while preserving query parameters."""
    parsed_url = urlparse(url)
    cleaned_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    if parsed_url.query:
        cleaned_url += f"?{parsed_url.query}"
    return cleaned_url

def get_pydantic_ai_docs_urls(base_url):
    # Fetch robots.txt
    robots_url = urljoin(base_url, '/robots.txt')
    rp = RobotFileParser()
    rp.set_url(robots_url)
    rp.read()

    # Try to fetch sitemap.xml
    sitemap_url = urljoin(base_url, '/sitemap.xml')
    response = requests.get(sitemap_url)
    
    if response.status_code == 200:
        # Sitemap exists, parse XML
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []
        for url in root.findall('.//ns:url', namespace):
            loc = url.find('ns:loc', namespace)
            if loc is not None:
                url = loc.text.strip()
                cleaned_url = clean_url(url)
                if rp.can_fetch("*", cleaned_url):
                    urls.append(cleaned_url)
        print("Sitemap URLs Length: ", len(urls))
        return urls
    else:
        # Sitemap doesn't exist, use web scraping
        return crawl_website(base_url, rp)

def crawl_website(base_url, rp, max_pages=100):
    visited = set()
    to_visit = [base_url]
    urls = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        visited.add(url)

        cleaned_url = clean_url(url)
            
        if rp.can_fetch("*", cleaned_url):
            urls.append(cleaned_url)

            try:
                response = requests.get(cleaned_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        # Clean the full URL before adding to to_visit
                        full_url = urljoin(url, href)
                        clean_full_url = clean_url(full_url)
                            
                        if urlparse(clean_full_url).netloc == urlparse(base_url).netloc:
                            to_visit.append(clean_full_url)
            except requests.RequestException:
                print(f"Failed to fetch {url}")
    print("Crawled URLs Length: ", len(urls))
    return urls
    
    

async def main() -> None:
    # lets get all the URLs required to scrape from the base URL
    urls = get_pydantic_ai_docs_urls("https://docs.crawl4ai.com/")
    
    print(urls)

if __name__ == "__main__":
    asyncio.run(main())

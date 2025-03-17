import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class SiteMapCrawler:
    def __init__(self, base_url):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.urls_to_visit = deque([base_url])
        self.sitemap = set()

    def is_valid_url(self, url):
        """Check if the URL belongs to the same domain and is not already visited."""
        parsed_url = urlparse(url)
        return parsed_url.netloc == self.domain and url not in self.visited_urls

    def extract_links(self, url):
        """Extract all links from a webpage."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            links = set()
            for anchor in soup.find_all("a", href=True):
                link = urljoin(url, anchor["href"])
                if self.is_valid_url(link):
                    links.add(link)

            return links
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return set()

    def crawl(self, max_urls=100):
        """Crawl the website up to a specified number of URLs."""
        while self.urls_to_visit and len(self.sitemap) < max_urls:
            url = self.urls_to_visit.popleft()
            if url in self.visited_urls:
                continue

            print(f"Crawling: {url}")
            self.visited_urls.add(url)
            self.sitemap.add(url)

            new_links = self.extract_links(url)
            self.urls_to_visit.extend(new_links)

            # Be polite: add a delay between requests
            time.sleep(1)

        print(f"Crawling complete. Found {len(self.sitemap)} URLs.")

    def save_sitemap(self, filename="sitemap.txt"):
        """Save the sitemap to a file."""
        with open(filename, "w") as file:
            for url in sorted(self.sitemap):
                file.write(url + "\n")
        print(f"Sitemap saved to {filename}")


def get_sitemap_urls(base_url: str, sitemap_filename: str = "sitemap.xml") -> List[str]:
    """Fetches and parses a sitemap XML file to extract URLs.

    Args:
        base_url: The base URL of the website
        sitemap_filename: The filename of the sitemap (default: sitemap.xml)

    Returns:
        List of URLs found in the sitemap. If sitemap is not found, returns a list
        containing only the base URL.

    Raises:
        ValueError: If there's an error fetching (except 404) or parsing the sitemap
    """
    try:
        sitemap_url = urljoin(base_url, sitemap_filename)

        response = requests.get(sitemap_url, timeout=10)

        print(response.content.decode("utf-8"))

        # # Return just the base URL if sitemap not found
        if response.status_code == 404:
            return [base_url.rstrip("/")]

        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Handle different XML namespaces that sitemaps might use
        namespaces = (
            {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else ""
        )

        # Extract URLs using namespace if present
        if namespaces:
            urls = [elem.text for elem in root.findall(".//ns:loc", namespaces)]
        else:
            urls = [elem.text for elem in root.findall(".//loc")]

        return urls

    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch sitemap: {str(e)}")
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse sitemap XML: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error processing sitemap: {str(e)}")

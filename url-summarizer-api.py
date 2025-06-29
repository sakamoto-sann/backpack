#!/usr/bin/env python3
"""
URL Summarization API
Takes a URL and returns a 1-sentence summary in JSON format.
"""

import json
import sys
import requests
import re
import logging
import ipaddress
import socket
from typing import Dict, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup


# Constants
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_TIMEOUT = 10
MAX_CONTENT_LENGTH = 1000


# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def is_safe_url(url: str) -> bool:
    """Check if the URL is safe to access (prevents SSRF attacks)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        if not hostname:
            return False
            
        # Block localhost and loopback addresses
        if hostname.lower() in ['localhost', '127.0.0.1', '0.0.0.0']:
            return False
            
        # Resolve hostname to IP and check if it's private
        try:
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            
            # Block private IP ranges
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                return False
                
        except (socket.gaierror, ValueError):
            return False
            
        return True
        
    except Exception:
        return False


def fetch_content(url: str) -> str:
    """Fetch and extract content from a URL with security checks."""
    if not is_safe_url(url):
        raise ValueError("URL is not safe to access (potential SSRF)")
    
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(
            url, 
            headers=headers, 
            timeout=REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Check content length to avoid large downloads
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Content too large")
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Extract title
        title_element = soup.find('title')
        title = title_element.get_text().strip() if title_element else ""
        
        # Extract main content with priority order
        content = ""
        
        # Try to find main content areas first
        main_selectors = [
            'article', 'main', '[role="main"]',
            '.content', '.post-content', '.entry-content',
            '.article-body', '.story-body'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                content = main_content.get_text(separator=' ', strip=True)
                break
        
        # Fallback to common content tags
        if not content:
            content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'div'])
            content = ' '.join([tag.get_text(strip=True) for tag in content_tags if tag.get_text(strip=True)])
        
        # Clean and limit content
        content = re.sub(r'\s+', ' ', content).strip()
        
        return f"{title}|{content}" if title else content
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        raise


def generate_summary(content: str, url: str) -> str:
    """Generate a one-sentence summary from extracted content."""
    if not content:
        return f"This webpage at {url} contains content that could not be extracted."
    
    # Split title and content
    parts = content.split('|', 1)
    title = parts[0] if len(parts) > 1 else ""
    text = parts[1] if len(parts) > 1 else parts[0]
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Limit text length for summary
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "..."
    
    # Create summary based on available content
    if title and text:
        summary_text = f"This webpage titled '{title}' discusses {text[:200]}"
    elif title:
        summary_text = f"This webpage is titled '{title}'"
    elif text:
        summary_text = f"This webpage discusses {text[:200]}"
    else:
        summary_text = f"This webpage at {url} contains content that could not be extracted"
    
    # Ensure it's one sentence
    sentences = re.split(r'[.!?]+', summary_text)
    if sentences and sentences[0].strip():
        summary = sentences[0].strip()
        if not summary.endswith(('.', '!', '?')):
            summary += "."
        return summary
    
    return summary_text + "."


def summarize_url(url: str) -> Dict[str, Any]:
    """Main function to summarize a URL."""
    try:
        # Validate URL format
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"url": url, "error": "Invalid URL format", "status": "error"}
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Fetch content
        content = fetch_content(url)
        
        # Generate summary
        summary = generate_summary(content, url)
        
        return {
            "url": url,
            "summary": summary,
            "status": "success"
        }
        
    except requests.exceptions.Timeout:
        logging.error(f"Timeout for {url}")
        return {"url": url, "error": "Request timeout", "status": "error"}
    except requests.exceptions.ConnectionError:
        logging.error(f"Connection error for {url}")
        return {"url": url, "error": "Connection failed", "status": "error"}
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for {url}: {e}")
        return {"url": url, "error": f"HTTP error: {e.response.status_code}", "status": "error"}
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        return {"url": url, "error": f"Failed to fetch URL: {str(e)}", "status": "error"}
    except ValueError as e:
        logging.error(f"Security/validation error for {url}: {e}")
        return {"url": url, "error": str(e), "status": "error"}
    except Exception as e:
        logging.error(f"Unexpected error for {url}: {e}")
        return {"url": url, "error": "An unexpected processing error occurred", "status": "error"}


def main():
    """Main function for CLI usage."""
    if len(sys.argv) != 2:
        print("Usage: python url-summarizer-api.py <URL>")
        print("Example: python url-summarizer-api.py https://example.com")
        sys.exit(1)
    
    url = sys.argv[1]
    result = summarize_url(url)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
"""
Page Parser Module
Extracts structured data from HTML quiz pages.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class PageParser:
    """
    Parses HTML content to extract quiz-related information.
    """
    
    # Patterns for detecting submit URLs
    SUBMIT_PATTERNS = [
        r'post\s+(?:your\s+)?answer\s+to\s+["\']?([^\s"\'<>]+)',
        r'submit\s+(?:your\s+)?(?:answer\s+)?to\s+["\']?([^\s"\'<>]+)',
        r'send\s+(?:your\s+)?(?:answer\s+)?to\s+["\']?([^\s"\'<>]+)',
        r'api[/\\]submit[^\s"\'<>]*',
        r'endpoint[:\s]+["\']?([^\s"\'<>]+submit[^\s"\'<>]*)',
    ]
    
    # File extension patterns
    FILE_EXTENSIONS = ['.csv', '.tsv', '.xlsx', '.xls', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.zip', '.json']
    
    def __init__(self, base_url: str = ''):
        """
        Initialize parser with base URL for resolving relative links.
        
        Args:
            base_url: Base URL for the page
        """
        self.base_url = base_url
        
    def parse(self, html: str) -> Dict[str, Any]:
        """
        Parse HTML and extract all quiz-related data.
        
        Args:
            html: HTML content to parse
            
        Returns:
            Dictionary with extracted data
        """
        soup = BeautifulSoup(html, 'lxml')
        
        result = {
            'question': self._extract_question(soup, html),
            'submit_url': self._extract_submit_url(soup, html),
            'file_links': self._extract_file_links(soup),
            'embedded_images': self._extract_embedded_images(soup),
            'tables': self._extract_tables(soup),
            'instructions': self._extract_instructions(soup, html),
            'api_endpoints': self._extract_api_endpoints(html),
            'numbers': self._extract_numbers(html),
            'all_links': self._extract_all_links(soup),
        }
        
        logger.info(f"Parsed page: {len(result['file_links'])} files, "
                   f"{len(result['tables'])} tables, submit_url: {result['submit_url']}")
        
        return result
        
    def _extract_question(self, soup: BeautifulSoup, html: str) -> str:
        """Extract the main quiz question text."""
        # Try common question containers
        question_selectors = [
            'div.question', 'div.quiz-question', 'p.question',
            '.question-text', '#question', 'main p', 'article p'
        ]
        
        for selector in question_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if len(text) > 20:  # Reasonable question length
                    return text
        
        # Fall back to extracting from body text
        body = soup.find('body')
        if body:
            # Get all paragraph text
            paragraphs = body.find_all('p')
            questions = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 20 and any(q in text.lower() for q in ['?', 'what', 'find', 'calculate', 'download', 'extract', 'sum', 'compute']):
                    questions.append(text)
            if questions:
                return ' '.join(questions)
        
        # Last resort: get main text content
        return soup.get_text(separator=' ', strip=True)[:2000]
        
    def _extract_submit_url(self, soup: BeautifulSoup, html: str) -> Optional[str]:
        """Extract the answer submission URL."""
        text = html.lower()
        
        # Try regex patterns
        for pattern in self.SUBMIT_PATTERNS:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                url = match.group(1) if match.groups() else match.group(0)
                url = self._resolve_url(url.strip('"\''))
                if self._is_valid_url(url):
                    return url
        
        # Look for links containing "submit" or "answer"
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text(strip=True).lower()
            href_lower = href.lower()
            
            if any(kw in href_lower or kw in link_text for kw in ['submit', 'answer', 'post']):
                resolved = self._resolve_url(href)
                if self._is_valid_url(resolved):
                    return resolved
        
        # Look for form action
        form = soup.find('form')
        if form and form.get('action'):
            return self._resolve_url(form['action'])
            
        # Search for URL patterns in text - prioritize submit URLs
        # First try to find explicit submit URLs
        submit_url_pattern = r'https?://[^\s<>"\']+/submit/[^\s<>"\']*'
        match = re.search(submit_url_pattern, html, re.IGNORECASE)
        if match:
            return match.group(0).rstrip('.,;:')
            
        # Also check for relative submit paths
        relative_submit_pattern = r'/submit/\d+'
        match = re.search(relative_submit_pattern, html)
        if match:
            return self._resolve_url(match.group(0))
            
        # Fallback to answer/api URLs (but not bare /api/ endpoints)
        url_pattern = r'https?://[^\s<>"\']+(?:submit|answer)[^\s<>"\']*'
        match = re.search(url_pattern, html, re.IGNORECASE)
        if match:
            return match.group(0).rstrip('.,;:')
            
        return None
        
    def _extract_file_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract all downloadable file links."""
        file_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            href_lower = href.lower()
            
            # Check if link points to a file
            for ext in self.FILE_EXTENSIONS:
                if href_lower.endswith(ext) or ext + '?' in href_lower:
                    file_links.append({
                        'url': self._resolve_url(href),
                        'text': link.get_text(strip=True),
                        'type': ext.lstrip('.')
                    })
                    break
        
        return file_links
        
    def _extract_embedded_images(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract embedded images (both base64 and URL)."""
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src:
                images.append({
                    'src': src if src.startswith('data:') else self._resolve_url(src),
                    'alt': img.get('alt', ''),
                    'is_base64': src.startswith('data:')
                })
                
        return images
        
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract HTML tables as lists of rows."""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = []
            
            # Extract headers
            headers = []
            for th in table.find_all('th'):
                headers.append(th.get_text(strip=True))
            if headers:
                table_data.append(headers)
            
            # Extract rows
            for tr in table.find_all('tr'):
                row = []
                for td in tr.find_all(['td', 'th']):
                    row.append(td.get_text(strip=True))
                if row and row != headers:
                    table_data.append(row)
                    
            if table_data:
                tables.append(table_data)
                
        return tables
        
    def _extract_instructions(self, soup: BeautifulSoup, html: str) -> List[str]:
        """Extract step-by-step instructions."""
        instructions = []
        
        # Look for ordered lists
        for ol in soup.find_all('ol'):
            for li in ol.find_all('li'):
                text = li.get_text(strip=True)
                if text:
                    instructions.append(text)
                    
        # Look for numbered instructions in text
        numbered_pattern = r'(?:^|\n)\s*(\d+)[.)]\s*([^\n]+)'
        matches = re.findall(numbered_pattern, html)
        for num, text in matches:
            instructions.append(f"{num}. {text.strip()}")
            
        # Look for bullet points
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                text = li.get_text(strip=True)
                if text and len(text) > 10:
                    instructions.append(text)
                    
        return instructions
        
    def _extract_api_endpoints(self, html: str) -> List[str]:
        """Extract API endpoint URLs from text."""
        endpoints = []
        
        # Look for API URLs
        api_pattern = r'https?://[^\s<>"\']+api[^\s<>"\']*'
        matches = re.findall(api_pattern, html, re.IGNORECASE)
        endpoints.extend(matches)
        
        # Look for endpoint mentions
        endpoint_pattern = r'/api/[^\s<>"\']*'
        matches = re.findall(endpoint_pattern, html)
        for match in matches:
            resolved = self._resolve_url(match)
            if resolved not in endpoints:
                endpoints.append(resolved)
                
        return endpoints
        
    def _extract_numbers(self, html: str) -> List[str]:
        """Extract important numbers from text."""
        soup = BeautifulSoup(html, 'lxml')
        text = soup.get_text()
        
        # Extract various number formats
        numbers = []
        
        # Page numbers
        page_pattern = r'page\s*(\d+)'
        numbers.extend(re.findall(page_pattern, text, re.IGNORECASE))
        
        # Column/row references
        col_pattern = r'column\s*["\']?(\w+)["\']?'
        numbers.extend(re.findall(col_pattern, text, re.IGNORECASE))
        
        # Numeric values
        num_pattern = r'\b(\d+(?:\.\d+)?)\b'
        numbers.extend(re.findall(num_pattern, text)[:10])  # Limit to first 10
        
        return list(set(numbers))
        
    def _extract_all_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            links.append({
                'url': self._resolve_url(link['href']),
                'text': link.get_text(strip=True)
            })
            
        return links
        
    def _resolve_url(self, url: str) -> str:
        """Resolve relative URL to absolute."""
        if url.startswith(('http://', 'https://', 'data:')):
            return url
        return urljoin(self.base_url, url)
        
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False


def extract_question_details(text: str) -> Dict[str, Any]:
    """
    Extract detailed question components from text.
    
    Args:
        text: Question text
        
    Returns:
        Dictionary with question details
    """
    details = {
        'raw_text': text,
        'operation': None,
        'target_column': None,
        'page_number': None,
        'file_type': None,
        'output_format': None
    }
    
    # Detect operation
    operations = {
        'sum': ['sum', 'total', 'add'],
        'mean': ['mean', 'average', 'avg'],
        'median': ['median'],
        'max': ['maximum', 'max', 'largest', 'highest'],
        'min': ['minimum', 'min', 'smallest', 'lowest'],
        'count': ['count', 'number of', 'how many'],
        'group': ['group by', 'grouped', 'categories'],
        'plot': ['plot', 'chart', 'graph', 'histogram', 'visualize'],
        'extract': ['extract', 'read', 'get', 'find'],
    }
    
    text_lower = text.lower()
    for op, keywords in operations.items():
        if any(kw in text_lower for kw in keywords):
            details['operation'] = op
            break
    
    # Extract target column
    col_match = re.search(r'(?:column|field)\s*["\']?(\w+)["\']?', text, re.IGNORECASE)
    if col_match:
        details['target_column'] = col_match.group(1)
    
    # Extract page number
    page_match = re.search(r'page\s*(\d+)', text, re.IGNORECASE)
    if page_match:
        details['page_number'] = int(page_match.group(1))
    
    # Detect file type
    for ftype in ['csv', 'pdf', 'xlsx', 'excel', 'image', 'png', 'jpg']:
        if ftype in text_lower:
            details['file_type'] = ftype
            break
    
    # Detect output format
    if 'json' in text_lower:
        details['output_format'] = 'json'
    elif 'base64' in text_lower or 'image' in text_lower:
        details['output_format'] = 'base64'
    elif 'number' in text_lower or 'integer' in text_lower:
        details['output_format'] = 'number'
    else:
        details['output_format'] = 'string'
    
    return details

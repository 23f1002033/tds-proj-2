"""
API Utilities Module
HTTP helpers and Gemini API wrapper.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class APIClient:
    """HTTP client with retry logic."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
    def get(self, url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> requests.Response:
        """GET request with retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
    def post(self, url: str, data: Any = None, json: Any = None, 
             headers: Optional[Dict] = None) -> requests.Response:
        """POST request with retries."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, data=data, json=json, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


class GeminiClient:
    """Wrapper for Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.api_url = api_url or os.getenv('GEMINI_API_URL', 
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent')
        self.max_retries = 3
        
    def call(self, prompt: str, model_args: Optional[Dict] = None) -> Dict[str, Any]:
        """Call Gemini API with prompt."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
            
        headers = {'Content-Type': 'application/json'}
        url = f"{self.api_url}?key={self.api_key}"
        
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': model_args or {'temperature': 0.1, 'maxOutputTokens': 1024}
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                if response.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                response.raise_for_status()
                
                data = response.json()
                text = ''
                if 'candidates' in data and data['candidates']:
                    parts = data['candidates'][0].get('content', {}).get('parts', [])
                    text = ' '.join(p.get('text', '') for p in parts)
                    
                return {'raw': data, 'text': text}
                
            except requests.RequestException as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
    def summarize(self, text: str) -> str:
        """Summarize text using Gemini."""
        try:
            result = self.call(f"Summarize this concisely:\n\n{text[:4000]}")
            return result.get('text', '')
        except Exception as e:
            logger.error(f"Summarize failed: {e}")
            return ""
            
    def extract_answer(self, question: str, context: str) -> str:
        """Extract answer from context for a question."""
        try:
            prompt = f"Question: {question}\n\nContext: {context[:4000]}\n\nAnswer concisely:"
            result = self.call(prompt)
            return result.get('text', '')
        except Exception as e:
            logger.error(f"Extract answer failed: {e}")
            return ""

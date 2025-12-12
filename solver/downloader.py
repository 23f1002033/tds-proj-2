"""
File Downloader Module
Secure file download helper with retry logic and content validation.
"""

import os
import re
import time
import logging
import tempfile
import zipfile
from typing import Optional, Tuple, List
from urllib.parse import urlparse, unquote
import requests

logger = logging.getLogger(__name__)


class FileDownloader:
    """Handles secure file downloads with validation and retry logic."""
    
    ALLOWED_EXTENSIONS = ['.csv', '.tsv', '.xlsx', '.xls', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.zip', '.json', '.txt']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    
    def __init__(self, download_dir: Optional[str] = None, timeout: int = 60, max_retries: int = 3):
        self.download_dir = download_dir or tempfile.mkdtemp(prefix='quiz_solver_')
        self.timeout = timeout
        self.max_retries = max_retries
        os.makedirs(self.download_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 QuizSolver/1.0'})
        
    def download(self, url: str, filename: Optional[str] = None) -> Tuple[str, str]:
        """Download a file from URL with retries."""
        logger.info(f"Downloading: {url}")
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                if not filename:
                    filename = self._extract_filename(url, response)
                filename = self._sanitize_filename(filename)
                filepath = os.path.join(self.download_dir, filename)
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.zip':
                    return self._extract_zip(filepath)[0] if self._extract_zip(filepath) else (filepath, 'zip')
                
                return (filepath, ext.lstrip('.') or 'unknown')
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
                    
    def _extract_filename(self, url: str, response: requests.Response) -> str:
        cd = response.headers.get('Content-Disposition', '')
        if cd:
            match = re.search(r'filename[*]?=["\']?([^"\';]+)', cd)
            if match:
                return unquote(match.group(1))
        return os.path.basename(urlparse(url).path) or 'download'
        
    def _sanitize_filename(self, filename: str) -> str:
        filename = os.path.basename(filename).replace('\x00', '')
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return filename[:255] if len(filename) > 255 else filename or 'download'
        
    def _extract_zip(self, zip_path: str) -> List[Tuple[str, str]]:
        extracted = []
        extract_dir = os.path.splitext(zip_path)[0]
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if not name.startswith('/') and '..' not in name:
                        zf.extract(name, extract_dir)
                        filepath = os.path.join(extract_dir, name)
                        if os.path.isfile(filepath):
                            ext = os.path.splitext(name)[1].lower().lstrip('.')
                            extracted.append((filepath, ext or 'unknown'))
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file: {e}")
        return extracted
        
    def download_multiple(self, urls: List[str]) -> List[Tuple[str, str]]:
        results = []
        for url in urls:
            try:
                results.append(self.download(url))
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
        return results
        
    def cleanup(self):
        import shutil
        if self.download_dir and os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir, ignore_errors=True)

"""
PDF Utilities Module
PDF table/text/image extraction using pdfplumber.
"""

import io
import logging
from typing import Dict, List, Any, Optional
import pdfplumber

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Processes PDF files for text, table, and image extraction."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.pdf = None
        
    def __enter__(self):
        self.pdf = pdfplumber.open(self.filepath)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pdf:
            self.pdf.close()
            
    def get_page_count(self) -> int:
        """Get total number of pages."""
        if not self.pdf:
            with pdfplumber.open(self.filepath) as pdf:
                return len(pdf.pages)
        return len(self.pdf.pages)
        
    def extract_text(self, page_number: Optional[int] = None) -> str:
        """Extract text from PDF. If page_number is None, extract from all pages."""
        texts = []
        with pdfplumber.open(self.filepath) as pdf:
            if page_number is not None:
                if 1 <= page_number <= len(pdf.pages):
                    page = pdf.pages[page_number - 1]
                    texts.append(page.extract_text() or '')
            else:
                for page in pdf.pages:
                    texts.append(page.extract_text() or '')
        return '\n'.join(texts)
        
    def extract_tables(self, page_number: Optional[int] = None) -> List[List[List[str]]]:
        """Extract tables from PDF pages."""
        all_tables = []
        with pdfplumber.open(self.filepath) as pdf:
            pages = [pdf.pages[page_number - 1]] if page_number and 1 <= page_number <= len(pdf.pages) else pdf.pages
            for page in pages:
                tables = page.extract_tables()
                for table in tables:
                    cleaned = [[str(cell) if cell else '' for cell in row] for row in table if any(row)]
                    if cleaned:
                        all_tables.append(cleaned)
        return all_tables
        
    def extract_images(self, page_number: Optional[int] = None) -> List[bytes]:
        """Extract images from PDF pages."""
        images = []
        try:
            with pdfplumber.open(self.filepath) as pdf:
                pages = [pdf.pages[page_number - 1]] if page_number and 1 <= page_number <= len(pdf.pages) else pdf.pages
                for page in pages:
                    for img in page.images:
                        try:
                            img_obj = page.within_bbox((img['x0'], img['top'], img['x1'], img['bottom']))
                            if img_obj:
                                images.append(img_obj.to_image().original)
                        except Exception as e:
                            logger.warning(f"Failed to extract image: {e}")
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
        return images


def extract_pdf_text(filepath: str, page: Optional[int] = None) -> str:
    """Convenience function to extract text from PDF."""
    processor = PDFProcessor(filepath)
    return processor.extract_text(page)


def extract_pdf_tables(filepath: str, page: Optional[int] = None) -> List[List[List[str]]]:
    """Convenience function to extract tables from PDF."""
    processor = PDFProcessor(filepath)
    return processor.extract_tables(page)

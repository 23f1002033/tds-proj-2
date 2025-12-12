"""
OCR Utilities Module
OCR using pytesseract for image text extraction.
"""

import io
import re
import logging
from typing import Optional
import base64

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available")


class OCRProcessor:
    """Processes images for text extraction using OCR."""
    
    def __init__(self, lang: str = 'eng'):
        self.lang = lang
        if not TESSERACT_AVAILABLE:
            logger.warning("OCR functionality limited - pytesseract not installed")
            
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image file."""
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            img = Image.open(image_path)
            img = self._preprocess_image(img)
            text = pytesseract.image_to_string(img, lang=self.lang)
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
            
    def extract_from_bytes(self, image_bytes: bytes) -> str:
        """Extract text from image bytes."""
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = self._preprocess_image(img)
            text = pytesseract.image_to_string(img, lang=self.lang)
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
            
    def extract_from_base64(self, base64_string: str) -> str:
        """Extract text from base64-encoded image."""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            image_bytes = base64.b64decode(base64_string)
            return self.extract_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return ""
            
    def _preprocess_image(self, img: 'Image.Image') -> 'Image.Image':
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        # Increase contrast
        img = ImageOps.autocontrast(img)
        # Apply slight sharpening
        img = img.filter(ImageFilter.SHARPEN)
        return img
        
    def _clean_text(self, text: str) -> str:
        """Clean OCR output text."""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        return text
        
    def extract_numbers(self, image_path: str) -> list:
        """Extract numbers from image."""
        text = self.extract_text(image_path)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        return [float(n) if '.' in n else int(n) for n in numbers]


def ocr_image(image_path: str) -> str:
    """Convenience function for OCR."""
    return OCRProcessor().extract_text(image_path)

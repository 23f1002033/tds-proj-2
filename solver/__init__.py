"""
Quiz Solver Module
Contains all solver components for the Universal Quiz Solver API.
"""

from .browser import BrowserManager
from .parser import PageParser
from .classifier import TaskClassifier
from .downloader import FileDownloader
from .pdf_utils import PDFProcessor
from .csv_utils import CSVProcessor
from .ocr_utils import OCRProcessor
from .ml_utils import MLPredictor
from .chart_utils import ChartGenerator
from .api_utils import APIClient, GeminiClient
from .solver_core import QuizSolver

__all__ = [
    'BrowserManager',
    'PageParser',
    'TaskClassifier',
    'FileDownloader',
    'PDFProcessor',
    'CSVProcessor',
    'OCRProcessor',
    'MLPredictor',
    'ChartGenerator',
    'APIClient',
    'GeminiClient',
    'QuizSolver'
]

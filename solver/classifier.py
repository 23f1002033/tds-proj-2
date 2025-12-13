"""
Task Classifier Module
Classifies quiz tasks using rule-based logic and optional Gemini AI.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of supported task types."""
    NUMERIC_SUM = "numeric_sum"
    NUMERIC_MEAN = "numeric_mean"
    NUMERIC_MEDIAN = "numeric_median"
    NUMERIC_MAX = "numeric_max"
    NUMERIC_MIN = "numeric_min"
    NUMERIC_COUNT = "numeric_count"
    
    PDF_EXTRACT = "pdf_extract"
    PDF_TABLE = "pdf_table"
    PDF_IMAGE = "pdf_image"
    
    CSV_LOAD = "csv_load"
    CSV_GROUPBY = "csv_groupby"
    CSV_MERGE = "csv_merge"
    CSV_FILTER = "csv_filter"
    
    VISUALIZATION_HISTOGRAM = "viz_histogram"
    VISUALIZATION_BAR = "viz_bar"
    VISUALIZATION_SCATTER = "viz_scatter"
    VISUALIZATION_LINE = "viz_line"
    
    IMAGE_OCR = "image_ocr"
    IMAGE_EXTRACT = "image_extract"
    
    ML_PREDICT = "ml_predict"
    ML_CLASSIFY = "ml_classify"
    
    API_CALL = "api_call"
    API_MERGE = "api_merge"
    JWT_DECODE = "jwt_decode"
    ENCODING_DECODE = "encoding_decode"
    
    TEXT_EXTRACT = "text_extract"
    JSON_RESPONSE = "json_response"
    
    UNKNOWN = "unknown"


class TaskClassifier:
    """
    Classifies quiz tasks based on question text and page content.
    Uses two-layer architecture: rule-based + optional Gemini refinement.
    """
    
    # Keyword mappings for rule-based classification
    KEYWORD_MAP = {
        # Numeric operations
        TaskType.NUMERIC_SUM: ['sum', 'total', 'add up', 'sum of', 'summation'],
        TaskType.NUMERIC_MEAN: ['mean', 'average', 'avg'],
        TaskType.NUMERIC_MEDIAN: ['median', 'middle value'],
        TaskType.NUMERIC_MAX: ['maximum', 'max', 'largest', 'highest', 'biggest'],
        TaskType.NUMERIC_MIN: ['minimum', 'min', 'smallest', 'lowest'],
        TaskType.NUMERIC_COUNT: ['count', 'number of', 'how many', 'total count'],
        
        # PDF operations
        TaskType.PDF_EXTRACT: ['pdf', 'document', 'page'],
        TaskType.PDF_TABLE: ['table on page', 'pdf table', 'table in pdf'],
        TaskType.PDF_IMAGE: ['image in pdf', 'figure in pdf', 'diagram'],
        
        # CSV operations
        TaskType.CSV_LOAD: ['csv', 'spreadsheet', 'excel', 'xlsx', 'tsv'],
        TaskType.CSV_GROUPBY: ['group by', 'grouped', 'per category', 'by category'],
        TaskType.CSV_MERGE: ['merge', 'join', 'combine', 'combine datasets'],
        TaskType.CSV_FILTER: ['filter', 'where', 'only rows'],
        
        # Visualization
        TaskType.VISUALIZATION_HISTOGRAM: ['histogram', 'distribution', 'frequency plot'],
        TaskType.VISUALIZATION_BAR: ['bar chart', 'bar graph', 'bar plot'],
        TaskType.VISUALIZATION_SCATTER: ['scatter', 'scatter plot', 'xy plot'],
        TaskType.VISUALIZATION_LINE: ['line chart', 'line graph', 'trend'],
        
        # Image/OCR
        TaskType.IMAGE_OCR: ['extract text from image', 'read text in image', 'ocr', 'recognize text'],
        TaskType.IMAGE_EXTRACT: ['image', 'picture', 'photo', 'png', 'jpg'],
        
        # ML
        TaskType.ML_PREDICT: ['predict', 'prediction', 'forecast'],
        TaskType.ML_CLASSIFY: ['classify', 'classification', 'categorize'],
        
        # API
        TaskType.API_CALL: ['call api', 'fetch from api', 'api endpoint', 'http request'],
        TaskType.API_MERGE: ['api with csv', 'merge api', 'combine with api'],
        TaskType.JWT_DECODE: ['jwt', 'secret_code'],
        TaskType.ENCODING_DECODE: ['rot13', 'hex', 'base64', 'decode', 'encoding chain', 'encoded', 'unravel'],
        
        # Text/JSON
        TaskType.TEXT_EXTRACT: ['extract', 'find', 'locate', 'parse'],
        TaskType.JSON_RESPONSE: ['json', 'object', 'dictionary', 'key-value'],
    }
    
    # Priority order for task types (higher priority types checked first)
    PRIORITY_ORDER = [
        TaskType.ENCODING_DECODE,  # Encoding tasks detected early
        TaskType.JWT_DECODE,  # JWT tasks should be detected early
        TaskType.API_MERGE, TaskType.API_CALL,
        TaskType.VISUALIZATION_HISTOGRAM, TaskType.VISUALIZATION_BAR,
        TaskType.VISUALIZATION_SCATTER, TaskType.VISUALIZATION_LINE,
        TaskType.IMAGE_OCR, TaskType.IMAGE_EXTRACT,
        TaskType.ML_PREDICT, TaskType.ML_CLASSIFY,
        TaskType.PDF_TABLE, TaskType.PDF_IMAGE, TaskType.PDF_EXTRACT,
        TaskType.CSV_GROUPBY, TaskType.CSV_MERGE, TaskType.CSV_FILTER, TaskType.CSV_LOAD,
        TaskType.NUMERIC_SUM, TaskType.NUMERIC_MEAN, TaskType.NUMERIC_MEDIAN,
        TaskType.NUMERIC_MAX, TaskType.NUMERIC_MIN, TaskType.NUMERIC_COUNT,
        TaskType.JSON_RESPONSE, TaskType.TEXT_EXTRACT,
    ]
    
    def __init__(self, gemini_client=None):
        """
        Initialize classifier.
        
        Args:
            gemini_client: Optional Gemini client for AI-assisted classification
        """
        self.gemini_client = gemini_client
        
    def classify(self, question: str, page_data: Dict[str, Any] = None) -> Tuple[TaskType, Dict[str, Any]]:
        """
        Classify a quiz task.
        
        Args:
            question: Question text
            page_data: Parsed page data (optional)
            
        Returns:
            Tuple of (TaskType, metadata dict)
        """
        # First layer: Rule-based classification
        rule_result = self._rule_based_classify(question, page_data)
        
        # Second layer: Gemini refinement (if available and result is uncertain)
        if self.gemini_client and rule_result[0] == TaskType.UNKNOWN:
            try:
                gemini_result = self._gemini_classify(question, page_data)
                if gemini_result[0] != TaskType.UNKNOWN:
                    return gemini_result
            except Exception as e:
                logger.warning(f"Gemini classification failed, using rule-based: {e}")
        
        return rule_result
        
    def _rule_based_classify(self, question: str, page_data: Dict[str, Any] = None) -> Tuple[TaskType, Dict[str, Any]]:
        """
        Classify using keyword matching rules.
        
        Returns:
            Tuple of (TaskType, metadata dict)
        """
        question_lower = question.lower()
        metadata = {
            'confidence': 0.0,
            'matched_keywords': [],
            'detected_operations': [],
            'detected_columns': [],
            'detected_files': [],
        }
        
        # Check for multiple task types (complex/compound tasks)
        detected_types = []
        
        for task_type in self.PRIORITY_ORDER:
            keywords = self.KEYWORD_MAP.get(task_type, [])
            for keyword in keywords:
                if keyword in question_lower:
                    metadata['matched_keywords'].append(keyword)
                    detected_types.append((task_type, keyword))
                    break
        
        # Analyze page data for additional context
        if page_data:
            if page_data.get('file_links'):
                for file_link in page_data['file_links']:
                    metadata['detected_files'].append(file_link['type'])
                    
                # Refine classification based on file types
                file_types = set(f['type'] for f in page_data['file_links'])
                if 'pdf' in file_types and TaskType.PDF_EXTRACT not in [t[0] for t in detected_types]:
                    detected_types.append((TaskType.PDF_EXTRACT, 'file_link'))
                if any(t in file_types for t in ['csv', 'xlsx', 'xls']):
                    if TaskType.CSV_LOAD not in [t[0] for t in detected_types]:
                        detected_types.append((TaskType.CSV_LOAD, 'file_link'))
        
        # Extract column names mentioned
        col_pattern = r'column\s*["\']?(\w+)["\']?'
        columns = re.findall(col_pattern, question, re.IGNORECASE)
        metadata['detected_columns'] = columns
        
        # Determine final task type
        if detected_types:
            primary_type = detected_types[0][0]
            metadata['confidence'] = min(0.9, 0.5 + 0.1 * len(detected_types))
            metadata['detected_operations'] = [t[0].value for t in detected_types]
            
            # Check for compound tasks
            if len(detected_types) > 1:
                metadata['is_compound'] = True
                metadata['secondary_types'] = [t[0].value for t in detected_types[1:]]
            
            return (primary_type, metadata)
        
        return (TaskType.UNKNOWN, metadata)
        
    def _gemini_classify(self, question: str, page_data: Dict[str, Any] = None) -> Tuple[TaskType, Dict[str, Any]]:
        """
        Use Gemini API for classification refinement.
        
        Returns:
            Tuple of (TaskType, metadata dict)
        """
        if not self.gemini_client:
            return (TaskType.UNKNOWN, {'error': 'Gemini client not available'})
        
        prompt = self._build_classification_prompt(question, page_data)
        
        try:
            response = self.gemini_client.call(prompt)
            return self._parse_gemini_response(response)
        except Exception as e:
            logger.error(f"Gemini classification error: {e}")
            return (TaskType.UNKNOWN, {'error': str(e)})
            
    def _build_classification_prompt(self, question: str, page_data: Dict[str, Any] = None) -> str:
        """Build prompt for Gemini classification."""
        task_types_str = ", ".join([t.value for t in TaskType])
        
        prompt = f"""Analyze this quiz question and classify it into one of these task types:
{task_types_str}

Question: {question}

"""
        if page_data:
            if page_data.get('file_links'):
                prompt += f"Available files: {page_data['file_links']}\n"
            if page_data.get('instructions'):
                prompt += f"Instructions: {page_data['instructions'][:500]}\n"
        
        prompt += """
Respond in this exact JSON format:
{
    "task_type": "<one of the task types>",
    "confidence": <0.0 to 1.0>,
    "operations": [<list of required operations>],
    "target_column": "<column name if applicable>",
    "output_format": "<expected output format>"
}
"""
        return prompt
        
    def _parse_gemini_response(self, response: Dict[str, Any]) -> Tuple[TaskType, Dict[str, Any]]:
        """Parse Gemini classification response."""
        try:
            text = response.get('text', '')
            
            # Extract JSON from response
            import json
            json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                task_type_str = data.get('task_type', 'unknown')
                
                # Map string to TaskType
                task_type = TaskType.UNKNOWN
                for t in TaskType:
                    if t.value == task_type_str:
                        task_type = t
                        break
                
                metadata = {
                    'confidence': data.get('confidence', 0.5),
                    'operations': data.get('operations', []),
                    'target_column': data.get('target_column'),
                    'output_format': data.get('output_format'),
                    'source': 'gemini'
                }
                
                return (task_type, metadata)
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
        
        return (TaskType.UNKNOWN, {'error': 'Failed to parse response'})
        
    def get_required_modules(self, task_type: TaskType) -> List[str]:
        """
        Get list of required solver modules for a task type.
        
        Args:
            task_type: The classified task type
            
        Returns:
            List of module names
        """
        module_map = {
            TaskType.NUMERIC_SUM: ['csv_utils'],
            TaskType.NUMERIC_MEAN: ['csv_utils'],
            TaskType.NUMERIC_MEDIAN: ['csv_utils'],
            TaskType.NUMERIC_MAX: ['csv_utils'],
            TaskType.NUMERIC_MIN: ['csv_utils'],
            TaskType.NUMERIC_COUNT: ['csv_utils'],
            
            TaskType.PDF_EXTRACT: ['downloader', 'pdf_utils'],
            TaskType.PDF_TABLE: ['downloader', 'pdf_utils', 'csv_utils'],
            TaskType.PDF_IMAGE: ['downloader', 'pdf_utils', 'ocr_utils'],
            
            TaskType.CSV_LOAD: ['downloader', 'csv_utils'],
            TaskType.CSV_GROUPBY: ['downloader', 'csv_utils'],
            TaskType.CSV_MERGE: ['downloader', 'csv_utils'],
            TaskType.CSV_FILTER: ['downloader', 'csv_utils'],
            
            TaskType.VISUALIZATION_HISTOGRAM: ['csv_utils', 'chart_utils'],
            TaskType.VISUALIZATION_BAR: ['csv_utils', 'chart_utils'],
            TaskType.VISUALIZATION_SCATTER: ['csv_utils', 'chart_utils'],
            TaskType.VISUALIZATION_LINE: ['csv_utils', 'chart_utils'],
            
            TaskType.IMAGE_OCR: ['downloader', 'ocr_utils'],
            TaskType.IMAGE_EXTRACT: ['downloader'],
            
            TaskType.ML_PREDICT: ['csv_utils', 'ml_utils'],
            TaskType.ML_CLASSIFY: ['csv_utils', 'ml_utils'],
            
            TaskType.API_CALL: ['api_utils'],
            TaskType.API_MERGE: ['api_utils', 'csv_utils'],
            
            TaskType.TEXT_EXTRACT: ['parser'],
            TaskType.JSON_RESPONSE: ['parser'],
        }
        
        return module_map.get(task_type, [])


def detect_output_format(question: str) -> str:
    """
    Detect expected output format from question text.
    
    Args:
        question: Question text
        
    Returns:
        Output format string ('number', 'string', 'json', 'base64', 'boolean')
    """
    question_lower = question.lower()
    
    if any(kw in question_lower for kw in ['json', 'object', 'dictionary']):
        return 'json'
    elif any(kw in question_lower for kw in ['base64', 'image', 'plot', 'chart', 'graph']):
        return 'base64'
    elif any(kw in question_lower for kw in ['true or false', 'yes or no', 'boolean']):
        return 'boolean'
    elif any(kw in question_lower for kw in ['number', 'sum', 'total', 'count', 'integer', 'float']):
        return 'number'
    else:
        return 'string'

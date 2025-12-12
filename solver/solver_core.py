"""
Solver Core Module
Main logic for solving quizzes with recursive loop handling.
"""

import asyncio
import logging
import time
import re
import json
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

from .browser import BrowserManager
from .parser import PageParser, extract_question_details
from .classifier import TaskClassifier, TaskType, detect_output_format
from .downloader import FileDownloader
from .pdf_utils import PDFProcessor
from .csv_utils import CSVProcessor
from .ocr_utils import OCRProcessor
from .chart_utils import ChartGenerator
from .api_utils import APIClient, GeminiClient
from .ml_utils import MLPredictor

logger = logging.getLogger(__name__)


class QuizSolver:
    """Main quiz solving engine with recursive loop handling."""
    
    def __init__(self, timeout: int = 180):
        """Initialize solver with timeout in seconds."""
        self.timeout = timeout
        self.start_time = None
        self.gemini_client = None
        try:
            self.gemini_client = GeminiClient()
        except:
            logger.warning("Gemini client not available")
            
        self.classifier = TaskClassifier(self.gemini_client)
        self.downloader = FileDownloader()
        self.api_client = APIClient()
        self.csv_processor = CSVProcessor()
        self.chart_generator = ChartGenerator()
        
    async def solve(self, url: str) -> Dict[str, Any]:
        """Main solving entry point with quiz chaining support."""
        self.start_time = time.time()
        results = []
        current_url = url
        
        async with BrowserManager() as browser:
            while current_url and self._check_timeout():
                try:
                    result = await self._solve_single_quiz(browser, current_url)
                    results.append(result)
                    
                    # Check for next quiz URL
                    if result.get('correct') and result.get('next_url'):
                        current_url = result['next_url']
                        logger.info(f"Chaining to next quiz: {current_url}")
                    else:
                        current_url = None
                        
                except Exception as e:
                    logger.error(f"Error solving quiz: {e}")
                    results.append({'error': str(e), 'url': current_url})
                    break
                    
        self.downloader.cleanup()
        return {'results': results, 'total_time': time.time() - self.start_time}
        
    async def _solve_single_quiz(self, browser: BrowserManager, url: str) -> Dict[str, Any]:
        """Solve a single quiz page."""
        # Load page
        page = await browser.load_page(url)
        content = await browser.get_page_content(page)
        
        # Parse page
        parser = PageParser(base_url=url)
        page_data = parser.parse(content['html'])
        
        # Classify task
        question = page_data.get('question', content['text'][:2000])
        task_type, metadata = self.classifier.classify(question, page_data)
        logger.info(f"Classified as: {task_type.value}, confidence: {metadata.get('confidence', 0)}")
        
        # Execute appropriate solver
        answer = await self._execute_solver(task_type, question, page_data, metadata)
        
        # Format answer
        output_format = detect_output_format(question)
        formatted_answer = self._format_answer(answer, output_format)
        
        # Submit answer
        submit_url = page_data.get('submit_url')
        submit_result = None
        if submit_url:
            submit_result = await self._submit_answer(submit_url, formatted_answer)
            
        await page.close()
        
        return {
            'url': url,
            'task_type': task_type.value,
            'question': question[:500],
            'answer': formatted_answer,
            'submit_url': submit_url,
            'submit_result': submit_result,
            'correct': submit_result.get('correct') if submit_result else None,
            'next_url': submit_result.get('url') if submit_result else None
        }
        
    async def _execute_solver(self, task_type: TaskType, question: str, 
                             page_data: Dict, metadata: Dict) -> Any:
        """Execute the appropriate solver based on task type."""
        details = extract_question_details(question)
        file_links = page_data.get('file_links', [])
        
        try:
            # Download files if needed
            files = []
            if file_links:
                for link in file_links[:3]:  # Limit to 3 files
                    try:
                        filepath, ftype = self.downloader.download(link['url'])
                        files.append({'path': filepath, 'type': ftype})
                    except Exception as e:
                        logger.warning(f"Failed to download {link['url']}: {e}")
            
            # Route to appropriate solver
            if task_type in [TaskType.NUMERIC_SUM, TaskType.NUMERIC_MEAN, TaskType.NUMERIC_MEDIAN,
                            TaskType.NUMERIC_MAX, TaskType.NUMERIC_MIN, TaskType.NUMERIC_COUNT]:
                return await self._solve_numeric(task_type, files, details, page_data)
                
            elif task_type in [TaskType.PDF_EXTRACT, TaskType.PDF_TABLE]:
                return await self._solve_pdf(files, details, page_data)
                
            elif task_type in [TaskType.CSV_LOAD, TaskType.CSV_GROUPBY, TaskType.CSV_MERGE]:
                return await self._solve_csv(task_type, files, details)
                
            elif task_type in [TaskType.VISUALIZATION_HISTOGRAM, TaskType.VISUALIZATION_BAR,
                              TaskType.VISUALIZATION_SCATTER]:
                return await self._solve_visualization(task_type, files, details)
                
            elif task_type in [TaskType.IMAGE_OCR, TaskType.IMAGE_EXTRACT]:
                return await self._solve_image(files, page_data, details)
                
            elif task_type in [TaskType.ML_PREDICT, TaskType.ML_CLASSIFY]:
                return await self._solve_ml(files, details)
                
            elif task_type in [TaskType.API_CALL, TaskType.API_MERGE]:
                return await self._solve_api(page_data, files, details)
                
            else:
                return await self._solve_text(question, page_data)
                
        except Exception as e:
            logger.error(f"Solver execution error: {e}")
            return await self._solve_with_gemini(question, page_data)
            
    async def _solve_numeric(self, task_type: TaskType, files: List, 
                            details: Dict, page_data: Dict) -> float:
        """Solve numeric computation tasks."""
        df = None
        
        # Load data from file
        for f in files:
            if f['type'] in ['csv', 'tsv', 'xlsx', 'xls']:
                df = self.csv_processor.load(f['path'])
                break
                
        # Or from HTML tables
        if df is None and page_data.get('tables'):
            import pandas as pd
            table = page_data['tables'][0]
            if len(table) > 1:
                df = pd.DataFrame(table[1:], columns=table[0])
                df = self.csv_processor._clean_dataframe(df)
                
        if df is None:
            return 0.0
            
        # Determine column
        column = details.get('target_column')
        if not column:
            numeric_cols = self.csv_processor.get_numeric_columns(df)
            column = numeric_cols[0] if numeric_cols else df.columns[0]
            
        # Map task type to operation
        op_map = {
            TaskType.NUMERIC_SUM: 'sum',
            TaskType.NUMERIC_MEAN: 'mean',
            TaskType.NUMERIC_MEDIAN: 'median',
            TaskType.NUMERIC_MAX: 'max',
            TaskType.NUMERIC_MIN: 'min',
            TaskType.NUMERIC_COUNT: 'count'
        }
        
        operation = op_map.get(task_type, 'sum')
        return self.csv_processor.compute(df, column, operation)
        
    async def _solve_pdf(self, files: List, details: Dict, page_data: Dict) -> Any:
        """Solve PDF extraction tasks."""
        for f in files:
            if f['type'] == 'pdf':
                processor = PDFProcessor(f['path'])
                page_num = details.get('page_number')
                
                # Try table extraction first
                tables = processor.extract_tables(page_num)
                if tables:
                    import pandas as pd
                    table = tables[0]
                    if len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        # Return sum of first numeric column
                        numeric = self.csv_processor.get_numeric_columns(
                            self.csv_processor._clean_dataframe(df))
                        if numeric:
                            return float(df[numeric[0]].sum())
                            
                # Fall back to text extraction
                text = processor.extract_text(page_num)
                return text[:1000] if text else ""
                
        return ""
        
    async def _solve_csv(self, task_type: TaskType, files: List, details: Dict) -> Any:
        """Solve CSV processing tasks."""
        dfs = []
        for f in files:
            if f['type'] in ['csv', 'tsv', 'xlsx', 'xls']:
                dfs.append(self.csv_processor.load(f['path']))
                
        if not dfs:
            return {}
            
        df = dfs[0]
        
        if task_type == TaskType.CSV_GROUPBY and len(df.columns) >= 2:
            group_col = df.columns[0]
            agg_col = self.csv_processor.get_numeric_columns(df)[0] if self.csv_processor.get_numeric_columns(df) else df.columns[1]
            return self.csv_processor.groupby(df, group_col, agg_col)
            
        elif task_type == TaskType.CSV_MERGE and len(dfs) > 1:
            df = self.csv_processor.merge(dfs[0], dfs[1])
            
        return df.to_dict(orient='records')
        
    async def _solve_visualization(self, task_type: TaskType, files: List, details: Dict) -> str:
        """Solve visualization tasks."""
        df = None
        for f in files:
            if f['type'] in ['csv', 'tsv', 'xlsx', 'xls']:
                df = self.csv_processor.load(f['path'])
                break
                
        if df is None:
            return ""
            
        column = details.get('target_column')
        if not column:
            numeric_cols = self.csv_processor.get_numeric_columns(df)
            column = numeric_cols[0] if numeric_cols else df.columns[0]
            
        data = df[column].dropna().tolist()
        
        if task_type == TaskType.VISUALIZATION_HISTOGRAM:
            return self.chart_generator.histogram(data, title=f"Histogram of {column}")
        elif task_type == TaskType.VISUALIZATION_BAR:
            categories = df[df.columns[0]].astype(str).tolist()[:20]
            values = data[:20]
            return self.chart_generator.bar_chart(categories, values)
        elif task_type == TaskType.VISUALIZATION_SCATTER:
            x = data
            y = df[df.columns[1]].dropna().tolist() if len(df.columns) > 1 else data
            return self.chart_generator.scatter_plot(x, y)
            
        return ""
        
    async def _solve_image(self, files: List, page_data: Dict, details: Dict) -> str:
        """Solve image/OCR tasks."""
        ocr = OCRProcessor()
        
        # Check files first
        for f in files:
            if f['type'] in ['png', 'jpg', 'jpeg', 'gif']:
                text = ocr.extract_text(f['path'])
                if text:
                    return text
                    
        # Check embedded images
        for img in page_data.get('embedded_images', []):
            if img.get('is_base64'):
                text = ocr.extract_from_base64(img['src'])
                if text:
                    return text
                    
        return ""
        
    async def _solve_ml(self, files: List, details: Dict) -> Any:
        """Solve ML prediction tasks."""
        import os
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
        
        predictor = MLPredictor(model_path)
        
        # Load features from CSV
        for f in files:
            if f['type'] in ['csv', 'xlsx']:
                df = self.csv_processor.load(f['path'])
                features = df.select_dtypes(include=['number']).values
                if len(features) > 0:
                    return predictor.predict(features[0])
                    
        return None
        
    async def _solve_api(self, page_data: Dict, files: List, details: Dict) -> Any:
        """Solve API call tasks."""
        endpoints = page_data.get('api_endpoints', [])
        
        for endpoint in endpoints:
            try:
                response = self.api_client.get(endpoint)
                data = response.json()
                
                # Merge with CSV if available
                if files:
                    for f in files:
                        if f['type'] in ['csv', 'xlsx']:
                            df = self.csv_processor.load(f['path'])
                            # Simple merge logic
                            return {'api_data': data, 'file_data': df.to_dict()}
                            
                return data
            except Exception as e:
                logger.warning(f"API call failed: {e}")
                
        return {}
        
    async def _solve_text(self, question: str, page_data: Dict) -> str:
        """Solve text extraction tasks."""
        text = page_data.get('question', '')
        
        # Look for specific patterns
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            return numbers[0]
            
        return text[:500]
        
    async def _solve_with_gemini(self, question: str, page_data: Dict) -> Any:
        """Fallback to Gemini for complex tasks."""
        if not self.gemini_client:
            return ""
            
        try:
            prompt = f"""Solve this quiz question and provide ONLY the answer:

Question: {question}

Available data:
- Tables: {page_data.get('tables', [])[:2]}
- File links: {page_data.get('file_links', [])}
- Instructions: {page_data.get('instructions', [])[:5]}

Provide the answer in the most appropriate format (number, string, or JSON).
Answer:"""
            
            result = self.gemini_client.call(prompt)
            return result.get('text', '').strip()
        except Exception as e:
            logger.error(f"Gemini fallback failed: {e}")
            return ""
            
    def _format_answer(self, answer: Any, output_format: str) -> Any:
        """Format answer according to expected output type."""
        if output_format == 'number':
            try:
                if isinstance(answer, str):
                    nums = re.findall(r'-?\d+(?:\.\d+)?', answer)
                    return float(nums[0]) if nums else 0.0
                return float(answer)
            except:
                return 0.0
                
        elif output_format == 'json':
            if isinstance(answer, (dict, list)):
                return answer
            try:
                return json.loads(answer)
            except:
                return {"value": str(answer)}
                
        elif output_format == 'boolean':
            if isinstance(answer, bool):
                return answer
            text = str(answer).lower()
            return text in ['true', 'yes', '1']
            
        elif output_format == 'base64':
            if isinstance(answer, str) and answer.startswith('data:image'):
                return answer
            return str(answer)
            
        return str(answer)
        
    async def _submit_answer(self, submit_url: str, answer: Any) -> Dict[str, Any]:
        """Submit answer to the quiz endpoint."""
        try:
            payload = {'answer': answer}
            response = self.api_client.post(submit_url, json=payload)
            return response.json()
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return {'error': str(e)}
            
    def _check_timeout(self) -> bool:
        """Check if we're still within timeout."""
        if self.start_time is None:
            return True
        elapsed = time.time() - self.start_time
        return elapsed < self.timeout

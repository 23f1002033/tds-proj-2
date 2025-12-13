"""
Solver Core Module
Main logic for solving quizzes with recursive loop handling.
"""

import asyncio
import logging
import time
import re
import json
import base64
import os
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
    
    def __init__(self, email: str = '', secret: str = '', timeout: int = 180):
        """Initialize solver with email, secret and timeout."""
        self.timeout = timeout
        self.start_time = None
        self.email = email or os.getenv('USER_EMAIL', '')
        self.secret = secret
        self.current_quiz_url = None  # Track current quiz URL for submission
        
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
        # Track current quiz URL for submission
        self.current_quiz_url = url
        
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
            
            elif task_type == TaskType.JWT_DECODE:
                return await self._solve_jwt(page_data, question)
            
            elif task_type == TaskType.ENCODING_DECODE:
                return await self._solve_encoding(page_data, question)
            
            elif task_type == TaskType.AUDIO_TRANSCRIBE:
                return await self._solve_audio(files, page_data, question)
            
            elif task_type == TaskType.SHELL_COMMAND:
                return await self._solve_shell(question, page_data)
            
            elif task_type == TaskType.FILE_SHARDS:
                return await self._solve_shards(files, page_data, question)
                
            else:
                return await self._solve_with_gemini_enhanced(question, page_data, files)
                
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
    
    async def _solve_jwt(self, page_data: Dict, question: str) -> Any:
        """Solve JWT token decoding tasks."""
        if not self.email:
            logger.error("USER_EMAIL not configured for JWT task")
            return await self._solve_with_gemini(question, page_data)
        
        # Find JWT API endpoint from page data
        api_endpoints = page_data.get('api_endpoints', [])
        jwt_endpoint = None
        
        for endpoint in api_endpoints:
            if 'jwt' in endpoint.lower():
                jwt_endpoint = endpoint
                break
        
        # Also search in instructions/question for JWT endpoint
        if not jwt_endpoint:
            jwt_pattern = r'https?://[^\s<>"\']*/api/jwt[^\s<>"\']*'
            match = re.search(jwt_pattern, question, re.IGNORECASE)
            if match:
                jwt_endpoint = match.group(0)
            else:
                # Try relative pattern
                rel_pattern = r'/api/jwt'
                match = re.search(rel_pattern, question)
                if match:
                    # Build from base URL in api_endpoints
                    for ep in api_endpoints:
                        if '/api/' in ep:
                            base = ep.split('/api/')[0]
                            jwt_endpoint = f"{base}/api/jwt"
                            break
        
        if not jwt_endpoint:
            logger.warning("Could not find JWT endpoint")
            return await self._solve_with_gemini(question, page_data)
        
        # Add email parameter
        if '?' in jwt_endpoint:
            jwt_url = f"{jwt_endpoint}&email={self.email}"
        else:
            jwt_url = f"{jwt_endpoint}?email={self.email}"
        
        # Replace placeholder if present
        jwt_url = jwt_url.replace('YOUR_EMAIL', self.email)
        jwt_url = jwt_url.replace('<YOUR_EMAIL>', self.email)
        
        logger.info(f"Fetching JWT from: {jwt_url}")
        
        try:
            response = self.api_client.get(jwt_url)
            data = response.json()
            token = data.get('token', '')
            
            if not token:
                logger.error("No token in JWT response")
                return await self._solve_with_gemini(question, page_data)
            
            # Decode JWT payload (middle part)
            parts = token.split('.')
            if len(parts) != 3:
                logger.error(f"Invalid JWT format: {token[:50]}...")
                return await self._solve_with_gemini(question, page_data)
            
            # Base64url decode the payload
            payload_b64 = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += '=' * padding
            # Replace base64url chars with base64 chars
            payload_b64 = payload_b64.replace('-', '+').replace('_', '/')
            
            payload_json = base64.b64decode(payload_b64).decode('utf-8')
            payload = json.loads(payload_json)
            
            logger.info(f"Decoded JWT payload: {payload}")
            
            # Extract secret_code or similar field
            if 'secret_code' in payload:
                return payload['secret_code']
            elif 'code' in payload:
                return payload['code']
            elif 'secret' in payload:
                return payload['secret']
            elif 'answer' in payload:
                return payload['answer']
            else:
                # Return first numeric value found
                for key, value in payload.items():
                    if isinstance(value, (int, float)) and key != 'exp' and key != 'iat':
                        return value
                return payload
                
        except Exception as e:
            logger.error(f"JWT solving failed: {e}")
            return await self._solve_with_gemini(question, page_data)
    
    async def _solve_encoding(self, page_data: Dict, question: str) -> Any:
        """Solve encoding chain puzzles (Base64, Hex, ROT13, etc.)."""
        import codecs
        
        # Find the encoded data endpoint
        api_endpoints = page_data.get('api_endpoints', [])
        encoded_endpoint = None
        
        for endpoint in api_endpoints:
            if 'encoded' in endpoint.lower():
                encoded_endpoint = endpoint
                break
        
        if not encoded_endpoint:
            # Try to find from question
            pattern = r'/api/encoded[^\s<>"\']*'
            match = re.search(pattern, question)
            if match:
                # Build full URL from base
                for ep in api_endpoints:
                    if '/api/' in ep:
                        base = ep.split('/api/')[0]
                        encoded_endpoint = f"{base}{match.group(0)}"
                        break
        
        if not encoded_endpoint:
            logger.warning("Could not find encoded endpoint")
            return await self._solve_with_gemini(question, page_data)
        
        logger.info(f"Fetching encoded data from: {encoded_endpoint}")
        
        try:
            response = self.api_client.get(encoded_endpoint)
            data = response.json()
            encoded_str = data.get('encoded', data.get('data', data.get('string', '')))
            
            if not encoded_str:
                # Try to get the first string value
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 5:
                        encoded_str = value
                        break
            
            logger.info(f"Encoded string: {encoded_str[:50]}...")
            
            # Detect encoding chain from question
            question_lower = question.lower()
            
            # Determine decoding order (reverse of encoding)
            # Common pattern: Original → ROT13 → Hex → Base64
            # Decode order: Base64 → Hex → ROT13
            
            decoded = encoded_str
            
            # Step 1: Base64 decode
            if 'base64' in question_lower:
                try:
                    padding = 4 - len(decoded) % 4
                    if padding != 4:
                        decoded += '=' * padding
                    decoded = base64.b64decode(decoded).decode('utf-8')
                    logger.info(f"After Base64 decode: {decoded[:50]}...")
                except Exception as e:
                    logger.warning(f"Base64 decode failed: {e}")
            
            # Step 2: Hex decode
            if 'hex' in question_lower:
                try:
                    decoded = bytes.fromhex(decoded).decode('utf-8')
                    logger.info(f"After Hex decode: {decoded[:50]}...")
                except Exception as e:
                    logger.warning(f"Hex decode failed: {e}")
            
            # Step 3: ROT13 decode
            if 'rot13' in question_lower:
                try:
                    decoded = codecs.decode(decoded, 'rot_13')
                    logger.info(f"After ROT13 decode: {decoded}")
                except Exception as e:
                    logger.warning(f"ROT13 decode failed: {e}")
            
            return decoded.strip()
            
        except Exception as e:
            logger.error(f"Encoding solving failed: {e}")
            return await self._solve_with_gemini(question, page_data)
    
    async def _solve_audio(self, files: List, page_data: Dict, question: str) -> Any:
        """Solve audio transcription tasks using Gemini."""
        # Download audio file if available
        audio_files = [f for f in files if f[1] in ['mp3', 'wav', 'ogg', 'm4a', 'webm']]
        
        if not audio_files and page_data.get('file_links'):
            for link in page_data['file_links']:
                link_lower = link.lower()
                if any(ext in link_lower for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                    try:
                        downloaded = self.downloader.download(link)
                        audio_files.append(downloaded)
                    except Exception as e:
                        logger.warning(f"Failed to download audio: {e}")
        
        if not audio_files:
            logger.warning("No audio files found, falling back to Gemini")
            return await self._solve_with_gemini_enhanced(question, page_data, files)
        
        # For Gemini audio transcription, we need to use the multimodal API
        # For now, use enhanced Gemini prompt with audio context
        try:
            audio_path = audio_files[0][0]
            logger.info(f"Processing audio file: {audio_path}")
            
            # Try using Gemini with audio file description
            prompt = f"""This is an audio transcription task.

Question: {question}

The audio file is at: {audio_path}

Based on the question, what is likely being asked? Common audio tasks include:
- Transcribing speech to text
- Finding a passphrase or code word
- Identifying specific spoken content

If this is a passphrase question, provide the most likely passphrase format.
Answer:"""
            
            if self.gemini_client:
                result = self.gemini_client.call(prompt)
                return result.get('text', '').strip()
            
            return ""
        except Exception as e:
            logger.error(f"Audio solving failed: {e}")
            return await self._solve_with_gemini_enhanced(question, page_data, files)
    
    async def _solve_shell(self, question: str, page_data: Dict) -> Any:
        """Solve shell command tasks (git, uv, npm, etc.)."""
        import subprocess
        
        question_lower = question.lower()
        
        try:
            # Git commands
            if 'git' in question_lower:
                if 'commit' in question_lower and 'hash' in question_lower:
                    # Get commit hash
                    result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                          capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        return result.stdout.strip()[:7]
                        
                elif 'branch' in question_lower:
                    result = subprocess.run(['git', 'branch', '--show-current'], 
                                          capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        return result.stdout.strip()
            
            # UV dependency check
            if 'uv' in question_lower:
                if 'version' in question_lower:
                    result = subprocess.run(['uv', '--version'], 
                                          capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        return result.stdout.strip()
            
            # Fall back to Gemini for complex shell tasks
            return await self._solve_with_gemini_enhanced(question, page_data, [])
            
        except Exception as e:
            logger.error(f"Shell solving failed: {e}")
            return await self._solve_with_gemini_enhanced(question, page_data, [])
    
    async def _solve_shards(self, files: List, page_data: Dict, question: str) -> Any:
        """Solve file shard reconstruction tasks."""
        # Download all shard files
        shard_files = []
        
        for link in page_data.get('file_links', []):
            if any(kw in link.lower() for kw in ['shard', 'part', 'fragment', 'piece']):
                try:
                    downloaded = self.downloader.download(link)
                    shard_files.append(downloaded)
                except Exception as e:
                    logger.warning(f"Failed to download shard: {e}")
        
        if not shard_files:
            return await self._solve_with_gemini_enhanced(question, page_data, files)
        
        try:
            # Sort shards by filename (usually numbered)
            shard_files.sort(key=lambda x: x[0])
            
            # Read and concatenate shards
            combined_content = b''
            for shard_path, _ in shard_files:
                with open(shard_path, 'rb') as f:
                    combined_content += f.read()
            
            # Try to decode as text
            try:
                text_content = combined_content.decode('utf-8')
                logger.info(f"Reconstructed file content: {text_content[:100]}...")
                
                # Look for answer patterns
                if 'password' in question.lower() or 'code' in question.lower():
                    # Extract password/code patterns
                    patterns = re.findall(r'[A-Z0-9_]{4,}', text_content)
                    if patterns:
                        return patterns[0]
                
                return text_content.strip()
            except UnicodeDecodeError:
                # Binary file - return base64
                import base64
                return base64.b64encode(combined_content).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Shard solving failed: {e}")
            return await self._solve_with_gemini_enhanced(question, page_data, files)
    
    async def _solve_with_gemini_enhanced(self, question: str, page_data: Dict, files: List) -> Any:
        """Enhanced Gemini fallback with more context for complex tasks."""
        if not self.gemini_client:
            return ""
        
        try:
            # Build comprehensive context
            context_parts = []
            
            # Add file contents if available
            for filepath, filetype in files[:3]:  # Limit to 3 files
                try:
                    if filetype in ['csv', 'txt', 'json', 'md']:
                        with open(filepath, 'r') as f:
                            content = f.read()[:2000]
                            context_parts.append(f"File ({filetype}): {content}")
                except:
                    pass
            
            # Add table data
            if page_data.get('tables'):
                for i, table in enumerate(page_data['tables'][:2]):
                    context_parts.append(f"Table {i+1}: {str(table)[:500]}")
            
            # Add API endpoints
            if page_data.get('api_endpoints'):
                context_parts.append(f"API endpoints: {page_data['api_endpoints']}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""You are a quiz-solving AI. Solve this problem step by step and provide ONLY the final answer.

QUESTION: {question}

AVAILABLE DATA:
{context}

INSTRUCTIONS FROM PAGE:
{page_data.get('instructions', [])[:10]}

Think through this carefully:
1. What is being asked?
2. What data do I have?
3. What computation or extraction is needed?

IMPORTANT: Return ONLY the final answer - no explanation. If it's a number, return the number. If it's text, return the text. If it's JSON, return valid JSON.

ANSWER:"""
            
            result = self.gemini_client.call(prompt, {'temperature': 0.1, 'maxOutputTokens': 500})
            answer = result.get('text', '').strip()
            
            # Clean up the answer
            answer = answer.strip('`').strip('"').strip("'")
            if answer.startswith('```'):
                answer = answer.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            
            logger.info(f"Gemini enhanced answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Gemini enhanced fallback failed: {e}")
            return ""
        
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
            # Build payload with all required fields per project spec
            payload = {
                'email': self.email,
                'secret': self.secret,
                'url': self.current_quiz_url,
                'answer': answer
            }
            
            logger.info(f"Submitting to {submit_url}: {payload}")
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

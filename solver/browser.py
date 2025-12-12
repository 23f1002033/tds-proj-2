"""
Browser Automation Module
Playwright wrapper for headless browser operations.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout

logger = logging.getLogger(__name__)


class BrowserManager:
    """
    Manages headless browser interactions using Playwright.
    Provides capabilities for loading JS-rendered pages and extracting content.
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize browser manager.
        
        Args:
            headless: Run browser in headless mode
            timeout: Default timeout in milliseconds
        """
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Browser] = None
        self.playwright = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def start(self):
        """Start the browser instance."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            logger.info("Browser started successfully")
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise
            
    async def close(self):
        """Close the browser instance."""
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        logger.info("Browser closed")
        
    async def create_page(self) -> Page:
        """Create a new browser page."""
        if not self.browser:
            await self.start()
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        page.set_default_timeout(self.timeout)
        return page
        
    async def load_page(self, url: str, wait_for: str = 'networkidle') -> Page:
        """
        Load a page with full JS execution.
        
        Args:
            url: URL to load
            wait_for: Wait condition ('load', 'domcontentloaded', 'networkidle')
            
        Returns:
            Loaded page instance
        """
        page = await self.create_page()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading page: {url} (attempt {attempt + 1})")
                await page.goto(url, wait_until=wait_for, timeout=self.timeout)
                
                # Wait for page to stabilize
                await self._wait_for_stability(page)
                
                logger.info(f"Page loaded successfully: {url}")
                return page
                
            except PlaywrightTimeout:
                logger.warning(f"Timeout loading page (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error loading page: {e}")
                if attempt == max_retries - 1:
                    raise
                await page.close()
                page = await self.create_page()
                await asyncio.sleep(2)
                
        return page
        
    async def _wait_for_stability(self, page: Page, check_interval: float = 0.5, stable_time: float = 1.0):
        """
        Wait for the page to stabilize (no more DOM changes).
        
        Args:
            page: Page instance
            check_interval: Time between checks in seconds
            stable_time: Required stable time in seconds
        """
        last_html = ""
        stable_count = 0
        required_stable = int(stable_time / check_interval)
        
        for _ in range(20):  # Max 10 seconds
            current_html = await page.content()
            
            if current_html == last_html:
                stable_count += 1
                if stable_count >= required_stable:
                    return
            else:
                stable_count = 0
                last_html = current_html
                
            await asyncio.sleep(check_interval)
            
    async def get_page_content(self, page: Page) -> Dict[str, Any]:
        """
        Extract all content from a page.
        
        Returns:
            Dictionary containing text, HTML, links, and images
        """
        content = {
            'text': '',
            'html': '',
            'links': [],
            'images': [],
            'url': page.url
        }
        
        try:
            content['html'] = await page.content()
            content['text'] = await page.evaluate('() => document.body.innerText')
            
            # Extract all anchor links
            content['links'] = await page.evaluate('''() => {
                return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    href: a.href,
                    text: a.innerText.trim()
                }));
            }''')
            
            # Extract embedded images
            content['images'] = await page.evaluate('''() => {
                return Array.from(document.querySelectorAll('img')).map(img => ({
                    src: img.src,
                    alt: img.alt || '',
                    isBase64: img.src.startsWith('data:')
                }));
            }''')
            
        except Exception as e:
            logger.error(f"Error extracting page content: {e}")
            
        return content
        
    async def get_text(self, page: Page) -> str:
        """Extract all visible text from page."""
        try:
            return await page.evaluate('() => document.body.innerText')
        except Exception as e:
            logger.error(f"Error getting text: {e}")
            return ""
            
    async def get_html(self, page: Page) -> str:
        """Get full HTML content."""
        try:
            return await page.content()
        except Exception as e:
            logger.error(f"Error getting HTML: {e}")
            return ""
            
    async def get_links(self, page: Page) -> List[Dict[str, str]]:
        """Extract all anchor links."""
        try:
            return await page.evaluate('''() => {
                return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    href: a.href,
                    text: a.innerText.trim()
                }));
            }''')
        except Exception as e:
            logger.error(f"Error getting links: {e}")
            return []
            
    async def get_images(self, page: Page) -> List[Dict[str, Any]]:
        """Extract all images including base64 embedded ones."""
        try:
            return await page.evaluate('''() => {
                return Array.from(document.querySelectorAll('img')).map(img => ({
                    src: img.src,
                    alt: img.alt || '',
                    isBase64: img.src.startsWith('data:')
                }));
            }''')
        except Exception as e:
            logger.error(f"Error getting images: {e}")
            return []
            
    async def screenshot(self, page: Page, path: Optional[str] = None) -> bytes:
        """Take a screenshot of the page."""
        try:
            return await page.screenshot(path=path, full_page=True)
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return b""
            
    async def execute_script(self, page: Page, script: str) -> Any:
        """Execute JavaScript on the page."""
        try:
            return await page.evaluate(script)
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return None

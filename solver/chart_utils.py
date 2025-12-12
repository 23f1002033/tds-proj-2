"""
Chart Utilities Module
Chart generation and base64 export using matplotlib.
"""

import io
import base64
import logging
from typing import List, Optional, Dict, Any, Union
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates charts and exports as base64 images."""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: tuple = (10, 6)):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.figsize = figsize
        
    def histogram(self, data: List[float], title: str = "Histogram", 
                  xlabel: str = "Value", ylabel: str = "Frequency", bins: int = 20) -> str:
        """Create histogram and return base64 image."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return self._fig_to_base64(fig)
        
    def bar_chart(self, categories: List[str], values: List[float], 
                  title: str = "Bar Chart", xlabel: str = "", ylabel: str = "Value") -> str:
        """Create bar chart and return base64 image."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(categories, values, edgecolor='black', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return self._fig_to_base64(fig)
        
    def scatter_plot(self, x: List[float], y: List[float], 
                     title: str = "Scatter Plot", xlabel: str = "X", ylabel: str = "Y") -> str:
        """Create scatter plot and return base64 image."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(x, y, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return self._fig_to_base64(fig)
        
    def line_chart(self, x: List, y: List[float], 
                   title: str = "Line Chart", xlabel: str = "X", ylabel: str = "Y") -> str:
        """Create line chart and return base64 image."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x, y, marker='o', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return self._fig_to_base64(fig)
        
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


def create_histogram(data: List[float], **kwargs) -> str:
    """Convenience function for histogram."""
    return ChartGenerator().histogram(data, **kwargs)


def create_bar_chart(categories: List[str], values: List[float], **kwargs) -> str:
    """Convenience function for bar chart."""
    return ChartGenerator().bar_chart(categories, values, **kwargs)

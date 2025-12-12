"""
CSV Utilities Module
CSV/table loading, merging, and computation using pandas.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CSVProcessor:
    """Processes CSV/TSV/XLSX files for data operations."""
    
    def __init__(self):
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
    def load(self, filepath: str, name: Optional[str] = None) -> pd.DataFrame:
        """Load a data file into a DataFrame."""
        ext = filepath.lower().split('.')[-1]
        name = name or filepath
        
        try:
            if ext == 'csv':
                df = pd.read_csv(filepath)
            elif ext == 'tsv':
                df = pd.read_csv(filepath, sep='\t')
            elif ext in ['xlsx', 'xls']:
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)  # Try CSV as default
                
            df = self._clean_dataframe(df)
            self.dataframes[name] = df
            logger.info(f"Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} cols")
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
            
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names and convert types."""
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        return df
        
    def compute(self, df: pd.DataFrame, column: str, operation: str) -> Any:
        """Perform computation on a column."""
        if column not in df.columns:
            # Try fuzzy match
            matches = [c for c in df.columns if column.lower() in c.lower()]
            column = matches[0] if matches else df.select_dtypes(include=[np.number]).columns[0]
            
        col_data = pd.to_numeric(df[column], errors='coerce')
        
        operations = {
            'sum': col_data.sum,
            'mean': col_data.mean,
            'median': col_data.median,
            'max': col_data.max,
            'min': col_data.min,
            'count': col_data.count,
            'std': col_data.std,
        }
        
        if operation in operations:
            result = operations[operation]()
            return float(result) if pd.notna(result) else 0.0
        return None
        
    def groupby(self, df: pd.DataFrame, group_col: str, agg_col: str, operation: str = 'sum') -> Dict[str, Any]:
        """Group by a column and aggregate."""
        grouped = df.groupby(group_col)[agg_col].agg(operation)
        return grouped.to_dict()
        
    def merge(self, df1: pd.DataFrame, df2: pd.DataFrame, on: Optional[str] = None, how: str = 'inner') -> pd.DataFrame:
        """Merge two DataFrames."""
        if on is None:
            common = list(set(df1.columns) & set(df2.columns))
            on = common[0] if common else None
        if on:
            return pd.merge(df1, df2, on=on, how=how)
        return pd.concat([df1, df2], ignore_index=True)
        
    def filter(self, df: pd.DataFrame, column: str, condition: str, value: Any) -> pd.DataFrame:
        """Filter DataFrame based on condition."""
        ops = {'==': lambda x, v: x == v, '!=': lambda x, v: x != v,
               '>': lambda x, v: x > v, '<': lambda x, v: x < v,
               '>=': lambda x, v: x >= v, '<=': lambda x, v: x <= v}
        if condition in ops:
            return df[ops[condition](df[column], value)]
        return df
        
    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns."""
        return df.select_dtypes(include=[np.number]).columns.tolist()


def load_csv(filepath: str) -> pd.DataFrame:
    """Convenience function to load CSV."""
    return CSVProcessor().load(filepath)


def compute_stat(filepath: str, column: str, operation: str) -> float:
    """Convenience function to compute statistic."""
    processor = CSVProcessor()
    df = processor.load(filepath)
    return processor.compute(df, column, operation)

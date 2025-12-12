"""
ML Utilities Module
Machine learning inference using scikit-learn.
"""

import os
import logging
from typing import Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import joblib
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")


class MLPredictor:
    """Handles ML model loading and inference."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def load_model(self, path: str) -> bool:
        """Load a model from file."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available")
            return False
        try:
            self.model = joblib.load(path)
            self.model_path = path
            logger.info(f"Loaded model from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
            
    def predict(self, features: Union[List, np.ndarray]) -> Any:
        """Make prediction with loaded model."""
        if self.model is None:
            logger.warning("No model loaded")
            return None
        try:
            features = np.array(features).reshape(1, -1) if np.ndim(features) == 1 else np.array(features)
            return self.model.predict(features).tolist()
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
            
    def predict_proba(self, features: Union[List, np.ndarray]) -> Any:
        """Get prediction probabilities."""
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            return None
        try:
            features = np.array(features).reshape(1, -1) if np.ndim(features) == 1 else np.array(features)
            return self.model.predict_proba(features).tolist()
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None


def create_dummy_model(save_path: str):
    """Create and save a dummy model for testing."""
    if not SKLEARN_AVAILABLE:
        return False
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model.fit(X, y)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        logger.info(f"Saved dummy model to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create dummy model: {e}")
        return False

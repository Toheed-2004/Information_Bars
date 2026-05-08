import json
import numpy as np
from decimal import Decimal
from datetime import datetime
from bitpredict.common.logging import get_logger

logger = get_logger(__name__)

class RobustJSONEncoder(json.JSONEncoder):
    """Robust JSON json_encoder for types that commonly appear in bar data.
    
    Handles datetime, numpy types, Decimal, and other problematic types
    that might appear in bar configuration and state data.
    """
    def default(self, obj):
        try:
            # Handle NaN and Inf values
            if isinstance(obj, float) and np.isnan(obj):
                return None
            
            # Handle datetime objects
            if isinstance(obj, datetime):
                return obj.isoformat()
            
            # Handle numpy types
            if isinstance(obj, (np.integer, np.int64, np.int32, np.intc)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
                
            # Handle Decimal types
            if isinstance(obj, Decimal):
                return float(obj)
                
            # Handle sets (convert to list)
            if isinstance(obj, set):
                return list(obj)
                
            # Handle complex numbers
            if isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag, '_type': 'complex'}
                
            # Last resort: convert to string but log a warning
            logger.warning(f"JSON serialization: Converting unknown type {type(obj)} to string: {obj}")
            return str(obj)
            
        except Exception as e:
            logger.error(f"JSON serialization failed for {type(obj)}: {e}")
            return f"<serialization_error: {type(obj).__name__}>"
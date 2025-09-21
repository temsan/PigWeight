"""
Middleware для API
"""

from .cors import setup_cors
from .error import setup_error_handling

__all__ = ['setup_cors', 'setup_error_handling']
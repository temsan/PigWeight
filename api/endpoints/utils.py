"""
API Utilities - Common functions to reduce code duplication

Provides shared utilities for endpoint implementations:
- Error handling
- Response formatting
- Pagination
- Input validation
"""

import logging
from typing import Any, Dict, Optional, TypeVar, Callable
from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar('T')


class APIResponse:
    """Standardized API response builder"""
    
    @staticmethod
    def success(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Return successful response"""
        return {
            "status": "success",
            "message": message,
            "data": data
        }
    
    @staticmethod
    def error(code: int, message: str, detail: str = "") -> Dict[str, Any]:
        """Return error response"""
        return {
            "status": "error",
            "code": code,
            "message": message,
            "detail": detail
        }
    
    @staticmethod
    def list_response(
        items: list,
        total: int,
        limit: int,
        offset: int,
        message: str = "Success"
    ) -> Dict[str, Any]:
        """Return paginated list response"""
        return {
            "status": "success",
            "message": message,
            "items": items,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "pages": (total + limit - 1) // limit
            }
        }


def handle_api_error(
    func_name: str,
    error: Exception,
    status_code: int = 500,
    default_detail: str = "Internal server error"
) -> HTTPException:
    """
    Standardized error handling for API endpoints
    
    Args:
        func_name: Name of the function that errored
        error: The exception that occurred
        status_code: HTTP status code
        default_detail: Default error detail message
    
    Returns:
        HTTPException to raise
    """
    error_msg = str(error) or default_detail
    logger.error(f"[{func_name}] Error: {error_msg}")
    
    return HTTPException(
        status_code=status_code,
        detail=error_msg
    )


def validate_pagination(
    limit: int = 100,
    offset: int = 0,
    max_limit: int = 1000,
    min_limit: int = 1
) -> tuple[int, int]:
    """
    Validate and normalize pagination parameters
    
    Args:
        limit: Number of items
        offset: Offset from start
        max_limit: Maximum allowed limit
        min_limit: Minimum allowed limit
    
    Returns:
        (normalized_limit, normalized_offset)
    """
    limit = max(min_limit, min(limit, max_limit))
    offset = max(0, offset)
    return limit, offset


def create_list_endpoint(
    fetch_function: Callable,
    stream_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    endpoint_name: str = "list"
) -> Dict[str, Any]:
    """
    Generic list endpoint implementation
    
    Args:
        fetch_function: Async function to fetch items
        stream_id: Optional stream filter
        limit: Number of items
        offset: Offset
        endpoint_name: Name for logging
    
    Returns:
        Formatted list response
    """
    try:
        limit, offset = validate_pagination(limit, offset)
        items = fetch_function(stream_id=stream_id, limit=limit, offset=offset)
        total = len(items) if items else 0
        
        return APIResponse.list_response(
            items=items or [],
            total=total,
            limit=limit,
            offset=offset,
            message=f"Retrieved {total} {endpoint_name} items"
        )
    except Exception as e:
        logger.error(f"Error in list endpoint ({endpoint_name}): {e}")
        raise handle_api_error(f"list_{endpoint_name}", e)


# Common response templates
EMPTY_LIST_RESPONSE = {
    "items": [],
    "pagination": {
        "total": 0,
        "limit": 100,
        "offset": 0,
        "pages": 0
    }
}

EMPTY_STATS_RESPONSE = {
    "total": 0,
    "average": 0.0,
    "min": 0.0,
    "max": 0.0,
    "count": 0
}

DEFAULT_CONFIG_RESPONSE = {
    "model": "models/pig_yolo11-seg.v4.pt",
    "device": "auto",
    "conf_threshold": 0.30,
    "img_size": 960,
    "batch_size": 4,
    "fps": 25
}


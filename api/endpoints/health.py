"""
Health check endpoints
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health")
async def api_health():
    """Health check endpoint"""
    return {"status": "ok", "service": "pigweight"}

@router.get("/status")
async def api_status():
    """Detailed status endpoint"""
    return {
        "status": "ok",
        "service": "pigweight",
        "version": "3.0",
        "components": {
            "api": "ok",
            "processor": "ok",
            "websocket": "ok"
        }
    }
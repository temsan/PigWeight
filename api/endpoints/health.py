"""
Health check endpoints with worker monitoring
"""

import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health")
async def api_health():
    """Health check endpoint"""
    return {"status": "ok", "service": "pigweight"}

async def _check_av_worker() -> Dict[str, Any]:
    """Check av_worker health"""
    try:
        from api.app import get_av
        av_worker = get_av()
        
        # Get health stats
        health_stats = av_worker.get_health_stats()
        
        # Try ping
        try:
            ping_result = av_worker.ping()
            ping_status = "ok"
            ping_latency = 0.0  # Could measure actual latency
        except Exception as e:
            ping_status = "failed"
            ping_latency = -1.0
            logger.warning(f"AV worker ping failed: {e}")
        
        status = "ok" if health_stats['process_alive'] and ping_status == "ok" else "degraded"
        if health_stats['consecutive_failures'] >= health_stats['max_consecutive_failures']:
            status = "critical"
        
        return {
            "status": status,
            "process_alive": health_stats['process_alive'],
            "consecutive_failures": health_stats['consecutive_failures'],
            "max_consecutive_failures": health_stats['max_consecutive_failures'],
            "ping_status": ping_status,
            "ping_latency_ms": ping_latency,
            "last_health_check": health_stats['last_health_check']
        }
    except Exception as e:
        logger.error(f"Failed to check av_worker health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def _check_frame_broker() -> Dict[str, Any]:
    """Check FrameBroker health"""
    try:
        from core.frame_broker import FRAME_BROKER
        
        if FRAME_BROKER is None:
            return {"status": "not_available"}
        
        health = FRAME_BROKER.get_health_status()
        stats = FRAME_BROKER.get_stats()
        
        return {
            "status": health['status'],
            "total_streams": health['total_streams'],
            "total_subscribers": health['total_subscribers'],
            "success_rate": health['success_rate'],
            "memory_usage_mb": health['memory_usage_estimate_mb'],
            "last_cleanup": health['last_cleanup']
        }
    except Exception as e:
        logger.error(f"Failed to check FrameBroker health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def _check_processors() -> Dict[str, Any]:
    """Check UnifiedVideoProcessor health"""
    try:
        from core.processor import _PROCESSORS
        
        processor_stats = {}
        total_processors = len(_PROCESSORS)
        active_processors = 0
        
        for stream_id, processor in _PROCESSORS.items():
            is_active = processor.is_active if hasattr(processor, 'is_active') else False
            if is_active:
                active_processors += 1
            
            processor_stats[stream_id] = {
                "active": is_active,
                "backend": getattr(processor.model_adapter, 'backend', 'unknown') if hasattr(processor, 'model_adapter') else 'unknown'
            }
        
        status = "ok" if active_processors == total_processors else "degraded" if active_processors > 0 else "critical"
        
        return {
            "status": status,
            "total_processors": total_processors,
            "active_processors": active_processors,
            "processors": processor_stats
        }
    except Exception as e:
        logger.error(f"Failed to check processors health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def _check_model_adapter() -> Dict[str, Any]:
    """Check ModelAdapter health"""
    try:
        from services.model_adapter import ModelAdapter
        from core.config import CONFIG
        
        # Create a test adapter to check model availability
        model_path = getattr(CONFIG, "MODEL_PATH", "")
        if not model_path:
            return {"status": "not_configured", "error": "MODEL_PATH not set"}
        
        try:
            adapter = ModelAdapter(model_path, device="auto")
            performance_stats = adapter.get_performance_stats()
            
            return {
                "status": "ok",
                "backend": adapter.backend,
                "device": adapter.device,
                "optimal_dtype": str(adapter.optimal_dtype),
                "total_inferences": performance_stats.get('total_inferences', 0),
                "model_path": model_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model_path": model_path
            }
    except Exception as e:
        logger.error(f"Failed to check ModelAdapter health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/status")
async def api_status():
    """Detailed status endpoint with worker monitoring"""
    start_time = time.time()
    
    # Check all components
    av_worker_health = await _check_av_worker()
    frame_broker_health = await _check_frame_broker()
    processors_health = await _check_processors()
    model_adapter_health = await _check_model_adapter()
    
    # Determine overall status
    component_statuses = [
        av_worker_health.get('status', 'unknown'),
        frame_broker_health.get('status', 'unknown'),
        processors_health.get('status', 'unknown'),
        model_adapter_health.get('status', 'unknown')
    ]
    
    if any(status == 'critical' for status in component_statuses):
        overall_status = 'critical'
    elif any(status in ['error', 'degraded'] for status in component_statuses):
        overall_status = 'degraded'
    elif all(status == 'ok' for status in component_statuses):
        overall_status = 'ok'
    else:
        overall_status = 'unknown'
    
    check_duration = time.time() - start_time
    
    return {
        "status": overall_status,
        "service": "pigweight",
        "version": "3.0",
        "timestamp": time.time(),
        "check_duration_ms": round(check_duration * 1000, 2),
        "components": {
            "av_worker": av_worker_health,
            "frame_broker": frame_broker_health,
            "processors": processors_health,
            "model_adapter": model_adapter_health
        }
    }

@router.get("/health/av-worker")
async def av_worker_health():
    """AV Worker specific health check"""
    return await _check_av_worker()

@router.get("/health/frame-broker")
async def frame_broker_health():
    """FrameBroker specific health check"""
    return await _check_frame_broker()

@router.get("/health/processors")
async def processors_health():
    """Processors specific health check"""
    return await _check_processors()

@router.get("/health/model-adapter")
async def model_adapter_health():
    """ModelAdapter specific health check"""
    return await _check_model_adapter()

@router.get("/metrics")
async def api_metrics():
    """Performance metrics endpoint"""
    try:
        metrics = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - _start_time if '_start_time' in globals() else 0
        }
        
        # AV Worker metrics
        try:
            from api.app import get_av
            av_worker = get_av()
            health_stats = av_worker.get_health_stats()
            metrics["av_worker"] = {
                "consecutive_failures": health_stats['consecutive_failures'],
                "health_check_interval": health_stats['health_check_interval']
            }
        except Exception:
            pass
        
        # FrameBroker metrics
        try:
            from core.frame_broker import FRAME_BROKER
            if FRAME_BROKER:
                broker_health = FRAME_BROKER.get_health_status()
                metrics["frame_broker"] = {
                    "total_streams": broker_health['total_streams'],
                    "total_subscribers": broker_health['total_subscribers'],
                    "success_rate": broker_health['success_rate'],
                    "memory_usage_mb": broker_health['memory_usage_estimate_mb']
                }
        except Exception:
            pass
        
        # Processor metrics
        try:
            from core.processor import _PROCESSORS
            metrics["processors"] = {
                "total_count": len(_PROCESSORS),
                "active_count": sum(1 for p in _PROCESSORS.values() if getattr(p, 'is_active', False))
            }
        except Exception:
            pass
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/dashboard")
async def monitoring_dashboard():
    """Monitoring dashboard data"""
    try:
        # Get comprehensive system status
        status_data = await api_status()
        
        # Add additional dashboard-specific data
        dashboard_data = {
            "overview": {
                "status": status_data["status"],
                "service": status_data["service"],
                "version": status_data["version"],
                "uptime_seconds": time.time() - _start_time,
                "last_check": status_data["timestamp"]
            },
            "components": status_data["components"],
            "alerts": []
        }
        
        # Generate alerts based on component status
        for component_name, component_data in status_data["components"].items():
            component_status = component_data.get("status", "unknown")
            
            if component_status == "critical":
                dashboard_data["alerts"].append({
                    "level": "critical",
                    "component": component_name,
                    "message": f"{component_name} is in critical state",
                    "timestamp": time.time()
                })
            elif component_status == "error":
                dashboard_data["alerts"].append({
                    "level": "error", 
                    "component": component_name,
                    "message": f"{component_name} has errors: {component_data.get('error', 'Unknown error')}",
                    "timestamp": time.time()
                })
            elif component_status == "degraded":
                dashboard_data["alerts"].append({
                    "level": "warning",
                    "component": component_name,
                    "message": f"{component_name} is degraded",
                    "timestamp": time.time()
                })
        
        # Add performance indicators
        dashboard_data["performance"] = {
            "frame_broker_success_rate": status_data["components"]["frame_broker"].get("success_rate", 0),
            "av_worker_failures": status_data["components"]["av_worker"].get("consecutive_failures", 0),
            "active_processors": status_data["components"]["processors"].get("active_processors", 0),
            "total_streams": status_data["components"]["frame_broker"].get("total_streams", 0)
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Initialize start time for uptime calculation
_start_time = time.time()
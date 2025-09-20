
from fastapi import APIRouter, UploadFile, File, JSONResponse
from pathlib import Path

# This is a temporary solution and might cause circular imports.
# We are importing from the main app module during this refactoring phase.
from api.app import UPLOAD_DIR, ocv_probe_file, perf_logger

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Always save under the original safe filename (overwrite if exists)
        safe_name = "".join(c for c in (file.filename or "") if c.isalnum() or c in "._-") or "upload.bin"
        dst = UPLOAD_DIR / safe_name
        content = await file.read()
        try:
            # Skip rewrite if file exists with same size to avoid SSD churn
            if not (dst.exists() and dst.stat().st_size == len(content)):
                with open(dst, "wb") as buffer:
                    buffer.write(content)
        except Exception:
            # Fallback to simple write
            with open(dst, "wb") as buffer:
                buffer.write(content)
        meta = ocv_probe_file(str(dst))
        resp = {"file_path": str(dst)}
        if meta and not meta.get("error"):
            resp.update({
                "duration": float(meta.get("duration", 0.0) or 0.0),
                "fps": float(meta.get("fps", 0.0) or 0.0),
                "frame_count": int(meta.get("frame_count", 0) or 0)
            })
        return resp
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

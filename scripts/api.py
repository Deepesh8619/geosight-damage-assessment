"""
FastAPI REST API for GeoSight Damage Assessment.

This turns the project from a CLI tool into a web service that
anyone can call over HTTP.

Run:
    pip install fastapi uvicorn python-multipart
    python3 scripts/api.py

Then:
    POST http://localhost:8000/assess
    (upload pre + post images as multipart form data)

    GET http://localhost:8000/health
    (check if service is running)
"""

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Query
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("Install dependencies: pip install fastapi uvicorn python-multipart")
    sys.exit(1)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GeoSight API",
    description="Satellite-based post-disaster building damage assessment",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (loaded once at startup)
pipeline = None
assessor = None


@app.on_event("startup")
def load_models():
    """Load models once at startup — not per-request."""
    global pipeline, assessor

    from src.inference.assessor import GeoSightAssessor

    seg_ckpt = os.environ.get("SEG_CHECKPOINT", "checkpoints/segmentation/best.pth")
    dmg_ckpt = os.environ.get("DMG_CHECKPOINT", "checkpoints/damage/best.pth")

    seg_path = seg_ckpt if Path(seg_ckpt).exists() else None
    dmg_path = dmg_ckpt if Path(dmg_ckpt).exists() else None

    assessor = GeoSightAssessor(
        seg_checkpoint=seg_path,
        dmg_checkpoint=dmg_path,
        tile_size=512,
        tile_overlap=64,
    )

    logger.info("GeoSight API ready.")
    if not seg_path:
        logger.warning("No segmentation checkpoint — using random weights")
    if not dmg_path:
        logger.warning("No damage checkpoint — using random weights")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": assessor is not None,
    }


@app.post("/assess")
async def assess_damage(
    pre_image: UploadFile = File(..., description="Pre-disaster image (PNG or TIFF)"),
    post_image: UploadFile = File(..., description="Post-disaster image (PNG or TIFF)"),
    event_name: Optional[str] = Query(None, description="Event identifier"),
):
    """
    Run full damage assessment on a pre/post image pair.

    Upload two images (PNG or GeoTIFF) and receive:
      - Disaster type prediction
      - Building damage counts
      - Population impact estimates
      - Economic loss estimates
      - Spatial analysis (epicentre, clusters, gradient)
      - Response protocol

    Returns JSON report.
    """
    if assessor is None:
        raise HTTPException(500, "Models not loaded")

    start = time.time()

    # Save uploads to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        pre_path  = os.path.join(tmpdir, f"pre{Path(pre_image.filename).suffix}")
        post_path = os.path.join(tmpdir, f"post{Path(post_image.filename).suffix}")

        with open(pre_path, "wb") as f:
            f.write(await pre_image.read())
        with open(post_path, "wb") as f:
            f.write(await post_image.read())

        output_dir = os.path.join(tmpdir, "output")

        try:
            report = assessor.assess(
                pre_image_path=pre_path,
                post_image_path=post_path,
                output_dir=output_dir,
                event_name=event_name or Path(pre_image.filename).stem,
                save_leaflet_map=False,  # HTML map not useful via API
            )
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            raise HTTPException(500, f"Assessment failed: {str(e)}")

    elapsed = time.time() - start
    report["processing_time_seconds"] = round(elapsed, 2)

    return JSONResponse(content=report)


@app.post("/assess/quick")
async def quick_assess(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
):
    """
    Lightweight assessment — returns only key stats, no file outputs.
    Faster for API integrations where you just need numbers.
    """
    if assessor is None:
        raise HTTPException(500, "Models not loaded")

    start = time.time()

    from PIL import Image

    pre_bytes  = await pre_image.read()
    post_bytes = await post_image.read()

    pre_arr  = np.array(Image.open(io.BytesIO(pre_bytes)).convert("RGB"))
    post_arr = np.array(Image.open(io.BytesIO(post_bytes)).convert("RGB"))

    result = assessor.pipeline.assess_full_scene(
        pre_arr, post_arr, tile_size=512, overlap=64, batch_size=4,
    )

    elapsed = time.time() - start

    return {
        "disaster_type": result.get("disaster_type", {}),
        "impact": result.get("impact_report", {}),
        "stats": result.get("stats", {}),
        "processing_time_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting GeoSight API on http://localhost:8000")
    logger.info("Docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

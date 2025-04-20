"""FastAPI app and API endpoints."""

from typing import List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from .segmentation import segment_image

app = FastAPI(title="PaliGemma Segmentation API")

# temp while testing
MODEL_PATH = "google/paligemma2-3b-mix-448"


class SegmentRequest(BaseModel):
    prompt: str
    image_url: HttpUrl = None


class MaskResult(BaseModel):
    mask: List[List[float]]
    coordinates: Tuple[int, int, int, int]


@app.get("/")
async def root():
    return {"message": "Welcome to the PaliGemma Segmentation API!"}


@app.post("/segment", response_model=List[MaskResult])
async def segment(
    prompt: str = Form(...),
    # image_file: UploadFile = File(None),
    image_url: str = Form(None),
    # model_path: str = Form(None),
    # vae_checkpoint: str = Form(None),
):
    # if not (image_file or image_url):
    #     raise HTTPException(400, "Provide either `file` or `url`.")

    try:
        masks = segment_image(MODEL_PATH, image_url, prompt)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {str(e)}")
    finally:
        None

    serialized = []
    for m in masks:
        serialized.append({"bbox": m["coordinates"], "mask": m["mask"].tolist()})

    return JSONResponse(content={"masks": serialized})

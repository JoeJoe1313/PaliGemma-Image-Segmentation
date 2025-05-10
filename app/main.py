import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from app.segmentation import segment_image

app = FastAPI(title="PaliGemma Segmentation API")

MODEL_ID = os.getenv("MODEL_ID", "google/paligemma2-3b-mix-448")


class MaskResult(BaseModel):
    mask: bytes
    coordinates: Tuple[int, int, int, int]


@app.get("/")
async def root():
    return {
        "message": "Welcome to the PaliGemma Segmentation API!",
        "model": MODEL_ID,
    }


@app.post("/segment", response_model=List[MaskResult])
async def segment(
    prompt: str = Form(...),
    image_url: Optional[str] = Form(default=None),
    image_file: Optional[UploadFile] = File(default=None),
):
    """Segment image based on text prompt.

    Args:
        prompt (str): Text description of objects to segment.
        image_url (str, optional): URL of the image to segment.
        image_file (UploadFile, optional): Uploaded image file to segment.

    Raises:
        HTTPException: When no image source is provided, when both image sources are provided, or when segmentation fails.

    Returns:
        List[MaskResult]: List of segmentation masks with their coordinates.
    """
    if not (image_file or image_url):
        raise HTTPException(
            status_code=400, detail="Provide either an image file or image URL."
        )
    if image_file and image_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either an image file or image URL, not both.",
        )

    try:
        masks = segment_image(MODEL_ID, prompt, image_url, image_file)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {str(e)}")

    return JSONResponse(content={"message": masks})

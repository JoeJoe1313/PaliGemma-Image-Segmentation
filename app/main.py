import os
from typing import List, Tuple

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from app.segmentation import segment_image

app = FastAPI(title="PaliGemma Segmentation API")

MODEL_ID = os.getenv("MODEL_ID", "google/paligemma2-3b-mix-448")


class SegmentRequest(BaseModel):
    prompt: str
    image_url: HttpUrl = None


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
    # image_file: UploadFile = File(None),
    image_url: str = Form(None),
):
    """Segment image based on text prompt.

    Args:
        prompt (str, optional): Text description of objects to segment. Defaults to Form(...).
        image_url (str, optional): URL of the image to segment. Defaults to Form(None).
        model_path (str, optional): Optional custom model path. Defaults to Form(None).

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    # if not (image_file or image_url):
    #     raise HTTPException(400, "Provide either `file` or `url`.")

    try:
        masks = segment_image(MODEL_ID, image_url, prompt)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {str(e)}")
    finally:
        None

    return JSONResponse(content={"message": masks})

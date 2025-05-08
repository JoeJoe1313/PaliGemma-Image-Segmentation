import os
import sys
from typing import List, Tuple

from config import ensure_dirs_exist, get_model_path
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from segmentation import segment_image

print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("File directory:", os.path.dirname(__file__))
print(os.path.dirname(__file__))
ensure_dirs_exist()

app = FastAPI(title="PaliGemma Segmentation API")

MODEL_PATH = get_model_path()


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
        "default_model": MODEL_PATH,
    }


@app.post("/segment", response_model=List[MaskResult])
async def segment(
    prompt: str = Form(...),
    # image_file: UploadFile = File(None),
    image_url: str = Form(None),
    model_path: str = Form(None),
    # vae_checkpoint: str = Form(None),
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

    model_to_use = model_path or MODEL_PATH
    try:
        masks = segment_image(model_to_use, image_url, prompt)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {str(e)}")
    finally:
        None

    return JSONResponse(content={"message": masks})

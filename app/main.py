import logging
import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.segmentation import segment_image

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

app = FastAPI(title="PaliGemma Segmentation API")

MODEL_ID = os.getenv("MODEL_ID", "google/paligemma2-3b-mix-448")


class MaskResult(BaseModel):
    """Model for segmentation mask result."""

    mask: str  # base64 encoded mask image
    coordinates: Tuple[int, int, int, int]

    model_config = {
        "json_schema_extra": {
            "example": {
                "mask": "base64_encoded_mask_data",
                "coordinates": [62, 254, 140, 348],
            }
        }
    }


class SegmentationResponse(BaseModel):
    """API response model for segmentation results."""

    image: str
    masks: List[MaskResult]

    model_config = {
        "json_schema_extra": {
            "example": {
                "image": "base64_encoded_image_data",
                "masks": [
                    {
                        "mask": "base64_encoded_mask_data",
                        "coordinates": [62, 254, 140, 348],
                    }
                ],
            }
        }
    }


@app.get("/")
async def root():
    return {
        "message": "Welcome to the PaliGemma Segmentation API!",
    }


@app.post("/segment", response_model=SegmentationResponse)
async def segment(
    prompt: str = Form(...),
    image_url: Optional[str] = Form(default=None),
    image_file: Optional[UploadFile] = File(default=None),
    model_id: Optional[str] = Form(default=None),
):
    """Segment image based on text prompt.

    Args:
        prompt (str): Text description of objects to segment.
        image_url (str, optional): URL of the image to segment.
        image_file (UploadFile, optional): Uploaded image file to segment.
        model_id (str, optional): Model ID to use for segmentation. Defaults to the configured MODEL_ID.

    Raises:
        HTTPException: When no image source is provided, when both image sources are provided, or when segmentation fails.

    Returns:
        SegmentationResponse: List of segmentation masks with their coordinates.
    """
    if not (image_file or image_url):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "Provide either an image file or image URL.",
            },
        )
    if image_file and image_url:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "Provide either an image file or image URL, not both.",
            },
        )

    used_model_id = model_id or MODEL_ID
    try:
        result = segment_image(used_model_id, prompt, image_url, image_file)
        masks = [
            MaskResult(mask=mask["mask"], coordinates=mask["coordinates"])
            for mask in result["masks"]
        ]
        return SegmentationResponse(image=result["image"], masks=masks)
    except Exception as e:
        log.error(f"Segmentation failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"Segmentation failed: {str(e)}"},
        )

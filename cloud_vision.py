from pathlib import Path

from google.cloud import vision


def gen_image(image_path: Path) -> vision.Image:
    with open(image_path, "rb") as f:
        content = f.read()
    return vision.Image(content=content)


def cloud_vision_ocr(image_path: Path) -> str:
    """Provides a quick start example for Cloud Vision."""

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    image = gen_image(image_path)

    # Performs label detection on the image file
    response = client.text_detection(image=image)

    labels = response.label_annotations

    return response.full_text_annotation.text

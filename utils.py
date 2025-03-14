import base64
from pathlib import Path


def process_image(image: Path) -> str:
    ext = image.suffix.lower()
    type_: str
    if ext == ".png":
        type_ = "image/png"
    elif ext == ".jpg" or ext == ".jpeg":
        type_ = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(image, "rb") as f:
        img = f.read()
        img_base64 = base64.b64encode(img).decode()
    return f"data:{type_};base64,{img_base64}"

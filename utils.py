import asyncio
import base64
import random
from pathlib import Path
import functools


def process_image(image: Path) -> str:
    return f"https://promoting-commitments-undefined-arrested.trycloudflare.com/{image.name}"

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


def retrynize(func):
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for i in range(5):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    print(f"Error on attempt {i+1}: {e}")
            raise Exception("Failed after 5 retries")
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for i in range(5):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error on attempt {i+1}: {e}")
            raise Exception("Failed after 5 retries")
        return sync_wrapper

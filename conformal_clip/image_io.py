from PIL import Image
from urllib.parse import urlparse
from io import BytesIO
import requests
import numpy as np

def load_image(image_path_or_url: str, mode: str = "RGB", timeout: int = 20) -> Image.Image:
    """Load an image from disk or URL and convert to a mode.

    Args:
        image_path_or_url: Local path or HTTP(S) URL to the image.
        mode: Desired mode for PIL conversion, for example "RGB" or "L".
        timeout: Request timeout in seconds when fetching URLs.

    Returns:
        A PIL Image object in the requested mode.

    Raises:
        ValueError: If the input is not a string.
        IOError: If loading fails from path or URL.
    """
    if not isinstance(image_path_or_url, str):
        raise ValueError("image_path_or_url must be a string path or URL")

    scheme = urlparse(image_path_or_url).scheme.lower()
    if scheme in {"http", "https"}:
        try:
            resp = requests.get(image_path_or_url, timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert(mode)
        except requests.RequestException as e:
            raise IOError(f"Failed to load image from URL: {e}") from e
    else:
        try:
            return Image.open(image_path_or_url).convert(mode)
        except Exception as e:
            raise IOError(f"Failed to load image from path: {e}") from e


# Helper: sample without replacement and return (chosen, remaining)
def sample_urls(urls, k, rng):
    urls = np.array(urls)
    if k > len(urls):
        raise ValueError(f"Requested {k} but only {len(urls)} available.")
    idx = rng.choice(len(urls), size=k, replace=False)
    chosen = urls[idx].tolist()
    remaining = np.delete(urls, idx).tolist()
    return chosen, remaining
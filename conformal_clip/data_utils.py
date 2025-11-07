from pathlib import Path
from importlib.util import find_spec

def get_textile_base_dir() -> Path:
    """
    Returns the base directory of the textile dataset if available.
    Prefers the optional conformal-clip-data package. Falls back to ./data for dev.
    """
    # Optional data package
    if find_spec("conformal_clip_data") is not None:
        from conformal_clip_data import textile_simulated_root
        return Path(textile_simulated_root())

    # Repo-relative fallback for dev
    repo = Path("data") / "textile_images" / "simulated"
    if (repo / "nominal").is_dir():
        return repo

    raise FileNotFoundError(
        "Dataset not found. Install with `pip install conformal-clip[data]` "
        "or place ./data/textile_images/simulated in your project."
    )

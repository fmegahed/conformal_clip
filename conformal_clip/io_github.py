from typing import List, Optional


def get_image_urls(
    repo_owner: str,
    repo_name: str,
    base_path: str,
    subfolder: str,
    branch: str = "main",
    file_extensions: Optional[List[str]] = None,
    timeout: int = 20
) -> List[str]:
    """Return raw file URLs in a GitHub repo subfolder.

    Args:
        repo_owner: Repository owner.
        repo_name: Repository name.
        base_path: Path inside the repository that contains subfolders.
        subfolder: Subfolder to list files from.
        branch: Branch name. Defaults to "main".
        file_extensions: Optional list of allowed extensions like ["jpg", "png"].
        timeout: Request timeout in seconds.

    Returns:
        List of raw.githubusercontent.com URLs for files in the subfolder.

    Raises:
        ImportError: If the optional ``requests`` package is not installed.
    """
    try:
        import requests  # optional; this module is only imported when the user calls it
    except ImportError as e:
        raise ImportError(
            "get_image_urls requires the optional 'requests' package. "
            "Install it with `pip install requests`."
        ) from e

    api_url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
        f"{base_path.strip('/')}/{subfolder.strip('/')}?ref={branch}"
    )

    try:
        resp = requests.get(api_url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[get_image_urls] Failed for {subfolder}: {e}")
        return []

    items = resp.json()
    urls: List[str] = []
    for item in items:
        if item.get("type") != "file":
            continue
        name = item.get("name", "")
        if file_extensions is not None:
            ok = any(name.lower().endswith(ext.lower()) for ext in file_extensions)
            if not ok:
                continue

        raw_url = (
            f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/"
            f"{base_path.strip('/')}/{subfolder.strip('/')}/{name}"
        )
        urls.append(raw_url)
    return urls

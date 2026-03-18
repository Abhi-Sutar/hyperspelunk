from urllib.parse import urlsplit
import os
import config

ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
IGNORED_PATTERNS = config.IGNORED_PATTERNS


def is_valid_link(full_url: str, domain: str, path_prefix: str) -> bool:
    """Checks if a URL is safe to crawl."""
    parsed_url = urlsplit(full_url)
    # 1. Did it leave the university domain?
    if parsed_url.netloc != domain:
        return False
    # 2. Did it leak outside the target folder?
    if not parsed_url.path.startswith(config.ALLOWED_PATH_PREFIX):
        return False
    # 3. Does it have an extension, and is it allowed?
    ext = os.path.splitext(parsed_url.path)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        return False
    # 4. Is it a frame trap?
    for pattern in IGNORED_PATTERNS:
        if pattern in parsed_url.path:
            return False

    return True
import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks based on word count."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size  # Avoid infinite loop if overlap >= chunk_size

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    logger.debug(f"Split {len(words)} words into {len(chunks)} chunks")
    return chunks
"""
indexing/file_router.py

Sorts a file into "image/video" or "text/document" based on its extension.
This is Phase 1 plumbing — just classification, no actual processing yet
(that's text_pipeline.py / vision_pipeline.py).
"""

from pathlib import Path

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv"
}

TEXT_DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".csv"
}

UNSUPPORTED_EXTENSIONS = {
    ".doc", ".rtf",
    ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    ".html", ".htm", ".xml", ".json", ".yaml", ".yml",
    ".eml", ".msg", ".epub",
    ".py", ".js", ".ts", ".java", ".cs", ".cpp", ".h", ".sql", ".log",
    ".tif", ".tiff", ".heic", ".heif", ".avif", ".svg",
    ".mp3", ".wav", ".m4a",
    ".zip", ".7z", ".rar",
}


def route_file(file_path):
    """
    Return 'image', 'text/document', 'video', 'unsupported', or 'unknown'.
    """
    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in TEXT_DOCUMENT_EXTENSIONS:
        return "text/document"
    elif ext in UNSUPPORTED_EXTENSIONS:
        return "unsupported"
    else:
        return "unknown"
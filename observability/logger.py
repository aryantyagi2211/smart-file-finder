"""
observability/logger.py

Simple human-readable activity log. Appends timestamped lines to
observability/activity.log for things like "file indexed", "query
received", etc. This is the plain-English log; for structured per-query
data used in debugging/eval, see observability/tracer.py.
"""

from pathlib import Path
from datetime import datetime

from app_paths import get_base_dir

LOG_PATH = get_base_dir() / "observability" / "activity.log"

def log(message, log_path=LOG_PATH):
    """Append a timestamped message to the activity log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
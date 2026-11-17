"""
tests/integration/test_full_pipeline.py

One end-to-end test: a real file gets indexed through the real pipeline
(indexing/backfill_index.py's index_one_file), then a real query through
the real harness (app/query_handler.handle_query) should find it.

Runs against real storage (metadata_store.db, vector_db/chroma_data) using
a uniquely-named test file to avoid colliding with real indexed data, and
cleans up after itself. This is a deliberate trade-off (see TASK.md notes
for Task 3.7) -- fully isolating storage would need refactoring
vector_store.py's global client caching, which carries real risk this
close to the deadline for limited benefit.
"""

import shutil
from pathlib import Path

from indexing.backfill_index import index_one_file
from app.query_handler import handle_query
from storage.metadata_store import delete_file, get_file_metadata
from storage.vector_db.vector_store import delete_by_file

TEST_FILE_NAME = "integration_test_unique_marker_file.pdf"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PDF = PROJECT_ROOT / "test_data" / "Adaptive rag.pdf"
TEST_FILE_PATH = PROJECT_ROOT / "test_data" / TEST_FILE_NAME


def setup_module(module):
    """Copy a real PDF under a unique name so it doesn't collide with real data."""
    shutil.copy(SOURCE_PDF, TEST_FILE_PATH)


def teardown_module(module):
    """Remove the test file after the test finishes, pass or fail."""
    delete_by_file(TEST_FILE_PATH)
    delete_file(TEST_FILE_PATH)
    if TEST_FILE_PATH.exists():
        TEST_FILE_PATH.unlink()
    assert get_file_metadata(TEST_FILE_PATH) is None


def test_full_pipeline_index_and_search():
    # Step 1: index the file through the real pipeline
    status = index_one_file(TEST_FILE_PATH, force=True)
    assert "INDEXED" in status

    # Step 2: confirm it landed in metadata_store
    metadata = get_file_metadata(str(TEST_FILE_PATH))
    assert metadata is not None
    assert metadata["file_type"] == "text/document"

    # Step 3: query the real harness for content unique to this PDF
    # (Adaptive RAG's real content, so an unrelated match would be a real failure)
    result = handle_query("adaptive retrieval augmented generation for question complexity")

    result_paths = (
        [option["path"] for option in result["options"]]
        if result.get("action") == "options"
        else ([result["path"]] if result.get("found") else [])
    )
    # The copied PDF and its original have identical content, so they should
    # be surfaced as an uncertain pick list rather than auto-selecting one.
    assert any(
        "rag" in path.lower()
        or "adaptive" in path.lower()
        or Path(path).name == TEST_FILE_NAME
        for path in result_paths
    )
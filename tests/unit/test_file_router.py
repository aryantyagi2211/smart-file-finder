"""
tests/unit/test_file_router.py

Basic unit tests for indexing/file_router.py.
"""

from indexing.file_router import route_file


def test_route_image_extensions():
    assert route_file("photo.png") == "image"
    assert route_file("clip.mp4") == "video"
    assert route_file("screenshot.webp") == "image"


def test_route_document_extensions():
    assert route_file("report.pdf") == "text/document"
    assert route_file("notes.md") == "text/document"
    assert route_file("essay.docx") == "text/document"


def test_route_unknown_extension():
    assert route_file("archive.zip") == "unsupported"
    assert route_file("data.xyz") == "unknown"


def test_route_known_unsupported_extensions():
    assert route_file("spreadsheet.xlsx") == "unsupported"
    assert route_file("document.rtf") == "unsupported"
    assert route_file("audio.mp3") == "unsupported"


def test_route_case_insensitive():
    assert route_file("PHOTO.PNG") == "image"
    assert route_file("REPORT.PDF") == "text/document"


def test_unsupported_file_skips_before_embedding(tmp_path, monkeypatch):
    from indexing import backfill_index

    monkeypatch.setattr(backfill_index, "get_file_metadata", lambda path: None)

    def fail_if_embedded(text):
        raise AssertionError("unsupported files must not be embedded")

    monkeypatch.setattr(backfill_index, "embed_text", fail_if_embedded)
    result = backfill_index.index_one_file(tmp_path / "report.xlsx")

    assert result.startswith("[UNSUPPORTED - unsupported]")
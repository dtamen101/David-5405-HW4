import pytest

from study_graph.extract import extract_chunks, resolve_source
from study_graph.security import DocumentParseError, InvalidInputError, NotFoundError


@pytest.mark.parametrize("name", ["../../.env", "/etc/passwd", "C:\\Windows\\win.ini", "sub/../../x.md", ""])
def test_resolve_source_blocks_escapes(settings, name):
    with pytest.raises((InvalidInputError, NotFoundError)):
        resolve_source(settings.sources_dir, name, 25)


def test_resolve_source_rejects_unsupported_type(settings):
    (settings.sources_dir / "script.py").write_text("print(1)")
    with pytest.raises(InvalidInputError):
        resolve_source(settings.sources_dir, "script.py", 25)


def test_resolve_source_missing_file_message_has_no_path(settings):
    with pytest.raises(NotFoundError) as exc:
        resolve_source(settings.sources_dir, "missing.pdf", 25)
    assert str(settings.sources_dir) not in exc.value.public_message


def test_extracts_each_format(settings):
    src = settings.sources_dir
    assert len(extract_chunks(src / "week2_lean_operations.pptx")) == 8
    assert len(extract_chunks(src / "reading_theory_of_constraints.pdf")) == 3
    md = extract_chunks(src / "reading_agile_vs_waterfall.md")
    assert [c.title for c in md][1] == "Waterfall"


def test_speaker_notes_are_included(settings):
    chunks = extract_chunks(settings.sources_dir / "week2_lean_operations.pptx")
    assert any("Speaker notes:" in c.text for c in chunks)


def test_corrupt_file_raises_safe_error(settings):
    bad = settings.sources_dir / "bad.pptx"
    bad.write_bytes(b"nope")
    with pytest.raises(DocumentParseError) as exc:
        extract_chunks(bad)
    assert str(settings.sources_dir) not in exc.value.public_message

import pytest

from study_graph.security import InvalidInputError, NotFoundError


def test_ingest_counts_and_reingest_replaces(kb):
    first = kb.ingest_document("week2_lean_operations.pptx")
    assert first["notes_created"] == 8
    again = kb.ingest_document("week2_lean_operations.pptx")
    assert again["replaced_previous_notes"] == 8
    assert kb.graph_overview()["notes"] == 8


@pytest.mark.parametrize("query, expected", [
    ("limiting work in progress", "Pull systems and kanban"),
    ("roles in scrum", "Agile and Scrum"),
    ("five focusing steps for the constraint", "The five focusing steps"),
])
def test_search_finds_relevant_note(loaded_kb, query, expected):
    titles = [r["title"] for r in loaded_kb.search(query, 3)["results"]]
    assert expected in titles


def test_search_can_filter_by_course(loaded_kb):
    loaded_kb.add_note("Other class", "Thermodynamics entropy lecture", course="MAE 3230")
    results = loaded_kb.search("entropy", 5, course="MAE 3230")["results"]
    assert [r["title"] for r in results] == ["Other class"]


def test_add_note_wikilinks_create_edges(loaded_kb):
    out = loaded_kb.add_note("My summary", "Links to [[Takt time]] and [[Does not exist]]", tags=["exam-2"])
    assert [w["title"] for w in out["wikilinks"]] == ["Takt time"]
    assert out["unresolved_wikilinks"] == ["Does not exist"]
    graph = loaded_kb.explore(note_id=out["note_id"])
    assert any(link["relation"] == "wikilink" for link in graph["links"])


def test_explore_concept_and_errors(loaded_kb):
    result = loaded_kb.explore(concept="bottleneck")
    assert len(result["notes"]) >= 2
    with pytest.raises(NotFoundError):
        loaded_kb.explore(concept="quantum chromodynamics")
    with pytest.raises(InvalidInputError):
        loaded_kb.explore()
    with pytest.raises(NotFoundError):
        loaded_kb.explore(note_id=99999)


@pytest.mark.parametrize("kwargs", [
    {"title": "", "body": "x"},
    {"title": "x" * 500, "body": "x"},
    {"title": "ok", "body": "x", "tags": ["bad;tag<script>"]},
])
def test_add_note_validation(kb, kwargs):
    with pytest.raises(InvalidInputError):
        kb.add_note(**kwargs)


def test_sql_metacharacters_are_stored_as_plain_text(kb):
    evil = "x'); DROP TABLE notes; --"
    note_id = kb.add_note(evil, evil)["note_id"]
    assert kb.get_note(note_id).title == evil
    assert kb.graph_overview()["notes"] == 1

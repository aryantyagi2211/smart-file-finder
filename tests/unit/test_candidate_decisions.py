from types import SimpleNamespace

from agent.candidate_utils import group_candidates_by_file
from agent.harness import AgentAction, decide_candidates, run_query
from app.command_parser import Command, should_auto_open
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication


def _candidate(path, score):
    return SimpleNamespace(path=path, score=score, metadata={"chunk_text": path})


def _limits(tmp_path, max_reformulations=2):
    path = tmp_path / "harness_limits.json"
    path.write_text(
        '{"max_reformulations": %d, "timeout_seconds": 2, '
        '"confidence_threshold": 0.55, "ambiguity_margin": 0.07, '
        '"options_floor": 0.48, "max_options": 3}' % max_reformulations,
        encoding="utf-8",
    )
    return str(path)


def test_group_candidates_by_file_keeps_each_file_once_with_best_score():
    grouped = group_candidates_by_file([
        _candidate(r"C:\docs\one.txt", 0.61),
        _candidate(r"C:\docs\two.txt", 0.58),
        _candidate(r"C:\docs\ONE.txt", 0.72),
    ])

    assert len(grouped) == 2
    assert grouped[0].path == r"C:\docs\ONE.txt"
    assert grouped[0].score == 0.72
    assert grouped[1].path == r"C:\docs\two.txt"


def test_decision_is_confident_when_threshold_and_margin_pass():
    decision = decide_candidates([
        _candidate("best.pdf", 0.72),
        _candidate("second.pdf", 0.61),
    ])

    assert decision.decision == "confident"
    assert len(decision.candidates) == 1


def test_close_scores_return_options():
    decision = decide_candidates([
        _candidate("first.pdf", 0.71),
        _candidate("second.pdf", 0.68),
    ])

    assert decision.decision == "options"
    assert [candidate.path for candidate in decision.candidates] == [
        "first.pdf", "second.pdf"
    ]


def test_below_options_floor_is_no_match():
    decision = decide_candidates([
        _candidate("weak.pdf", 0.44),
        _candidate("weaker.pdf", 0.40),
    ])

    assert decision.decision == "no_match"


def test_only_one_candidate_above_threshold_is_confident():
    decision = decide_candidates([_candidate("only.pdf", 0.56)])

    assert decision.decision == "confident"
    assert decision.candidates[0].path == "only.pdf"


def test_options_are_limited_to_three_files():
    decision = decide_candidates([
        _candidate(f"{index}.txt", 0.54 - index * 0.01)
        for index in range(5)
    ])

    assert decision.decision == "options"
    assert len(decision.candidates) == 3


def test_exhausted_reformulation_returns_options_when_candidate_passes_floor(tmp_path):
    candidates = [
        [_candidate("weak.txt", 0.42)],
        [_candidate("best.txt", 0.53), _candidate("other.txt", 0.51)],
    ]
    calls = []

    def retrieve(query):
        calls.append(query)
        return candidates[len(calls) - 1]

    def corrective(found, query):
        return SimpleNamespace(
            confident=False,
            best_score=found[0].score,
            reformulated_query="second query" if len(calls) == 1 else None,
            strategy="refine",
        )

    result = run_query(
        "first query",
        retrieve,
        corrective,
        lambda found, query: found[0],
        lambda candidate, query: f"About {candidate.path}",
        config_path=_limits(tmp_path, max_reformulations=1),
    )

    assert result["action"] == AgentAction.OPTIONS
    assert [option["path"] for option in result["options"]] == [
        "best.txt", "other.txt"
    ]
    assert calls == ["first query", "second query"]


def test_open_command_does_not_auto_open_options():
    assert should_auto_open(Command.OPEN, "options") is False
    assert should_auto_open(Command.OPEN, "answer") is True
    assert should_auto_open(Command.FIND, "answer") is False


def test_options_card_keyboard_opens_selected_file_and_escape_closes(
    tmp_path, monkeypatch
):
    from app.result_card import OptionsResultCard

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    opened = []
    monkeypatch.setattr("app.result_card.os.startfile", opened.append, raising=False)

    card = OptionsResultCard([
        {"path": str(first), "score": 0.6, "explanation": "First explanation"},
        {"path": str(second), "score": 0.55, "explanation": "Second explanation"},
    ])
    card.show()
    app.processEvents()

    QTest.keyClick(card, Qt.Key.Key_2)
    assert opened == [str(second.resolve())]

    QTest.keyClick(card, Qt.Key.Key_Escape)
    assert not card.isVisible()
    card.deleteLater()
    app.processEvents()

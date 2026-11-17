from pathlib import Path
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


def test_same_filename_in_different_folders_is_kept_as_two_files():
    grouped = group_candidates_by_file([
        _candidate(r"C:\Users\me\Downloads\captureit.png", 0.49),
        _candidate(r"D:\project\test_data\captureit.png", 0.43),
    ])

    assert len(grouped) == 2
    assert {candidate.score for candidate in grouped} == {0.49, 0.43}
    assert {candidate.path for candidate in grouped} == {
        r"C:\Users\me\Downloads\captureit.png",
        r"D:\project\test_data\captureit.png",
    }


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


def _popup_group(tmp_path, count=3):
    from app.result_card import OptionsPopupGroup

    from PyQt6.QtWidgets import QWidget
    anchor = QWidget()
    anchor.setGeometry(100, 450, 520, 84)
    anchor.show()
    options = []
    for rank in range(count):
        path = tmp_path / f"option-{rank + 1}.txt"
        path.write_text(f"option {rank + 1}", encoding="utf-8")
        options.append({
            "path": str(path),
            "score": 0.7 - rank * 0.05,
            "explanation": f"Explanation for option {rank + 1}",
        })
    return OptionsPopupGroup(options, anchor=anchor), anchor


def _qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_three_popup_cards_are_ranked_and_fit_screen(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path)
    group.show()
    app.processEvents()

    assert len(group.cards) == 3
    by_rank = {card.rank: card for card in group.cards}
    assert by_rank[2].x() < by_rank[1].x() < by_rank[3].x()
    screen = anchor.screen().availableGeometry()
    for card in group.cards:
        rect = card.frameGeometry()
        assert screen.left() <= rect.left()
        assert screen.top() <= rect.top()
        assert rect.right() <= screen.right()
        assert rect.bottom() <= screen.bottom()
    group.close()
    anchor.close()


def test_two_popup_cards_put_best_on_left(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path, count=2)
    group.show()
    app.processEvents()

    by_rank = {card.rank: card for card in group.cards}
    assert len(group.cards) == 2
    assert by_rank[1].x() < by_rank[2].x()
    screen = anchor.screen().availableGeometry()
    for card in group.cards:
        rect = card.frameGeometry()
        assert screen.left() <= rect.left()
        assert rect.right() <= screen.right()
    group.close()
    anchor.close()


def test_open_choice_closes_group_and_emits_once(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path)
    opened = []
    group.choice_made.connect(
        lambda action, path: opened.append((action, path, all(
            not card.isVisible() for card in group.cards
        )))
    )
    group.show()
    app.processEvents()

    rank_two = next(card for card in group.cards if card.rank == 2)
    rank_two.open_button.click()

    assert opened == [("open", str(Path(tmp_path / "option-2.txt").resolve()), True)]
    group.close()
    anchor.close()


def test_number_keys_open_matching_rank_not_screen_position(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    for key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3):
        group, anchor = _popup_group(tmp_path)
        choices = []
        group.choice_made.connect(
            lambda action, path: choices.append((action, Path(path).name))
        )
        group.show()
        app.processEvents()

        QTest.keyClick(group.cards[0], key)
        app.processEvents()
        rank = key.value - Qt.Key.Key_0.value
        assert choices == [("open", f"option-{rank}.txt")]
        assert all(not card.isVisible() for card in group.cards)
        group.close()
        anchor.close()


def test_escape_closes_all_popup_cards(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path)
    group.show()
    app.processEvents()

    QTest.keyClick(group.cards[0], Qt.Key.Key_Escape)
    app.processEvents()

    assert all(not card.isVisible() for card in group.cards)
    anchor.close()


def test_close_button_closes_all_popup_cards(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path)
    group.show()
    app.processEvents()

    group.cards[-1].close_button.click()
    app.processEvents()

    assert all(not card.isVisible() for card in group.cards)
    anchor.close()


def test_focus_loss_does_not_close_popup_group(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path)
    group.show()
    app.processEvents()

    group.cards[0].clearFocus()
    app.processEvents()

    assert all(card.isVisible() for card in group.cards)
    group.close()
    anchor.close()


def test_popup_card_labels_have_no_outlined_boxes(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QLabel

    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path)
    group.show()
    app.processEvents()

    for card in group.cards:
        for label in card.findChildren(QLabel):
            assert "border:" not in label.styleSheet().casefold()
    group.close()
    anchor.close()

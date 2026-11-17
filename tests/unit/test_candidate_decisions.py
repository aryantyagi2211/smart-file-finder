from pathlib import Path
from types import SimpleNamespace

from agent.candidate_utils import (
    group_candidates_by_file,
    select_options_by_score,
)
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


def test_option_filter_returns_one_two_or_three_cards_and_caps_at_three():
    options = [
        {"path": f"{rank}.txt", "score": score}
        for rank, score in enumerate([0.70, 0.68, 0.66, 0.65, 0.61])
    ]

    assert len(select_options_by_score(options, 0.64, 0.05)) == 3
    assert len(select_options_by_score(options[:3], 0.67, 0.05)) == 2
    assert len(select_options_by_score(options[:3], 0.69, 0.05)) == 1
    assert [
        option["path"]
        for option in select_options_by_score(options, 0.64, 0.05)
    ] == ["0.txt", "1.txt", "2.txt"]


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


def _popup_group(tmp_path, count=3, file_names=None):
    from app.result_card import OptionsPopupGroup

    from PyQt6.QtWidgets import QWidget
    anchor = QWidget()
    anchor.setGeometry(100, 450, 520, 84)
    anchor.show()
    options = []
    for rank in range(count):
        filename = (
            file_names[rank]
            if file_names is not None
            else f"option-{rank + 1}.txt"
        )
        path = tmp_path / filename
        path.write_text(f"option {rank + 1}", encoding="utf-8")
        options.append({
            "path": str(path),
            "score": 0.70 - rank * 0.01,
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
    assert by_rank[1].y() < by_rank[2].y() < by_rank[3].y()
    screen = anchor.screen().availableGeometry()
    for card in group.cards:
        rect = card.frameGeometry()
        assert screen.left() <= rect.left()
        assert screen.top() <= rect.top()
        assert rect.right() <= screen.right()
        assert rect.bottom() <= screen.bottom()
    assert (
        len(group.cards) * group.CARD_HEIGHT
        + (len(group.cards) - 1) * group.GAP
        <= 768
    )
    group.close()
    anchor.close()


def test_two_popup_cards_put_best_on_left(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path, count=2)
    group.show()
    app.processEvents()

    by_rank = {card.rank: card for card in group.cards}
    assert len(group.cards) == 2
    assert by_rank[1].y() < by_rank[2].y()
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
    assert rank_two.open_button.isVisible()
    assert rank_two.show_button.isVisible()
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


def test_close_button_closes_only_its_card(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path)
    group.show()
    app.processEvents()

    group.cards[-1].close_button.click()
    app.processEvents()

    by_rank = {card.rank: card for card in group.cards}
    assert not by_rank[3].isVisible()
    assert by_rank[1].isVisible()
    assert by_rank[2].isVisible()
    group.close()
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


def test_popup_card_render_has_opaque_dark_background(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path, count=1)
    assert group.cards[0].header_title.text() == "Not sure - is this it?"
    group.show()
    app.processEvents()

    image = group.cards[0].grab().toImage()
    center = image.pixelColor(image.width() // 2, image.height() // 2)
    assert center.alpha() == 255
    assert (center.red(), center.green(), center.blue()) == (18, 19, 24)
    group.close()
    anchor.close()


def test_confident_result_card_render_has_opaque_dark_background(
    tmp_path, monkeypatch
):
    from app.result_card import ResultCard

    app = _qt_app(monkeypatch)
    path = tmp_path / "confident.txt"
    path.write_text("text", encoding="utf-8")
    card = ResultCard(str(path), 0.9, "A confident result")
    card.show()
    app.processEvents()

    image = card.grab().toImage()
    background = image.pixelColor(image.width() - 10, image.height() // 2)
    assert background.alpha() == 255
    assert (
        background.red(), background.green(), background.blue()
    ) == (18, 19, 24)
    card.close()


def test_popup_card_can_be_dragged_and_is_clamped_to_screen(
    tmp_path, monkeypatch
):
    from PyQt6.QtCore import QPoint

    app = _qt_app(monkeypatch)
    group, anchor = _popup_group(tmp_path, count=1)
    group.show()
    app.processEvents()
    card = group.cards[0]
    original = card.pos()

    QTest.mousePress(card, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    QTest.mouseMove(card, QPoint(120, 100), delay=20)
    QTest.mouseRelease(card, Qt.MouseButton.LeftButton, pos=QPoint(120, 100))
    app.processEvents()

    screen = card.screen().availableGeometry()
    assert card.pos() != original
    assert screen.left() <= card.x() <= screen.right() - card.width() + 1
    assert screen.top() <= card.y() <= screen.bottom() - card.height() + 1
    group.close()
    anchor.close()


def test_filename_middle_elision_preserves_extension(tmp_path, monkeypatch):
    app = _qt_app(monkeypatch)
    full_name = "captureit_12-11-2026_at_22-37-18.png"
    group, anchor = _popup_group(
        tmp_path, count=1, file_names=[full_name]
    )
    card = group.cards[0]
    rendered = card.filename_label.text()

    assert rendered.endswith(".png")
    assert "…" in rendered
    assert card.filename_label.toolTip() == full_name
    assert card.filename_label.fontMetrics().horizontalAdvance(rendered) <= (
        card.filename_label.width()
    )
    group.close()
    anchor.close()

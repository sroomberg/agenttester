"""Tests for agenttester.questions."""

from __future__ import annotations

import threading
import time

from agenttester.questions import QuestionRegistry


class TestQuestionRegistry:
    def test_ask_and_respond(self) -> None:
        registry = QuestionRegistry()
        result_holder: list[str] = []

        def asker():
            result_holder.append(registry.ask("model-a", "what color?"))

        t = threading.Thread(target=asker)
        t.start()
        time.sleep(0.05)

        assert len(registry.pending()) == 1
        assert registry.pending()[0].model_name == "model-a"
        assert registry.pending()[0].question == "what color?"

        assert registry.respond("model-a", "blue")
        t.join(timeout=1)
        assert result_holder == ["blue"]
        assert len(registry.pending()) == 0

    def test_respond_to_nonexistent_model_returns_false(self) -> None:
        registry = QuestionRegistry()
        assert registry.respond("nobody", "hello") is False

    def test_timeout_returns_none(self) -> None:
        registry = QuestionRegistry()
        result = registry.ask("model-b", "anything?", timeout=0.05)
        assert result is None

    def test_cancel_all_unblocks_waiting(self) -> None:
        registry = QuestionRegistry()
        result_holder: list = []

        def asker():
            result_holder.append(registry.ask("model-c", "hey?"))

        t = threading.Thread(target=asker)
        t.start()
        time.sleep(0.05)

        registry.cancel_all()
        t.join(timeout=1)
        assert result_holder[0] is None

    def test_multiple_models_waiting(self) -> None:
        registry = QuestionRegistry()
        results: dict[str, str] = {}

        def asker(name: str, q: str):
            results[name] = registry.ask(name, q)

        t1 = threading.Thread(target=asker, args=("m1", "q1"))
        t2 = threading.Thread(target=asker, args=("m2", "q2"))
        t1.start()
        t2.start()
        time.sleep(0.05)

        assert len(registry.pending()) == 2

        registry.respond("m1", "answer1")
        registry.respond("m2", "answer2")
        t1.join(timeout=1)
        t2.join(timeout=1)

        assert results == {"m1": "answer1", "m2": "answer2"}

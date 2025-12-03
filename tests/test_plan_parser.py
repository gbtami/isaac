from __future__ import annotations

from isaac.agent.brain.planner import parse_plan_from_text


def test_parse_plan_without_header():
    plan = "1. do A\n2. do B"
    parsed = parse_plan_from_text(plan)
    assert parsed is not None
    assert getattr(parsed, "entries", None)


def test_parse_plan_inline_after_header():
    plan = "Plan: 1. step one 2. step two"
    parsed = parse_plan_from_text(plan)
    assert parsed is not None
    assert getattr(parsed, "entries", None)
    assert len(parsed.entries) == 2

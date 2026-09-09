"""
Tests for the golden_tasks loader module.
Run with: pytest tests/test_golden_tasks.py -v
"""
import pytest
from diffprompt.core.golden_tasks import load_golden_tasks


def test_load_yaml(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text(
        """
tasks:
  - input: "What is the capital of France?"
    golden_answer: "The capital of France is Paris."
    checks:
      - type: keyword
        keywords: ["Paris"]
        weight: 2.0
  - input: "Return the user's age as JSON."
    weight: 1.5
    checks:
      - type: json_schema
        schema: {type: object, required: [age]}
""",
        encoding="utf-8",
    )
    tasks = load_golden_tasks(str(p))
    assert len(tasks) == 2
    assert tasks[0].input == "What is the capital of France?"
    assert tasks[0].checks[0].keywords == ["Paris"]
    assert tasks[1].weight == 1.5


def test_load_jsonl(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        '{"input": "hi", "checks": [{"type": "keyword", "keywords": ["hello"]}]}\n'
        '{"input": "bye", "checks": [{"type": "keyword", "keywords": ["goodbye"]}]}\n',
        encoding="utf-8",
    )
    tasks = load_golden_tasks(str(p))
    assert len(tasks) == 2
    assert tasks[1].input == "bye"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_golden_tasks(str(tmp_path / "nope.yaml"))


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "tasks.txt"
    p.write_text("tasks: []", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_golden_tasks(str(p))


def test_empty_task_list_raises(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text("tasks: []", encoding="utf-8")
    with pytest.raises(ValueError, match="no golden tasks"):
        load_golden_tasks(str(p))


def test_task_with_no_scoring_signal_raises(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text("tasks:\n  - input: \"hi\"\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_golden_tasks(str(p))

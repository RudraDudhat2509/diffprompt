"""
Tests for the checks module.
Run with: pytest tests/test_checks.py -v
"""
from diffprompt.core.checks import score_check
from diffprompt.models import Check, CheckType


def test_regex_match():
    check = Check(type=CheckType.REGEX, pattern=r"\bParis\b")
    assert score_check(check, "The capital is Paris.") == 1.0
    assert score_check(check, "The capital is Berlin.") == 0.0


def test_regex_ignore_case():
    check = Check(type=CheckType.REGEX, pattern="paris", ignore_case=True)
    assert score_check(check, "PARIS is the capital.") == 1.0


def test_keyword_all_required_by_default():
    check = Check(type=CheckType.KEYWORD, keywords=["Paris", "France"])
    assert score_check(check, "Paris is the capital of France.") == 1.0
    assert score_check(check, "Paris is a city.") == 0.0


def test_keyword_match_any():
    check = Check(type=CheckType.KEYWORD, keywords=["Paris", "London"], match_any=True)
    assert score_check(check, "I visited Paris last year.") == 1.0
    assert score_check(check, "I visited Tokyo last year.") == 0.0


def test_keyword_case_insensitive():
    check = Check(type=CheckType.KEYWORD, keywords=["paris"])
    assert score_check(check, "PARIS is beautiful.") == 1.0


def test_json_schema_valid():
    check = Check(
        type=CheckType.JSON_SCHEMA,
        schema={"type": "object", "required": ["age"], "properties": {"age": {"type": "integer"}}},
    )
    assert score_check(check, '{"age": 21}') == 1.0


def test_json_schema_invalid_shape():
    check = Check(
        type=CheckType.JSON_SCHEMA,
        schema={"type": "object", "required": ["age"], "properties": {"age": {"type": "integer"}}},
    )
    assert score_check(check, '{"age": "twenty-one"}') == 0.0


def test_json_schema_not_json():
    check = Check(type=CheckType.JSON_SCHEMA, schema={"type": "object"})
    assert score_check(check, "not json at all") == 0.0


def test_numeric_eq_within_tolerance():
    check = Check(type=CheckType.NUMERIC, expected=42, tolerance=1)
    assert score_check(check, "The answer is 42.5") == 1.0
    assert score_check(check, "The answer is 50") == 0.0


def test_numeric_gte():
    check = Check(type=CheckType.NUMERIC, expected=18, comparator="gte")
    assert score_check(check, "Age: 25") == 1.0
    assert score_check(check, "Age: 10") == 0.0


def test_numeric_lte():
    check = Check(type=CheckType.NUMERIC, expected=100, comparator="lte")
    assert score_check(check, "Total: 50") == 1.0
    assert score_check(check, "Total: 150") == 0.0


def test_numeric_extract_pattern():
    check = Check(
        type=CheckType.NUMERIC, expected=21, comparator="gte",
        extract_pattern=r'"age"\s*:\s*(-?\d+)',
    )
    assert score_check(check, '{"age": 25, "id": 3}') == 1.0
    assert score_check(check, '{"age": 10, "id": 3}') == 0.0


def test_numeric_no_number_present():
    check = Check(type=CheckType.NUMERIC, expected=42)
    assert score_check(check, "no digits here") == 0.0

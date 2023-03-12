import pytest
from opof.registry import parse_callable


def test_callable_invalid():
    with pytest.raises(ValueError):
        parse_callable("TestFunc")
    with pytest.raises(ValueError):
        parse_callable("TestFunct['TestArgs']")
    with pytest.raises(ValueError):
        parse_callable("")


def test_callable_valid():
    assert parse_callable("GC[Table, BKPIECE1, LinearProjection[2]]") == (
        "GC",
        "'Table', 'BKPIECE1', 'LinearProjection[2]'",
    )

    assert parse_callable(
        "TestFunc[1.0, None, 3, [5, [Nested, [LLList, RightHere[5.0]]]], AndHere[]]"
    ) == (
        "TestFunc",
        "1.0, None, 3, [5, ['Nested', ['LLList', 'RightHere[5.0]']]], 'AndHere[]'",
    )

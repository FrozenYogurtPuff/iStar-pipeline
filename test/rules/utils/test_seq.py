from src.rules.utils.seq import is_entity_type_ok


def test_is_entity_type_ok():
    assert is_entity_type_ok("Actor", "B-Actor")
    assert is_entity_type_ok("Both", "Actor")
    assert not is_entity_type_ok("Both", "O")

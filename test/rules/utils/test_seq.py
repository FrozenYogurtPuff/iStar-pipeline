from src.rules.utils.seq import get_s2b_idx, is_entity_type_ok


def test_is_entity_type_ok():
    assert is_entity_type_ok("Actor", "B-Actor")
    assert is_entity_type_ok("Both", "Actor")
    assert not is_entity_type_ok("Both", "O")


def test_get_s2b_idx():
    s2b = [[0], [1], [2, 3], [3], [4], [5], [6]]
    idx = [1, 2]
    assert get_s2b_idx(s2b, idx) == [1, 2]

    s2b = [[0], [1], [2, 3], [3], [4], [5], [6]]
    idx = [0, 2]
    assert get_s2b_idx(s2b, idx) == [0, 2]

    s2b = [[0], [1], [2, 3], [3], [4], [5], [6]]
    idx = [0, 3]
    assert get_s2b_idx(s2b, idx) == [0, 3]

    s2b = [[0], [1], [2, 3], [3], [4], [5], [6]]
    idx = [2]
    assert get_s2b_idx(s2b, idx) == [2, 2]

    s2b = [[0], [1], [2, 3], [3], [4], [5], [6]]
    idx = [6]
    assert get_s2b_idx(s2b, idx) == [6, 6]

    s2b = [[0], [1], [2, 3], [3], [4], [5], [6]]
    idx = [0]
    assert get_s2b_idx(s2b, idx) == [0, 0]

    s2b = [[0, 1], [1], [2, 3], [3], [4], [5], [6]]
    idx = [1]
    assert get_s2b_idx(s2b, idx) == [1, 1]

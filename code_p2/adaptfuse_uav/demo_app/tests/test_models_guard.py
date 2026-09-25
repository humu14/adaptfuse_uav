import pytest

from engine.models import DETECTOR_NAMES, check_detector_names


def test_rejects_swapped_names():
    with pytest.raises(ValueError, match="swapped"):
        check_detector_names({0: "fire", 1: "smoke"}, DETECTOR_NAMES, "old.pt", exact=True)


def test_accepts_correct_names():
    check_detector_names({0: "smoke", 1: "fire"}, DETECTOR_NAMES, "best.pt", exact=True)


def test_exact_rejects_extra_classes():
    with pytest.raises(ValueError):
        check_detector_names({0: "smoke", 1: "fire", 2: "x"}, DETECTOR_NAMES, "b.pt", exact=True)


def test_person_subset_ok():
    coco = {i: f"c{i}" for i in range(80)}
    coco[0] = "person"
    check_detector_names(coco, {0: "person"}, "coco.pt", exact=False)


from engine.models import resolve_members


def test_resolve_members_one_missing(tmp_path, capsys):
    (tmp_path / "a.pth").write_bytes(b"x")
    got = resolve_members(("a.pth", "b.pth"), tmp_path)
    assert got == [tmp_path / "a.pth"]
    assert "b.pth" in capsys.readouterr().out


def test_resolve_members_none_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        resolve_members(("a.pth", "b.pth"), tmp_path)

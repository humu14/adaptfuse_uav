import numpy as np
import pytest
import torch

from engine.hazard import extract_zones, input_size, softmax_bias, to_coarse
from engine.labels import HAZARD_EVENTS, HAZARD_NAMES, SOUND_GROUPS
from engine.models import load_shards, save_shards
from engine.pipeline import event_label, headline, sound_events


def test_input_size_keeps_aspect_in_patch_multiples():
    assert input_size(640, 360) == (400, 224)
    assert input_size(1920, 1080) == (400, 224)
    assert input_size(720, 1280) == (224, 400)
    assert input_size(300, 300) == (224, 224)
    assert input_size(4000, 500) == (448, 224)          # capped at 2:1
    for w, h in (input_size(640, 360), input_size(720, 1280)):
        assert w % 16 == 0 and h % 16 == 0


def test_to_coarse_sums_events_into_scene_classes():
    p = np.zeros(len(HAZARD_NAMES), np.float32)
    p[HAZARD_NAMES.index("flood")] = 0.6
    p[HAZARD_NAMES.index("collapsed building")] = 0.3
    p[HAZARD_NAMES.index("smoke")] = 0.1
    c = to_coarse(p)
    assert c.shape == (4,)
    assert c[2] == pytest.approx(0.9) and c[1] == pytest.approx(0.1)
    assert to_coarse(np.stack([p, p])).shape == (2, 4)


def test_softmax_bias_shifts_toward_biased_class():
    z = np.zeros(len(HAZARD_NAMES))
    b = np.zeros(len(HAZARD_NAMES)); b[3] = 2.0
    p = softmax_bias(z, b)
    assert p.sum() == pytest.approx(1.0, abs=1e-6) and p.argmax() == 3
    assert softmax_bias(z, None) == pytest.approx(np.full(len(z), 1 / len(z)))


def _dense_with_block(event: str, gh=14, gw=25, y=(3, 9), x=(5, 14), p=0.9):
    E = len(HAZARD_NAMES)
    d = np.full((gh, gw, E), (1 - p) / (E - 1), np.float32)
    d[..., 0] = p                                        # background: normal
    d[..., HAZARD_NAMES.index(event)] = (1 - p) / (E - 1)
    blk = np.full(E, (1 - p) / (E - 1), np.float32); blk[HAZARD_NAMES.index(event)] = p
    d[y[0]:y[1], x[0]:x[1]] = blk
    return d


def test_extract_zones_finds_the_region_when_scene_supports_it():
    d = _dense_with_block("flood")
    scene = np.zeros(len(HAZARD_NAMES), np.float32); scene[HAZARD_NAMES.index("flood")] = 0.8; scene[0] = 0.2
    zones, grid = extract_zones(d, scene)
    assert len(zones) == 1
    z = zones[0]
    assert z["cls"] == "flood water" and z["event"] == "flood"
    x1, y1, x2, y2 = z["box"]
    assert x1 == pytest.approx(5 / 25, abs=0.05) and x2 == pytest.approx(14 / 25, abs=0.05)
    assert y1 == pytest.approx(3 / 14, abs=0.08) and y2 == pytest.approx(9 / 14, abs=0.08)
    assert grid["w"] == 25 and grid["h"] == 14 and len(grid["cls"]) == 25 * 14
    assert HAZARD_NAMES.index("flood") in grid["cls"]


def test_extract_zones_is_gated_by_the_scene():
    d = _dense_with_block("collapsed building")
    scene = np.zeros(len(HAZARD_NAMES), np.float32); scene[0] = 0.9; scene[HAZARD_NAMES.index("collapsed building")] = 0.1
    zones, grid = extract_zones(d, scene)
    assert zones == [] and set(grid["cls"]) == {-1}


def test_extract_zones_ignores_specks():
    d = _dense_with_block("landslide", y=(4, 5), x=(4, 5))          # a single patch
    scene = np.zeros(len(HAZARD_NAMES), np.float32); scene[HAZARD_NAMES.index("landslide")] = 1.0
    assert extract_zones(d, scene)[0] == []


def test_event_label_stays_inside_winning_scene_class():
    scene = np.array([0.1, 0.2, 0.6, 0.1], np.float32)             # collapse / flood wins
    ev = np.zeros(len(HAZARD_NAMES), np.float32)
    ev[HAZARD_NAMES.index("fire")] = 0.5                           # strongest event overall, wrong class
    ev[HAZARD_NAMES.index("landslide")] = 0.3
    ev[HAZARD_NAMES.index("flood")] = 0.2
    e = event_label(scene, ev)
    assert e["label"] == "landslide" and e["group"] == "collapse / flood" and e["conf"] == pytest.approx(0.6)
    assert HAZARD_EVENTS[e["index"]]["coarse"] == 2


def test_sound_headline_prefers_active_hazard_over_context():
    scores = np.zeros(len(SOUND_GROUPS), np.float32)
    names = list(SOUND_GROUPS)
    scores[names.index("speech")] = 0.9
    scores[names.index("siren / alarm")] = 0.35
    thr = {"siren / alarm": 0.3}
    ev = sound_events(scores, thr)
    assert headline(ev) == {"name": "siren / alarm", "score": pytest.approx(0.35), "hazard": True}
    scores[names.index("siren / alarm")] = 0.1
    assert headline(sound_events(scores, thr))["name"] == "speech"
    assert headline(sound_events(np.zeros(len(SOUND_GROUPS)), thr))["name"] == "background noise"


def test_sound_groups_index_valid_audioset_classes():
    for g in SOUND_GROUPS.values():
        assert all(0 <= i < 527 for i in g["idx"])
    assert sum(bool(g.get("victim")) for g in SOUND_GROUPS.values()) == 1


def test_shards_round_trip(tmp_path):
    sd = {"a": torch.randn(300, 300), "b": torch.randn(10), "c": torch.arange(5)}
    files = save_shards(sd, tmp_path, "t", max_bytes=200_000)
    assert len(files) >= 2
    back = load_shards(tmp_path, "t")
    assert set(back) == set(sd) and all(torch.equal(back[k], sd[k]) for k in sd)

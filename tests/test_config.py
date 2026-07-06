from src.config import load_config, resolve_point


def test_resolve_point_scales_ratio_to_pixels():
    cfg = load_config("config.json")
    phone = cfg.phones[0]
    x, y = resolve_point(cfg.anchors_ratio["ball_center"], phone)
    assert (x, y) == (540, 2124)


def test_timing_is_a_range():
    cfg = load_config("config.json")
    lo, hi = cfg.timing["encounter_load_ms"]
    assert lo < hi

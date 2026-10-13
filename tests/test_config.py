"""Pruebas mínimas de Fase 0: config.yaml es coherente y los splits no se solapan."""
import pandas as pd

from src.config import load_config


def test_config_loads_and_has_required_blocks():
    cfg = load_config()
    for key in ("project", "zone", "dates", "label", "clustering", "rl", "paths"):
        assert key in cfg


def test_temporal_splits_are_ordered_and_disjoint():
    s = load_config()["dates"]["splits"]
    tr, va, te = (tuple(map(pd.Timestamp, s[k])) for k in ("train", "val", "test"))
    assert tr[0] < tr[1] < va[0] < va[1] < te[0] < te[1]


def test_label_quantiles_valid():
    q = load_config()["label"]["quantiles"]
    assert len(q) == 2 and 0 < q[0] < q[1] < 1

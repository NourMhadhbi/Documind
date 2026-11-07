import pytest

from benchmarks.calibrate_guard import (
    best_threshold,
    candidate_thresholds,
    evaluate_threshold,
    labelled_distances,
    separation_auc,
)


def test_labelled_distances_marks_traps_and_misses_as_unanswerable():
    rows = [
        {"trap": False, "rank": 1, "top1_distance": 0.3},  # à traiter
        {"trap": False, "rank": None, "top1_distance": 0.8},  # page ratée -> à refuser
        {"trap": True, "rank": None, "top1_distance": 0.95},  # piège -> à refuser
        {"trap": False, "rank": 2, "top1_distance": None},  # sans distance : ignorée
        {
            "trap": False,
            "rank": 1,
            "top1_distance": 0.2,
            "error": "x",
        },  # en erreur : ignorée
    ]
    answerable, unanswerable = labelled_distances(rows)
    assert answerable == [0.3] and unanswerable == [0.8, 0.95]


def test_separation_auc_perfect_none_and_ties():
    assert separation_auc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert separation_auc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert separation_auc([0.5], [0.5]) == 0.5
    assert separation_auc([], [0.5]) is None


def test_best_threshold_separates_perfectly_separable_data():
    best = best_threshold([0.2, 0.3, 0.4], [0.7, 0.8])
    assert 0.4 < best["threshold"] < 0.7
    assert best["balanced"] == 1.0


def test_best_threshold_prefers_keeping_good_questions_on_ties():
    # Les deux groupes se chevauchent : plusieurs seuils ont la même exactitude équilibrée.
    best = best_threshold([0.3, 0.6], [0.5, 0.9])
    for t in candidate_thresholds([0.3, 0.6, 0.5, 0.9]):
        other = evaluate_threshold(t, [0.3, 0.6], [0.5, 0.9])
        assert (round(best["balanced"], 9), best["answerable_kept"]) >= (
            round(other["balanced"], 9),
            other["answerable_kept"],
        ) or best["balanced"] > other["balanced"]


def test_evaluate_threshold_counts():
    r = evaluate_threshold(0.5, [0.2, 0.6], [0.4, 0.9])
    assert r["answerable_kept"] == 0.5 and r["unanswerable_refused"] == 0.5 and r["balanced"] == 0.5


def test_candidate_thresholds_cover_refuse_nothing_and_everything():
    ts = candidate_thresholds([0.2, 0.5, 0.5, 0.9])  # 3 valeurs distinctes
    assert ts[0] < 0.2 and ts[-1] > 0.9
    assert len(ts) == 4  # 2 points médians + 2 extrêmes
    assert ts[1] == pytest.approx(0.35) and ts[2] == pytest.approx(0.7)

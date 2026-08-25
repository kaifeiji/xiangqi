from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from data_utils import apply_move, encode_fen, iccs_to_indices, position_to_fen
from prepare_pikafish import export_records


STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def action_id(move: str) -> int:
    start, end = iccs_to_indices(move)
    return 90 * start + end


def annotation_record(
    *, game_id: str, split: str, ply: int, fen: str,
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1, "game_id": game_id, "split": split, "ply": ply,
        "fen": fen, "move": "C3-C4",
        "teacher": {
            "score_kind": "cp", "score": 50, "bestmove": "C3-C4",
            "candidates": candidates if candidates is not None else [
                {"rank": 1, "move": "C3-C4", "score_kind": "cp", "score": 50},
                {"rank": 2, "move": "H2-H3", "score_kind": "cp", "score": 40},
            ],
        },
    }


def test_export_records_writes_current_view_soft_policy_inputs(tmp_path: Path) -> None:
    summary = export_records(
        [annotation_record(game_id="game-3", split="train", ply=0, fen=STARTING_FEN)],
        tmp_path, max_games=None, max_candidates=5,
    )
    dataset = tmp_path / "dataset"
    candidate_ids = np.load(dataset / "train-000-candidate_action_ids.npy")
    legal_ids = np.load(dataset / "train-000-legal_action_ids.npy")
    legal_offsets = np.load(dataset / "train-000-legal_action_offsets.npy")

    assert summary["counts"]["train_samples"] == 1
    assert np.load(dataset / "train-000-positions.npy").shape == (1, 15, 10, 9)
    assert np.load(dataset / "train-000-teacher_score_kinds.npy").tolist() == [0]
    assert np.load(dataset / "train-000-teacher_scores.npy").tolist() == [50.0]
    assert candidate_ids.tolist() == [[action_id("C3-C4"), action_id("H2-H3"), -1, -1, -1]]
    assert legal_offsets.tolist() == [0, len(legal_ids)]
    assert set(candidate_ids[0, :2]).issubset(set(legal_ids.tolist()))
    games = (dataset / "train-000-games.jsonl").read_text(encoding="utf-8").splitlines()
    assert games == ['{"game_id": "game-3", "sample_start": 0, "sample_end": 1}']
    assert not (dataset / "train-000-game_indices.npy").exists()
    assert not (dataset / "train-000-plys.npy").exists()
    assert not (dataset / "train-000-metadata.jsonl").exists()
    assert not list(dataset.glob("*-human_action_ids.npy"))
    assert not list(dataset.glob("*-human_move_valid.npy"))


def test_export_records_rotates_black_actions_into_current_view(tmp_path: Path) -> None:
    position = apply_move(encode_fen(STARTING_FEN), *iccs_to_indices("C3-C4"))
    record = annotation_record(
        game_id="game-9", split="train", ply=1, fen=position_to_fen(position),
        candidates=[{"rank": 1, "move": "C6-C5", "score_kind": "cp", "score": 12}],
    )
    record["move"] = "C6-C5"
    record["teacher"]["bestmove"] = "C6-C5"  # type: ignore[index]
    export_records([record], tmp_path, max_games=None, max_candidates=5)

    candidate_ids = np.load(tmp_path / "dataset" / "train-000-candidate_action_ids.npy")
    expected_start, expected_end = iccs_to_indices("G3-G4")
    assert candidate_ids[0, 0] == 90 * expected_start + expected_end
    assert np.load(tmp_path / "dataset" / "train-000-positions.npy")[0, 14].sum() == 0.0


def test_export_records_preserves_mate_in_raw_policy_inputs(tmp_path: Path) -> None:
    record = annotation_record(
        game_id="game-5", split="train", ply=0, fen=STARTING_FEN,
        candidates=[
            {"rank": 1, "move": "C3-C4", "score_kind": "cp", "score": 50},
            {"rank": 2, "move": "H2-H3", "score_kind": "mate", "score": 3},
        ],
    )
    summary = export_records([record], tmp_path, max_games=None, max_candidates=5)

    dataset = tmp_path / "dataset"
    assert np.load(dataset / "train-000-candidate_score_kinds.npy").tolist() == [[0, 1, 0, 0, 0]]
    assert np.load(dataset / "train-000-candidate_scores.npy").tolist() == [[50.0, 3.0, 0.0, 0.0, 0.0]]
    assert summary["counts"]["train_samples"] == 1


def test_export_records_preserves_root_mate_for_a_bounded_value_target(tmp_path: Path) -> None:
    record = annotation_record(game_id="game-11", split="train", ply=0, fen=STARTING_FEN)
    record["teacher"]["score_kind"] = "mate"  # type: ignore[index]
    record["teacher"]["score"] = -2  # type: ignore[index]

    export_records([record], tmp_path, max_games=None, max_candidates=5)

    dataset = tmp_path / "dataset"
    assert np.load(dataset / "train-000-teacher_score_kinds.npy").tolist() == [1]
    assert np.load(dataset / "train-000-teacher_scores.npy").tolist() == [-2.0]
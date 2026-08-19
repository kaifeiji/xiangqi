from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from annotate_pikafish import (
    build_teacher,
    completed_game_ids,
    ensure_annotation_config,
    input_fingerprint,
    input_paths,
)
from data_utils import iccs_to_uci


STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def test_build_teacher_preserves_raw_current_player_scores_and_iccs_pvs() -> None:
    annotation = {
        "bestmove": "c3c4",
        "variations": [
            {
                "multipv": 1,
                "score_kind": "cp",
                "score": 83,
                "depth": 10,
                "nodes": 321,
                "pv": ["c3c4", "c6c5"],
            },
            {
                "multipv": 2,
                "score_kind": "mate",
                "score": -4,
                "depth": 10,
                "nodes": 233,
                "pv": ["h2e2"],
            },
        ],
    }

    teacher = build_teacher(STARTING_FEN, annotation, requested_multipv=5, include_pv=True)

    assert teacher == {
        "score_kind": "cp",
        "score": 83,
        "bestmove": "C3-C4",
        "requested_multipv": 5,
        "returned_multipv": 2,
        "candidates": [
            {
                "rank": 1,
                "move": "C3-C4",
                "score_kind": "cp",
                "score": 83,
                "depth": 10,
                "nodes": 321,
                "pv": ["C3-C4", "C6-C5"],
            },
            {
                "rank": 2,
                "move": "H2-E2",
                "score_kind": "mate",
                "score": -4,
                "depth": 10,
                "nodes": 233,
                "pv": ["H2-E2"],
            },
        ],
    }


def test_build_teacher_omits_full_pvs_by_default() -> None:
    annotation = {
        "bestmove": "c3c4",
        "variations": [
            {"multipv": 1, "score_kind": "cp", "score": 83, "depth": 10, "nodes": 321, "pv": ["c3c4", "c6c5"]},
        ],
    }

    teacher = build_teacher(STARTING_FEN, annotation, requested_multipv=5)

    assert teacher["candidates"] == [
        {"rank": 1, "move": "C3-C4", "score_kind": "cp", "score": 83, "depth": 10, "nodes": 321}
    ]


def test_iccs_to_uci_normalizes_case_for_the_engine_protocol() -> None:
    assert iccs_to_uci("C3-C4") == "c3c4"


def test_completed_game_ids_requires_the_source_game_ply_count(tmp_path: Path) -> None:
    records = tmp_path / "annotations.partial.jsonl"
    records.write_text(
        "\n".join(
            json.dumps({"schema_version": 1, "game_id": "complete", "ply": ply})
            for ply in range(2)
        )
        + "\n"
        + json.dumps({"schema_version": 1, "game_id": "truncated", "ply": 0})
        + "\n",
        encoding="utf-8",
    )

    assert completed_game_ids(records, {"complete": 2, "truncated": 2}) == {"complete"}


def test_annotation_config_rejects_an_incompatible_resume(tmp_path: Path) -> None:
    config = {"schema_version": 1, "input_fingerprint": "abc", "depth": 10, "multipv": 5}

    ensure_annotation_config(tmp_path, config)
    ensure_annotation_config(tmp_path, config)

    with pytest.raises(ValueError, match="configuration does not match"):
        ensure_annotation_config(tmp_path, {**config, "depth": 12})


def test_input_paths_excludes_duplicate_game_audit_files(tmp_path: Path) -> None:
    for name in ("train-000.jsonl", "duplicates.jsonl", "train.duplicates.jsonl"):
        (tmp_path / name).touch()

    assert input_paths(tmp_path) == [tmp_path / "train-000.jsonl"]


def test_annotation_config_migrates_an_excluded_duplicate_audit_file(tmp_path: Path) -> None:
    game_shard = {"path": "train-000.jsonl", "bytes": 1, "sha256": "game"}
    duplicate_audit = {"path": "duplicates.jsonl", "bytes": 1, "sha256": "audit"}
    config = {"schema_version": 1, "input_shards": [game_shard], "input_fingerprint": input_fingerprint([game_shard])}
    old_config = {
        "schema_version": 1,
        "input_shards": [duplicate_audit, game_shard],
        "input_fingerprint": input_fingerprint([duplicate_audit, game_shard]),
    }
    (tmp_path / "annotation_config.json").write_text(json.dumps(old_config), encoding="utf-8")

    ensure_annotation_config(tmp_path, config)

    assert json.loads((tmp_path / "annotation_config.json").read_text(encoding="utf-8")) == {
        **config,
        "hash_mb": None,
        "progress_every_positions": 16,
    }


def test_annotation_config_migrates_default_runtime_fields(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "input_shards": [],
        "input_fingerprint": input_fingerprint([]),
        "depth": None,
        "movetime_ms": None,
        "nodes": 50000,
        "multipv": 1,
        "pikafish_threads": 2,
        "include_pv": False,
        "hash_mb": None,
        "progress_every_positions": 16,
    }
    old_config = {key: value for key, value in config.items() if key not in {"hash_mb", "progress_every_positions"}}
    (tmp_path / "annotation_config.json").write_text(json.dumps(old_config), encoding="utf-8")

    ensure_annotation_config(tmp_path, config)

    assert json.loads((tmp_path / "annotation_config.json").read_text(encoding="utf-8")) == {
        **config,
        "hash_mb": None,
        "progress_every_positions": 16,
    }
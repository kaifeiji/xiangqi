from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from annotate_pikafish import (
    build_teacher,
    compress_shard,
    completed_game_ids,
    input_paths,
    log_shard_completed,
    log_shard_started,
    read_shard_log,
    recover_latest_shard,
)

from data_utils import iccs_to_uci


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

    teacher = build_teacher(annotation, requested_multipv=5)

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


def test_build_teacher_omits_full_pvs_for_single_pv() -> None:
    annotation = {
        "bestmove": "c3c4",
        "variations": [
            {"multipv": 1, "score_kind": "cp", "score": 83, "depth": 10, "nodes": 321, "pv": ["c3c4", "c6c5"]},
        ],
    }

    teacher = build_teacher(annotation, requested_multipv=1)

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


def test_input_paths_excludes_duplicate_game_audit_files(tmp_path: Path) -> None:
    for name in ("train-000.jsonl", "duplicates.jsonl", "train.duplicates.jsonl"):
        (tmp_path / name).touch()

    assert input_paths(tmp_path) == [tmp_path / "train-000.jsonl"]


def test_recover_latest_shard_discards_an_incomplete_game(tmp_path: Path) -> None:
    records = tmp_path / "train-000-000.jsonl"
    records.write_text(
        "\n".join(
            [
                json.dumps({"game_id": "complete", "ply": 0}),
                json.dumps({"game_id": "complete", "ply": 1}),
                json.dumps({"game_id": "interrupted", "ply": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed, positions = recover_latest_shard(records, {"complete": 2, "interrupted": 2})

    assert completed == {"complete"}
    assert positions == 2
    assert [json.loads(line)["game_id"] for line in records.read_text(encoding="utf-8").splitlines()] == [
        "complete",
        "complete",
    ]


def test_compress_shard_replaces_raw_file_with_zstandard_file(tmp_path: Path) -> None:
    records = tmp_path / "train-000-000.jsonl"
    records.write_text('{"game_id":"complete","ply":0}\n', encoding="utf-8")

    compress_shard(records)

    compressed = tmp_path / "train-000-000.jsonl.zst"
    assert not records.exists()
    assert compressed.is_file()
    assert completed_game_ids(compressed, {"complete": 1}) == {"complete"}


def test_shard_log_has_one_record_updated_on_completion(tmp_path: Path) -> None:
    source = tmp_path / "train-000.jsonl"
    shard = tmp_path / "train-000-000.jsonl"

    log_shard_started(tmp_path, source, shard)
    log_shard_started(tmp_path, source, shard)
    log_shard_completed(tmp_path, source, shard, 8195)

    records = read_shard_log(tmp_path / "shard_times.jsonl")
    assert len(records) == 1
    assert records[0]["source"] == str(source)
    assert records[0]["shard"] == shard.name
    assert isinstance(records[0]["started_at"], str)
    assert isinstance(records[0]["completed_at"], str)
    assert records[0]["positions"] == 8195
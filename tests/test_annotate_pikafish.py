from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import annotate_pikafish
from annotate_pikafish import (
    annotate_source,
    build_teacher,
    compress_shard,
    completed_game_ids,
    ensure_shard_started,
    input_paths,
    log_shard_completed,
    prioritize_sources_for_resume,
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


def test_build_teacher_skips_incomplete_non_primary_variations() -> None:
    annotation = {
        "bestmove": "c3c4",
        "variations": [
            {"multipv": 1, "score_kind": "cp", "score": 83, "depth": 10, "nodes": 321, "pv": ["c3c4", "c6c5"]},
            {"multipv": 2, "score_kind": "cp", "score": 27, "depth": 10, "nodes": 210},
            {"multipv": 3, "score_kind": "cp", "score": 14, "depth": 10, "nodes": 198, "pv": ["h2e2"]},
        ],
    }

    teacher = build_teacher(annotation, requested_multipv=5)

    assert teacher["returned_multipv"] == 2
    assert teacher["candidates"] == [
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
            "rank": 3,
            "move": "H2-E2",
            "score_kind": "cp",
            "score": 14,
            "depth": 10,
            "nodes": 198,
            "pv": ["H2-E2"],
        },
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

    ensure_shard_started(tmp_path, source, shard)
    ensure_shard_started(tmp_path, source, shard)
    log_shard_completed(tmp_path, source, shard, 8195)

    records = read_shard_log(tmp_path / "shard_times.jsonl")
    assert len(records) == 1
    assert records[0]["source"] == str(source)
    assert records[0]["shard"] == shard.name
    assert isinstance(records[0]["started_at"], str)
    assert isinstance(records[0]["completed_at"], str)
    assert records[0]["positions"] == 8195


def test_resume_skip_only_does_not_create_empty_new_shard(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "train-000.jsonl"
    source.write_text("\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    records_path = output_dir / "train-000-001.jsonl"

    class DummyAnnotator:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr(annotate_pikafish, "find_pikafish", lambda: Path("dummy.exe"))
    monkeypatch.setattr(annotate_pikafish, "PikafishAnnotator", lambda *args, **kwargs: DummyAnnotator())
    monkeypatch.setattr(
        annotate_pikafish,
        "prepare_output",
        lambda *args, **kwargs: ({"done-game"}, 1, 0, records_path),
    )
    monkeypatch.setattr(
        annotate_pikafish,
        "iter_unified_games",
        lambda _path: iter([{"game_id": "done-game", "fen": "ignored", "moves": []}]),
    )

    counts = annotate_source(
        source,
        output_dir,
        depth=1,
        movetime_ms=None,
        nodes=None,
        pikafish_threads=1,
        multipv=1,
        hash_mb=None,
        shard_size=8192,
        resume=True,
    )

    assert counts["skipped_games"] == 1
    assert not records_path.exists()


def test_prioritize_sources_for_resume_puts_incomplete_raw_shards_first(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    source_a = input_dir / "train-000.jsonl"
    source_b = input_dir / "train-001.jsonl"
    source_a.write_text("\n", encoding="utf-8")
    source_b.write_text("\n", encoding="utf-8")

    # Source B has an unfinished raw shard; source A only has compressed output.
    (output_dir / "train-000-000.jsonl.zst").write_text("", encoding="utf-8")
    (output_dir / "train-001-005.jsonl").write_text("", encoding="utf-8")

    ordered = prioritize_sources_for_resume([source_a, source_b], output_dir, resume=True)

    assert ordered == [source_b, source_a]


def test_single_file_resume_reuses_existing_source_named_shards(tmp_path: Path) -> None:
    source = tmp_path / "test-001.jsonl"
    source.write_text("\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_shard = output_dir / "test-001-005.jsonl.zst"
    existing_shard.write_text("", encoding="utf-8")

    assert annotate_pikafish.output_shards_for(source, output_dir) == [(5, existing_shard)]


def test_prioritize_sources_for_resume_keeps_input_order_when_not_resuming(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    source_a = input_dir / "train-000.jsonl"
    source_b = input_dir / "train-001.jsonl"
    source_a.write_text("\n", encoding="utf-8")
    source_b.write_text("\n", encoding="utf-8")

    ordered = prioritize_sources_for_resume([source_a, source_b], output_dir, resume=False)

    assert ordered == [source_a, source_b]


def test_annotate_source_compresses_tail_shard_even_below_shard_size(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "test-001.jsonl"
    source.write_text("\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    class DummyAnnotator:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def annotate(self, _fen: str):
            return {"bestmove": "a0a1", "variations": [{"multipv": 1, "score_kind": "cp", "score": 1, "pv": ["a0a1"]}]}

    monkeypatch.setattr(annotate_pikafish, "find_pikafish", lambda: Path("dummy.exe"))
    monkeypatch.setattr(annotate_pikafish, "PikafishAnnotator", lambda *args, **kwargs: DummyAnnotator())
    monkeypatch.setattr(
        annotate_pikafish,
        "iter_unified_games",
        lambda _path: iter([{"game_id": "g1", "fen": "fen", "moves": ["A0-A1"]}]),
    )
    monkeypatch.setattr(annotate_pikafish, "iccs_to_indices", lambda _move: (0, 1))
    monkeypatch.setattr(annotate_pikafish, "encode_fen", lambda _fen: object())
    monkeypatch.setattr(annotate_pikafish, "position_to_fen", lambda _position: "fen")
    monkeypatch.setattr(annotate_pikafish, "apply_move", lambda position, _start, _end: position)
    monkeypatch.setattr(annotate_pikafish, "indices_to_iccs", lambda _start, _end: "A0-A1")
    monkeypatch.setattr(annotate_pikafish, "split_for", lambda _game_id: "test")
    monkeypatch.setattr(
        annotate_pikafish,
        "build_teacher",
        lambda _annotation, requested_multipv: {
            "score_kind": "cp",
            "score": 1,
            "bestmove": "A0-A1",
            "requested_multipv": requested_multipv,
            "returned_multipv": 1,
            "candidates": [
                {
                    "rank": 1,
                    "move": "A0-A1",
                    "score_kind": "cp",
                    "score": 1,
                    "depth": 1,
                    "nodes": 1,
                }
            ],
        },
    )

    counts = annotate_source(
        source,
        output_dir,
        depth=1,
        movetime_ms=None,
        nodes=None,
        pikafish_threads=1,
        multipv=1,
        hash_mb=None,
        shard_size=8192,
        resume=False,
    )

    assert counts["valid_games"] == 1
    assert not (output_dir / "test-001-000.jsonl").exists()
    assert (output_dir / "test-001-000.jsonl.zst").is_file()


def test_resume_skip_only_compresses_recovered_non_empty_raw_shard(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "test-001.jsonl"
    source.write_text("\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    records_path = output_dir / "test-001-005.jsonl"
    records_path.write_text('{"game_id":"done-game","ply":0}\n', encoding="utf-8")

    class DummyAnnotator:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr(annotate_pikafish, "find_pikafish", lambda: Path("dummy.exe"))
    monkeypatch.setattr(annotate_pikafish, "PikafishAnnotator", lambda *args, **kwargs: DummyAnnotator())
    monkeypatch.setattr(
        annotate_pikafish,
        "prepare_output",
        lambda *args, **kwargs: ({"done-game"}, 5, 1, records_path),
    )
    monkeypatch.setattr(
        annotate_pikafish,
        "iter_unified_games",
        lambda _path: iter([{"game_id": "done-game", "fen": "ignored", "moves": []}]),
    )

    counts = annotate_source(
        source,
        output_dir,
        depth=1,
        movetime_ms=None,
        nodes=None,
        pikafish_threads=1,
        multipv=1,
        hash_mb=None,
        shard_size=8192,
        resume=True,
    )

    assert counts["skipped_games"] == 1
    assert not records_path.exists()
    assert (output_dir / "test-001-005.jsonl.zst").is_file()
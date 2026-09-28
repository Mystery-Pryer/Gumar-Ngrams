#!/usr/bin/env python3
"""Deterministic bounded ArSyra QA transform dry run.

Processes exactly the first ten records from the verified in-repository
all.jsonl primary recordset. That subset covers all eight observed source
categories, both source-generation modes, and the mapped nullable metadata
patterns for difficulty and dialect_group.

Writes a temporary Parquet and a no-source-text validation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SNAPSHOT_ID = "SNP-000023"
RESOURCE_ID = "RES-000021"
ARTIFACT_ID = "ART-000533"
SOURCE_OBJECT = "all.jsonl"
SOURCE_REPO_PATH = Path(
    "source/legacy_relocated/gulf/uae/mirrored/arsyra_gulf_kaggle/data/data/all.jsonl"
)
EXPECTED_SHA256 = "589a2cc92c1e4f677f2437191177fc8954ea4f50243b6c0d712288996d020eda"
EXPECTED_BYTES = 24886
EXPECTED_ROWS = 50
SELECTED_ROWS = 10
EXPECTED_SELECTED_CATEGORY_COUNTS = {
    "conversation_pairs": 1,
    "dialect": 2,
    "formality_transfer": 2,
    "medical_dialect": 1,
    "named_entities_local": 1,
    "slang": 1,
    "tech_dialect": 1,
    "vocabulary": 1,
}
EXPECTED_SELECTED_SOURCE_COUNTS = {"ai_generated": 1, "static": 9}
SOURCE_ATTRIBUTE_FIELDS = (
    "response_time_ms",
    "quality_score",
    "answered_at",
    "question_source",
    "difficulty",
    "quality_grade",
    "speaker_hash",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def foundation_id(question_code: str) -> str:
    preimage = SNAPSHOT_ID + ARTIFACT_ID + question_code
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def source_attr_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    raise RuntimeError(f"unexpected source_attributes value type: {type(value).__name__}")


def read_source(source: Path) -> tuple[list[dict], dict]:
    source_bytes = source.stat().st_size
    if source_bytes != EXPECTED_BYTES:
        raise RuntimeError(f"source byte-size mismatch: {source_bytes} != {EXPECTED_BYTES}")
    source_sha = sha256_file(source)
    if source_sha != EXPECTED_SHA256:
        raise RuntimeError(f"source sha256 mismatch: {source_sha}")

    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != EXPECTED_ROWS:
        raise RuntimeError(f"source row count mismatch: {len(lines)} != {EXPECTED_ROWS}")

    all_codes: list[str] = []
    parsed_all: list[dict] = []
    for physical_line_number, raw_line in enumerate(lines, start=1):
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{SOURCE_OBJECT}:{physical_line_number}: malformed JSON"
            ) from exc
        required = {
            "question_code",
            "category",
            "subcategory",
            "question_text",
            "answer_text",
            "response_time_ms",
            "quality_score",
            "country",
            "answered_at",
            "question_source",
            "quality_grade",
            "speaker_hash",
        }
        missing = sorted(required - obj.keys())
        if missing:
            raise RuntimeError(
                f"{SOURCE_OBJECT}:{physical_line_number}: missing required fields {missing}"
            )
        all_codes.append(str(obj["question_code"]))
        parsed_all.append(obj)

    if len(set(all_codes)) != EXPECTED_ROWS:
        raise RuntimeError("question_code is not unique across the primary recordset")

    rows: list[dict] = []
    for physical_line_number, obj in enumerate(parsed_all[:SELECTED_ROWS], start=1):
        question_code = str(obj["question_code"])
        attrs = {
            key: source_attr_value(obj.get(key))
            for key in SOURCE_ATTRIBUTE_FIELDS
        }
        rows.append(
            {
                "foundation_record_id": foundation_id(question_code),
                "snapshot_id": SNAPSHOT_ID,
                "resource_id": RESOURCE_ID,
                "artifact_id": ARTIFACT_ID,
                "source_object": SOURCE_OBJECT,
                "source_record_locator": (
                    "git:source/legacy_relocated/gulf/uae/mirrored/"
                    "arsyra_gulf_kaggle/data/data/all.jsonl"
                    f"#line={physical_line_number}"
                ),
                "source_split": None,
                "source_record_id": question_code,
                "record_family": "qa_pair",
                "source_attributes": attrs,
                "source_question_id": question_code,
                "question_text": str(obj["question_text"]),
                "answer_text": str(obj["answer_text"]),
                "source_category": str(obj["category"]),
                "source_subcategory": str(obj["subcategory"]),
                "source_country": str(obj["country"]),
                "source_dialect_group": (
                    None if obj.get("dialect_group") is None
                    else str(obj["dialect_group"])
                ),
            }
        )

    return rows, {
        "source_sha256": source_sha,
        "source_bytes": source_bytes,
        "primary_recordset_rows": len(parsed_all),
        "unique_question_codes": len(set(all_codes)),
    }


def write_parquet(rows: list[dict], output: Path) -> None:
    schema = pa.schema(
        [
            pa.field("foundation_record_id", pa.string(), nullable=False),
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("resource_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("source_object", pa.string(), nullable=False),
            pa.field("source_record_locator", pa.string(), nullable=False),
            pa.field("source_split", pa.string(), nullable=True),
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("record_family", pa.string(), nullable=False),
            pa.field("source_attributes", pa.map_(pa.string(), pa.string()), nullable=False),
            pa.field("source_question_id", pa.string(), nullable=False),
            pa.field("question_text", pa.string(), nullable=False),
            pa.field("answer_text", pa.string(), nullable=False),
            pa.field("source_category", pa.string(), nullable=False),
            pa.field("source_subcategory", pa.string(), nullable=False),
            pa.field("source_country", pa.string(), nullable=False),
            pa.field("source_dialect_group", pa.string(), nullable=True),
        ]
    )
    normalized = []
    for row in rows:
        item = dict(row)
        item["source_attributes"] = list(item["source_attributes"].items())
        normalized.append(item)
    table = pa.Table.from_pylist(normalized, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd", use_dictionary=True)


def validate(rows: list[dict], output: Path, source_meta: dict) -> dict:
    table = pq.read_table(output)
    out = table.to_pylist()
    if table.num_rows != SELECTED_ROWS:
        raise RuntimeError(f"expected {SELECTED_ROWS} rows, found {table.num_rows}")
    if len({row["foundation_record_id"] for row in out}) != SELECTED_ROWS:
        raise RuntimeError("foundation IDs are not unique")
    if len({row["source_question_id"] for row in out}) != SELECTED_ROWS:
        raise RuntimeError("selected source question IDs are not unique")

    category_counts = dict(sorted(Counter(row["source_category"] for row in out).items()))
    if category_counts != EXPECTED_SELECTED_CATEGORY_COUNTS:
        raise RuntimeError(f"source category accounting mismatch: {category_counts}")

    source_counts = dict(
        sorted(
            Counter(dict(row["source_attributes"])["question_source"] for row in out).items()
        )
    )
    if source_counts != EXPECTED_SELECTED_SOURCE_COUNTS:
        raise RuntimeError(f"question_source accounting mismatch: {source_counts}")

    difficulty_nulls = 0
    dialect_group_nulls = 0
    for original, emitted in zip(rows, out, strict=True):
        for field in (
            "foundation_record_id",
            "source_record_locator",
            "source_record_id",
            "source_question_id",
            "question_text",
            "answer_text",
            "source_category",
            "source_subcategory",
            "source_country",
            "source_dialect_group",
        ):
            if original[field] != emitted[field]:
                raise RuntimeError(f"{field} changed after Parquet round trip")

        original_attrs = original["source_attributes"]
        emitted_attrs = dict(emitted["source_attributes"])
        if original_attrs != emitted_attrs:
            raise RuntimeError("source_attributes changed after Parquet round trip")
        if emitted_attrs["difficulty"] is None:
            difficulty_nulls += 1
        if emitted["source_dialect_group"] is None:
            dialect_group_nulls += 1

    if difficulty_nulls != 1:
        raise RuntimeError(f"expected one selected null difficulty, found {difficulty_nulls}")
    if dialect_group_nulls != 9:
        raise RuntimeError(
            f"expected nine selected null dialect_group values, found {dialect_group_nulls}"
        )

    qa_collection = "\n".join(
        f"{row['question_text']}\t{row['answer_text']}" for row in out
    )
    source_label_collection = "\n".join(
        f"{row['source_category']}\t{row['source_subcategory']}\t"
        f"{row['source_country']}\t{row['source_dialect_group'] or ''}"
        for row in out
    )
    locator_collection = "\n".join(row["source_record_locator"] for row in out)
    schema_text = str(table.schema)

    return {
        "checkpoint": "8C-1G",
        "source_snapshot": SNAPSHOT_ID,
        "record_family": "qa_pair",
        "subset_rule": (
            "first 10 physical all.jsonl rows; covers all 8 source categories, "
            "both question_source modes, nullable difficulty, and nullable dialect_group"
        ),
        "canonical_row_count": SELECTED_ROWS,
        "primary_recordset_rows_verified": source_meta["primary_recordset_rows"],
        "unique_primary_question_codes_verified": source_meta["unique_question_codes"],
        "selected_category_row_counts": category_counts,
        "selected_question_source_row_counts": source_counts,
        "selected_null_difficulty_count": difficulty_nulls,
        "selected_null_source_dialect_group_count": dialect_group_nulls,
        "unique_foundation_record_ids": SELECTED_ROWS,
        "identity_preimage_rule": "sha256(snapshot_id + artifact_id + question_code)",
        "source_sha256": source_meta["source_sha256"],
        "source_bytes": source_meta["source_bytes"],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "output_parquet_sha256": sha256_file(output),
        "qa_text_collection_sha256": hashlib.sha256(
            qa_collection.encode("utf-8")
        ).hexdigest(),
        "source_label_collection_sha256": hashlib.sha256(
            source_label_collection.encode("utf-8")
        ).hexdigest(),
        "source_locator_collection_sha256": hashlib.sha256(
            locator_collection.encode("utf-8")
        ).hexdigest(),
        "compression": "zstd",
        "checks": {
            "schema_readable": True,
            "bounded_subset_selection": True,
            "stable_unique_record_ids": True,
            "primary_question_code_uniqueness": True,
            "source_provenance": True,
            "question_text_preservation": True,
            "answer_text_preservation": True,
            "source_label_preservation": True,
            "source_attribute_preservation": True,
            "source_null_behavior_preservation": True,
            "no_controlled_semantic_population": True,
            "source_sha256": True,
            "full_primary_row_accounting": True,
        },
        "raw_source_text_emitted_in_manifest": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=SOURCE_REPO_PATH)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    rows, source_meta = read_source(args.source)
    write_parquet(rows, args.output_parquet)
    manifest = validate(rows, args.output_parquet, source_meta)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

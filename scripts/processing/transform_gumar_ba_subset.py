#!/usr/bin/env python3
"""Deterministic bounded Gumar ngram_stat transform dry run.

Processes exactly the first three physical data rows from each 1-gram through
5-gram member in the already-verified BA.tar.xz archive. Writes a temporary
Parquet and a no-source-text validation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq

SNAPSHOT_ID = "SNP-000001"
RESOURCE_ID = "RES-000001"
ARTIFACT_ID = "ART-000514"
ARCHIVE_TAG = "BA"
EXPECTED_ARCHIVE_SHA256 = "d9b6478a451d0cabd95590ccee80bddb82033cde51de1060a94bf25654390cf3"
ROWS_PER_MEMBER = 3
MEMBERS = [f"BA/{n}-grams_BA.tsv" for n in range(1, 6)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def foundation_id(member: str, line_no: int, raw_line: str) -> str:
    preimage = SNAPSHOT_ID + ARTIFACT_ID + member + str(line_no) + raw_line
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def read_subset(archive: Path) -> tuple[list[dict], dict]:
    archive_sha = sha256_file(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"archive sha256 mismatch: {archive_sha}")

    rows: list[dict] = []
    member_meta = {}
    with tarfile.open(archive, mode="r:xz") as tf:
        for order, member in enumerate(MEMBERS, start=1):
            info = tf.getmember(member)
            handle = tf.extractfile(info)
            if handle is None:
                raise RuntimeError(f"cannot open {member}")

            emitted = 0
            physical_line = 0
            for raw in handle:
                physical_line += 1
                raw_line = raw.decode("utf-8", errors="strict").rstrip("\r\n")
                if not raw_line:
                    continue
                fields = raw_line.split("\t")
                if len(fields) != 3:
                    raise RuntimeError(f"{member}:{physical_line}: expected 3 fields")
                ngram, frequency_text, document_count_text = fields
                frequency = int(frequency_text)
                document_count = int(document_count_text)
                token_count = len(ngram.split(" "))
                if token_count != order:
                    raise RuntimeError(
                        f"{member}:{physical_line}: token count {token_count} != order {order}"
                    )
                rows.append(
                    {
                        "foundation_record_id": foundation_id(member, physical_line, raw_line),
                        "snapshot_id": SNAPSHOT_ID,
                        "resource_id": RESOURCE_ID,
                        "artifact_id": ARTIFACT_ID,
                        "source_object": member,
                        "source_record_locator": f"tar.xz://BA.tar.xz/{member}#line={physical_line}",
                        "source_split": ARCHIVE_TAG,
                        "source_record_id": f"{member}:{physical_line}",
                        "record_family": "ngram_stat",
                        "source_attributes": {
                            "archive_tag": ARCHIVE_TAG,
                            "physical_line_number": str(physical_line),
                        },
                        "source_dialect_tag": ARCHIVE_TAG,
                        "ngram_order": order,
                        "raw_ngram": ngram,
                        "frequency": frequency,
                        "document_count": document_count,
                    }
                )
                emitted += 1
                if emitted == ROWS_PER_MEMBER:
                    break
            if emitted != ROWS_PER_MEMBER:
                raise RuntimeError(f"{member}: expected {ROWS_PER_MEMBER} rows, found {emitted}")
            member_meta[member] = {
                "uncompressed_bytes": info.size,
                "subset_rows": emitted,
                "ngram_order": order,
            }

    if len(rows) != 15:
        raise RuntimeError(f"subset row count mismatch: {len(rows)}")
    return rows, {"archive_sha256": archive_sha, "members": member_meta}


def write_parquet(rows: list[dict], output: Path) -> None:
    schema = pa.schema(
        [
            pa.field("foundation_record_id", pa.string(), nullable=False),
            pa.field("snapshot_id", pa.string(), nullable=False),
            pa.field("resource_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("source_object", pa.string(), nullable=False),
            pa.field("source_record_locator", pa.string(), nullable=False),
            pa.field("source_split", pa.string(), nullable=False),
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("record_family", pa.string(), nullable=False),
            pa.field("source_attributes", pa.map_(pa.string(), pa.string()), nullable=False),
            pa.field("source_dialect_tag", pa.string(), nullable=False),
            pa.field("ngram_order", pa.int8(), nullable=False),
            pa.field("raw_ngram", pa.string(), nullable=False),
            pa.field("frequency", pa.int64(), nullable=False),
            pa.field("document_count", pa.int64(), nullable=False),
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
    if table.num_rows != 15:
        raise RuntimeError(f"expected 15 rows, found {table.num_rows}")
    if len({row["foundation_record_id"] for row in out}) != 15:
        raise RuntimeError("foundation IDs are not unique")

    order_counts = dict(sorted(Counter(row["ngram_order"] for row in out).items()))
    if order_counts != {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}:
        raise RuntimeError(f"order accounting mismatch: {order_counts}")

    for row in out:
        if row["frequency"] < 0 or row["document_count"] < 0:
            raise RuntimeError("negative source count")
        if len(row["raw_ngram"].split(" ")) != row["ngram_order"]:
            raise RuntimeError("round-trip ngram order mismatch")
        if row["source_split"] != ARCHIVE_TAG or row["source_dialect_tag"] != ARCHIVE_TAG:
            raise RuntimeError("archive tag preservation mismatch")

    original_ngram = "\n".join(row["raw_ngram"] for row in rows)
    output_ngram = "\n".join(row["raw_ngram"] for row in out)
    if original_ngram != output_ngram:
        raise RuntimeError("raw n-gram preservation mismatch")

    original_numeric = "\n".join(
        f"{row['frequency']}\t{row['document_count']}" for row in rows
    )
    output_numeric = "\n".join(
        f"{row['frequency']}\t{row['document_count']}" for row in out
    )
    if original_numeric != output_numeric:
        raise RuntimeError("numeric field preservation mismatch")

    schema_text = str(table.schema)
    return {
        "checkpoint": "8C-1D",
        "source_snapshot": SNAPSHOT_ID,
        "record_family": "ngram_stat",
        "subset_rule": "BA.tar.xz only; first three nonempty physical rows from each 1-gram through 5-gram member",
        "canonical_row_count": 15,
        "ngram_order_row_counts": order_counts,
        "source_archive": source_meta,
        "unique_foundation_record_ids": 15,
        "identity_preimage_rule": "sha256(snapshot_id + artifact_id + member_path + physical_line_number + raw_line)",
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "output_parquet_sha256": sha256_file(output),
        "raw_ngram_collection_sha256": hashlib.sha256(output_ngram.encode("utf-8")).hexdigest(),
        "numeric_collection_sha256": hashlib.sha256(output_numeric.encode("utf-8")).hexdigest(),
        "compression": "zstd",
        "checks": {
            "schema_readable": True,
            "bounded_subset_selection": True,
            "stable_unique_record_ids": True,
            "archive_member_provenance": True,
            "archive_tag_preservation": True,
            "ngram_order_preservation": True,
            "frequency_preservation": True,
            "document_count_preservation": True,
            "raw_ngram_preservation": True,
            "source_archive_sha256": True,
        },
        "raw_source_text_emitted_in_manifest": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    rows, source_meta = read_subset(args.archive)
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

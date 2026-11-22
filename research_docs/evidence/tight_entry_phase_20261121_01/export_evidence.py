"""Export existing receipts only; never launch training or simulation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
RUN = PROJECT / "outputs/visuotactile/run/tight_entry_phase_20261121_01"
AUDIT = PROJECT / "outputs/diagnostics/tight_entryfix_20261121_01"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local-artifact-hashes", action="store_true")
    args = parser.parse_args()
    copied = []
    hashes = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        checksum = digest(source)
        assert checksum == digest(target)
        copied.append({"source": str(source.relative_to(PROJECT)),
                       "published": str(target.relative_to(PROJECT)),
                       "bytes": source.stat().st_size, "sha256": checksum})

    for source in sorted(AUDIT.iterdir()):
        if source.is_file() and not source.name.lower().startswith(("cleanup", "archive", "checkpoint_archive")):
            copy(source, DEST / "audit" / source.name)
    old = PROJECT / "outputs/diagnostics/tight_adaptive_audit_20261121_01"
    old_dest = DEST.parent / "tight_adaptive_audit_20261121_01"
    for source in sorted(old.iterdir()):
        if source.is_file() and source.suffix in {".py", ".json", ".png", ".md", ".log"}:
            copy(source, old_dest / source.name)
    for root, folder in [(RUN, "run"), (RUN.with_name(RUN.name + "_recovery01"), "recovery")]:
        for source in sorted(root.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(root)
            if relative.parts[0].startswith("development_"):
                folder_relative = Path("development") / Path(*relative.parts[1:])
            else:
                folder_relative = Path(folder) / relative
            if source.suffix in {".json", ".jsonl", ".csv", ".png", ".log"} and source.stat().st_size <= 2 * 1024**2:
                copy(source, DEST / folder_relative)
    for role in "AB":
        copy(RUN / role / "state_data/data.json", DEST / "run" / role / "state_data/data.json")
        for relative in [Path(role) / "training/learning.jsonl", Path(role) / "backend" / role / "training/loss.jsonl"]:
            source = RUN / relative
            target = (DEST / "run" / relative).with_suffix(".receipts.jsonl.gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            count = 0
            # gzip timestamp is fixed, so exporting unchanged receipts is deterministic.
            with source.open("rb") as handle, target.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as output:
                    for line in handle:
                        sha.update(line)
                        row = json.loads(line)
                        keys = ("epoch", "batch_index", "updates", "update", "loss", "gradient_norm", "deck_sha256",
                                "effective_tick_contributions", "entry_tick_contributions", "supervised_ticks", "loss_weight_mass")
                        compact = {key: row[key] for key in keys if key in row}
                        if "visits" in row:
                            compact["visits"] = [{key: visit[key] for key in ("episode", "episode_visit", "anchor_sampling") if key in visit}
                                                 for visit in row["visits"]]
                        output.write((json.dumps(compact, separators=(",", ":")) + "\n").encode())
                        count += 1
            hashes[str(source)] = sha.hexdigest()
            copied.append({"source": str(source.relative_to(PROJECT)), "published": str(target.relative_to(PROJECT)),
                           "source_bytes": source.stat().st_size, "source_sha256": sha.hexdigest(),
                           "bytes": target.stat().st_size, "sha256": digest(target), "rows": count,
                           "transformation": "Selected existing committed-update receipt fields; geometry/performance payload omitted"})
    cleanup = PROJECT / "outputs/diagnostics/system_cleanup_20261121_01"
    for name in ("dead_code_audit.json", "dead_code_result.json", "production_source_identity.json"):
        copy(cleanup / name, DEST / "cleanup" / name)
    raw = subprocess.check_output(["git", "diff", "--raw", "--no-abbrev", "--diff-filter=D"], cwd=PROJECT, text=True)
    deleted = []
    for line in raw.splitlines():
        meta, path = line.split("\t", 1)
        deleted.append({"path": path, "git_blob_before_cleanup": meta.split()[2]})
    write_json(DEST / "cleanup/tracked_deletions.json", {"basis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
               "count": len(deleted), "files": deleted, "recovery": "git show <basis_commit>:<path>"})
    local = []
    for source in sorted(RUN.rglob("*")):
        if not source.is_file() or source.is_symlink() or source.suffix not in {".pt", ".h5", ".npz", ".mp4"}:
            continue
        row = {"path": str(source.relative_to(PROJECT)), "bytes": source.stat().st_size, "published_payload": False}
        if args.include_local_artifact_hashes:
            row["sha256"] = hashes.get(str(source)) or digest(source)
        local.append(row)
    write_json(DEST / "local_artifacts_manifest.json", {"scope": "Current entry-fix run; weights/caches/HDF5/NPZ/video payloads remain on the research computer per AGENTS.md", "files": local})
    write_json(DEST / "export_manifest.json", {"version": "tight_entry_publication_export_v1", "files": copied,
               "source_receipts_byte_identical_unless_transformation_is_explicit": True,
               "absolute_paths_in_original_receipts": "Historical research-machine provenance; not portable download URLs"})
    print(json.dumps({"copied_files": len(copied), "copied_bytes": sum(x["bytes"] for x in copied), "local_artifacts": len(local)}))


if __name__ == "__main__":
    main()

"""Check saved model paths and hashes without importing PyTorch or loading pickle."""

import argparse
import hashlib
import json
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checkpoints/manifest.json"


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", action="store_true",
                        help="also require the original local training checkpoints and archives")
    parser.add_argument("--verify", action="store_true",
                        help="read each selected file and verify its SHA-256 (large originals are slow)")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    entries = [entry for entry in manifest["files"]
               if args.originals or entry["role"] == "inference"]
    if not entries:
        parser.error("manifest contains no selected checkpoints")
    failures = 0
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            status = "MISSING"
        elif path.stat().st_size != entry["bytes"]:
            status = "SIZE MISMATCH"
        elif args.verify and entry.get("sha256") is None:
            status = "UNVERIFIED (no recorded hash; see manifest note)"
        elif args.verify and getattr(path.stat(), "st_flags", 0) & getattr(stat, "SF_DATALESS", 0):
            status = "UNAVAILABLE (download this cloud-backed file first)"
        elif args.verify and file_identity(path)["sha256"] != entry["sha256"]:
            status = "HASH MISMATCH"
        else:
            status = "OK"
        failures += status != "OK"
        print(f"{status}: {entry['path']}")
    if failures:
        print("See checkpoints/README.md for original-file locations and recovery instructions.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

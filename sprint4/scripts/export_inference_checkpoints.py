"""Export the two saved best models without modifying original training files.

Run from sprint4: python -m scripts.export_inference_checkpoints
Existing exports are refused. Use --output-dir with a new directory to compare
or regenerate copies without overwriting the curated exports.
"""

import argparse
import gc
import importlib
import json
from pathlib import Path

from .check_checkpoints import ROOT, file_identity

RUNS = (
    ("cartpole_muzero", "state4d_seed0_R9ncWC/run.pt", "cartpole_state4d.pt"),
    ("minatar_muzero", "breakout_screenshots/best.pt", "minatar_breakout.pt"),
)
TRAINING_ONLY = {
    "replay_state", "optimizer_state_dict", "history", "raw_episode_scores",
    "python_rng_state", "numpy_rng_state", "torch_rng_state",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/inference")
    args = parser.parse_args()
    # Fail before doing any work if a destination already exists.
    for name in [run[2] for run in RUNS] + ["provenance.json"]:
        if (args.output_dir / name).exists():
            parser.error(f"refusing to overwrite {args.output_dir / name}; use a new --output-dir")

    import torch
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    for package, relative, name in RUNS:
        source = ROOT / "checkpoints" / package / relative
        source_identity = {"path": source.relative_to(ROOT).as_posix(), **file_identity(source)}
        print(f"Exporting {source_identity['path']}", flush=True)
        loader = importlib.import_module(f"{package}.train").load_latent_checkpoint
        *networks, original = loader(source, torch.device("cpu"))
        payload = {key: value for key, value in original.items() if key not in TRAINING_ONLY}
        payload["inference_only"] = True
        payload["source_checkpoint"] = source_identity
        target = args.output_dir / name
        with target.open("xb") as stream:
            torch.save(payload, stream)
        *exported_networks, exported = loader(target, torch.device("cpu"))
        # Validate every tensor through the actual loaders before delivery.
        for before, after in zip(networks, exported_networks):
            before_state, after_state = before.state_dict(), after.state_dict()
            if before_state.keys() != after_state.keys() or any(
                not torch.equal(value, after_state[key]) for key, value in before_state.items()
            ):
                raise RuntimeError(f"network weights changed during export: {target}")
        if file_identity(source) != {key: source_identity[key] for key in ("bytes", "sha256")}:
            raise RuntimeError(f"source checkpoint changed: {source}")
        provenance.append({"file": name, **file_identity(target), "source": source_identity,
                           "all_network_tensors_equal": True})
        print(f"Verified identical network weights: {target.name} ({target.stat().st_size:,} bytes)", flush=True)
        del networks, original, payload, exported_networks, exported, before, after, before_state, after_state
        gc.collect()
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

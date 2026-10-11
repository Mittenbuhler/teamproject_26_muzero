# Trained-model inventory

All paths below are relative to `sprint4`. [Commands](../COMMANDS.md) use the
portable inference copies for easy evaluation. Original checkpoint files have
not been renamed, moved, converted or overwritten during this cleanup.

## Portable models included in the project

| File | Original trained weights | Use |
| --- | --- | --- |
| `inference/cartpole_state4d.pt` | `cartpole_muzero/state4d_seed0_R9ncWC/run.pt`, episode 800, CartPole v15 | Evaluate / record the four-state-input agent |
| `inference/minatar_breakout.pt` | `minatar_muzero/breakout_screenshots/best.pt`, episode 990 of a 1,500-episode run, screenshot format v2 | Evaluate / record Breakout-v1 |

The exports are approximately 240 KiB (CartPole) and 365 KiB (Breakout).
These small exports preserve representation, dynamics and prediction weights,
architecture, preprocessing and training configuration. Every network tensor
was compared with its original through the current loaders. Source SHA-256
hashes are in [inference/provenance.json](inference/provenance.json).
Exports omit replay, optimizer, RNG state and training history, so they cannot
resume training or supply replay-based diagnostics. Numeric training histories
remain in the curated results and full state remains in the original files.

## Original training checkpoints: preserved paths

| Directory | Files | Availability |
| --- | --- | --- |
| `cartpole_muzero/state4d_seed0_R9ncWC/` | `run.pt`, `run_latest.pt`, `run_final.pt`, `training.log`, `run_rewards.png` | Local training files; ignored by Git |
| `minatar_muzero/breakout_screenshots/` | `best.pt`, `latest.pt`, `final.pt` | Local training files; approximately 1 GB each, ignored by Git |
| `state_mcts/` | `state_dynamics.pt`, `state_policy.pt`, `state_value.pt`, MCTS-teacher variants and timestamped run | Separate component-test checkpoints |
| `cartpole_muzero/archive/pre_v8/` | `dynamics_cartpole.pt`, `policy_value_cartpole.pt` | OUTDATED format, retained for history |

Best files were selected by scheduled evaluation. Latest/final retain resumable
state, with different selection semantics. Use package checkpoint READMEs for
format details: [CartPole](cartpole_muzero/README.md),
[MinAtar](minatar_muzero/README.md), [testbed](state_mcts/README.md).

## Availability after cloning and recovery

The inference files and curated result reports are allowed into Git. Large
original training files remain ignored: a Git clone alone does not include
those replay buffers. For training continuation or exact report regeneration,
obtain the original run directories from the project's saved local copy or a
team backup and copy them into the exact paths above. No external download
location is configured in this repository. Do not substitute inference files
for full training state.

[manifest.json](manifest.json) records byte sizes and available SHA-256 hashes
for originals and inference copies. Two originals were cloud-backed placeholders
during cleanup: MinAtar `breakout_screenshots/final.pt` and testbed
`state_mcts/20261004_175041/state_value.pt`. Their paths, sizes and modification
times are retained, but hashes could not be verified. The main runs’ best/latest
checkpoints are locally readable. Run:

```bash
python -m scripts.check_checkpoints --verify
python -m scripts.check_checkpoints --originals --verify
```

The first checks the two portable files. The second also reads the large files
and intentionally fails if any listed original is absent, changed, or lacks a
verified hash. Download cloud-backed files before attempting to read them. It needs
only Python's standard library and does not deserialize model files.

Historical `legacy_muzero` and `muzero` paths inside original metadata record
previous locations. The active loaders use the explicit path supplied on the
command line; metadata does not need editing. Save new training in unique run
directories. Never run `git clean -fdx`: ignored original training files are
valuable project data.

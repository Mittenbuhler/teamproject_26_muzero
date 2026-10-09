# Outdated material: retained in place

Active results are [CartPole MuZero](../cartpole_muzero/README.md) and
[MinAtar MuZero](../minatar_muzero/README.md). `state_mcts` remains a supported
testing experiment. The entries below are historical or supporting material.
Existing data, filenames and model paths are retained. Only truly empty
directories may be removed; no nonempty directory is deleted by this cleanup.

| Location | Status and reason to keep it |
| --- | --- |
| [`S4_alphaZero_rebuild/`](../../S4_alphaZero_rebuild/README.md) | **OUTDATED implementation**: separately trained CartPole models; the earlier starting point. |
| [`example_code/`](../../example_code/README.md) | **HISTORICAL learning material**: MCTS notebook, not a current entry point. |
| [`tutorials/`](../../tutorials/README.md) | **HISTORICAL learning material**: Gymnasium, PyTorch, Q-learning notebooks and associated data. |
| [`slides/`](../../slides/README.md) | **HISTORICAL presentations**: snapshots that may predate the final implementation. |
| [`docs/history/`](history/README.md) | **OUTDATED prose** recovered from previous branch documentation; read with correction notes. |
| [`artifacts/cartpole_muzero/training_progress/`](../artifacts/cartpole_muzero/training_progress/README.md) | **OUTDATED plots** from earlier models, not evidence for v15. |
| [`checkpoints/cartpole_muzero/archive/pre_v8/`](../checkpoints/cartpole_muzero/archive/pre_v8/README.md) | **OUTDATED format**: separate state-model files, incompatible with current MuZero loaders. |
| [`cartpole_muzero/image_observation.py`](../cartpole_muzero/image_observation.py) | **OUTDATED helper** retained at its path; active CartPole input comes from `environment.py`. |
| [`artifacts/state_mcts/`](../artifacts/state_mcts/README.md) | **TESTBED history**: component diagnostics and older comparison plots, separate from main results. |

`legacy_muzero/` and `muzero/` are earlier Git layouts. The working tree already
replaced them with `cartpole_muzero/` and `minatar_muzero/` before this cleanup.
Earlier source remains in Git history; do not recreate duplicate active
packages from old instructions.

Read-only historical inspection from the repository root:

```bash
git log --all --oneline -- sprint4 S4_alphaZero_rebuild
git show 1f4529b:sprint4/README.md
```

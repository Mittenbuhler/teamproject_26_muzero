# Legacy MuZero checkpoint archive

The checkpoint root is reserved for fresh checkpoint-v14 runs. The collapsed v13 trial is currently preserved at the root for evaluation and must not be resumed as v14; it can be moved into a versioned archive after any old training process is confirmed stopped. All other older, mutually incompatible formats remain under `archive/`; no model files were deleted.

## Layout

```text
legacy_muzero/
├── README.md
├── cartpole_policy_v13.pt
├── cartpole_policy_v13_latest.pt
└── archive/
    ├── pre_v8/
    ├── v8/
    ├── v9/
    ├── v11/
    └── v12/
        ├── cartpole_spatial_v12.pt
        ├── cartpole_spatial_v12_latest.pt
        ├── cartpole_spatial_v12_final.pt
        └── cartpole_spatial_v12_rewards.png
```

## Compatibility

| Directory | Contents | Status |
| --- | --- | --- |
| `archive/pre_v8/` | Separate dynamics and policy/value checkpoints | Historical pre-unified models; incompatible with the current loader |
| `archive/v8/` | First unified flattened-latent checkpoint | Historical; incompatible with v14 |
| `archive/v9/` | Terminal-head and consistency-era checkpoints | Historical; incompatible with v14 |
| `archive/v11/` | Spatial latent with globally pooled prediction heads | Preserved experiment baselines; incompatible with position-preserving heads |
| `archive/v12/cartpole_spatial_v12*` | Position-preserving networks with temperature-sharpened replay policy targets | Evaluation-only baseline; must not be resumed as v14 |
| root `cartpole_policy_v13*` | Raw visit targets, but untrained warm-up search was valid policy supervision and warm-up had no updates | Preserved failed/evaluation-only run; must not be resumed as v14 |
| new root `cartpole_policy_v14*` | Raw visit targets plus policy-valid replay flags and active masked-policy warm-up learning | Active v14 training format |

The current training loader intentionally rejects incompatible files instead of partially loading weights or mixing replay-target semantics. A fresh v14 run is required: v13 replay does not contain trustworthy warm-up policy validity. Diagnostics may load preserved legacy weights for evaluation where supported; the documented v12 GIF commands remain valid.

## Version 11 experiments

| Experiment | Recorded extent | Files |
| --- | ---: | --- |
| `cartpole_first` | 100 episodes in saved history; predates complete resumable metadata | best, final, reward plot |
| `cartpole_v11` | 300 episodes | best at episode 260, latest/final at 300, reward plot |
| `cartpole_stable_v11` | 800 episodes | best at episode 560, latest/final at 800, reward plot |

Within an experiment, the base `.pt` is the best-evaluation checkpoint, `_latest.pt` is the resumable state, `_final.pt` is the end of the run, and `_rewards.png` is its raw/evaluation reward graph. These semantics apply where that file exists.

The preserved position-preserving v12 run reached episode 2800, with its best scheduled evaluation at episode 2720. Its base `.pt` is the best checkpoint; `_latest.pt` and `_final.pt` contain episode 2800. The v13 files at the root belong to the collapsed trial and are not valid resume inputs for v14. New runs should use a distinct v14 name at the checkpoint root, for example:

```text
checkpoints/legacy_muzero/cartpole_policy_v14.pt
```

Its best, latest, final, and reward-plot companions will remain together at the checkpoint root.

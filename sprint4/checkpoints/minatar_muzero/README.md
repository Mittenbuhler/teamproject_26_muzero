# Screenshot MinAtar MuZero checkpoints

[Model inventory and portable copies](../README.md) · [Central commands](../../COMMANDS.md)

The current loader accepts `screenshot_minatar_vanilla_muzero`, version 2.
Native feature-grid version-1 checkpoints and legacy CartPole checkpoints are
incompatible. Keep game, action-set, image-size, and history variants separate.

## Files present on 4 October 2026

| Path under this directory | Completed episodes | Role |
| --- | ---: | --- |
| `breakout_screenshots/best.pt` | 990 | Best periodic evaluation: 4.6667 on three fixed seeds |
| `breakout_screenshots/latest.pt` | 1500 | Resumable actor/learner state |
| `breakout_screenshots/final.pt` | 1500 | End of the last invocation; final periodic evaluation: 3.0 |

These files load with the current implementation. Each stores networks,
optimizer, replay, training/evaluation histories, configuration, and random
states. The best score is a selection statistic, not an independent estimate
of generalization. Use [the diagnostic command](../../minatar_muzero/README.md)
to evaluate a saved checkpoint on fresh seeds.

This run uses Breakout-v1 (3 minimal actions), four 32×32 RGB screenshots
(12 input channels), a 32×8×8 latent, 25 search simulations, discount 0.997,
and a five-step training unroll. Latest/final contain 19,966 replay transitions.
The large files are mostly replay observations stored as float32 histories.

## Default and explicit destinations

The trainer defaults to `checkpoints/minatar_muzero/<game>/best.pt` with companion
`latest.pt` and `final.pt`. The saved screenshot run used an explicit
`breakout_screenshots` directory. Resume that exact run from sprint4 with the
active project environment, retaining its recorded configuration:

```bash
python -m minatar_muzero.train --game breakout \
  --checkpoint-path checkpoints/minatar_muzero/breakout_screenshots/best.pt \
  --reward-plot-path artifacts/minatar_muzero/training/breakout_screenshots/rewards.png \
  --history-length 4 --image-size 32 --latent-channels 32 \
  --simulations 25 --batch-size 32 --updates-per-transition 0.25 \
  --min-updates-per-episode 5 --max-updates-per-episode 100 \
  --episodes 500 --reuse-checkpoint
```

`--episodes` counts additional episodes when resuming. Resume resolves latest,
then final, then the specified best path. Checkpoints are atomically replaced.
Use a new experiment directory for fresh training to preserve this run.

## Compatibility metadata

A current checkpoint records environment ID, action count, RGB render source,
image size, nearest-neighbor resize, uint8/255 normalization, history length,
input/latent shapes, head dimensions, policy-target/warm-up semantics, discount,
reward scale, and full training configuration. Structural or semantic
incompatibility must be resolved before weights/replay can be reused.
Cross-game transfer and conversion from the old native format are unsupported.

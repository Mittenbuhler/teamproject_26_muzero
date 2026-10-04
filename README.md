# Team Project: MuZero

This repository contains the development history and implementations of our
MuZero project.

## Repository structure

- [`archive/`](archive/) contains the work from previous sprints (sprint1-3),
  including the earlier AlphaZero/MuZero experiments and alternative
  implementations.
- [`sprint4/`](sprint4/) contains the current Sprint 4 work. It has three
  deliberately separate implementations:
  - [`state_mcts/`](sprint4/state_mcts/) - a real-state MCTS experiment for
    CartPole.
  - [`legacy_muzero/`](sprint4/legacy_muzero/) - the earlier
    screenshot/latent MuZero implementation.
  - [`muzero/`](sprint4/muzero/) - the native-grid MinAtar MuZero
    implementation.

The three Sprint 4 implementations have separate models, checkpoints, and
artifacts. Do not mix checkpoints between them.

## Running native MinAtar MuZero

### Train MuZero on Breakout

Run a longer Breakout training job from the `sprint4` directory:

```bash
../../.venv/bin/python -m muzero.train --game breakout \
  --episodes 500 --simulations 25 --batch-size 64 \
  --updates-per-transition 0.25 --min-updates-per-episode 5 \
  --max-updates-per-episode 100
```

To continue training from
`checkpoints/muzero/breakout/latest.pt`, repeat the command with
`--reuse-checkpoint`. In this mode, `--episodes` specifies how many
additional episodes to train.

### Evaluate a trained model

Evaluate or inspect a trained model and save a JSON report:

```bash
../../.venv/bin/python -m muzero.diagnose \
  --checkpoint checkpoints/muzero/breakout/best.pt \
  --game breakout \
  --output-json artifacts/muzero/diagnostics/breakout/best_seed0.json
```

### Create a GIF

Create a synchronized GIF showing gameplay and the native MinAtar feature
channels:

```bash
../../.venv/bin/python -m muzero.diagnose \
  --checkpoint checkpoints/muzero/breakout/best.pt \
  --game breakout \
  --record-gif artifacts/muzero/gifs/breakout/breakout_mcts.gif
```

Short game names such as `breakout`, `seaquest`, and `space_invaders` use the
MinAtar v1 environments by default. Use a full ID such as
`MinAtar/Breakout-v0` only when the v0 action variant is intended. For the
complete option reference and implementation details, see
[`sprint4/muzero/README.md`](sprint4/muzero/README.md).

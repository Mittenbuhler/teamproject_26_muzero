# Native MinAtar MuZero

This package is the native-grid successor to `legacy_muzero`. It keeps the same small, vanilla MuZero algorithm but consumes MinAtar's semantic channel planes directly. It does not render a screenshot, convert it to grayscale, or reuse a CartPole checkpoint.

## Native data flow

For a game returning an observation with height `H`, width `W`, and `C` native channels, the adapter converts the observation to normalized `float32` channel-first form. With the default `history_length=1`, the representation input is:

```text
MinAtar observation [H,W,C]
    -> adapter [C,H,W]
    -> h(observation history) [latent_channels,latent_H,latent_W]
    -> f(latent) -> policy logits over action_space.n, scalar value
```

If `history_length` is greater than one, time slots are zero-padded at the beginning of an episode and concatenated along the channel dimension, producing `[history_length * C,H,W]`. No observation-history RNN is used.

Only real native observation histories enter the representation network. Every hypothetical search step is produced from a latent state and discrete action:

```text
(latent, tiled one-hot action planes)
    -> g(latent,action) -> next latent, scalar reward
    -> f(next latent) -> policy logits, scalar value
```

The adapter obtains `C`, `H`, `W`, and the discrete action count from the selected environment. The model does not hard-code a 10x10 grid, a channel count, or an action count. Each game still receives its own networks, replay, artifacts, and checkpoints because channel meanings and action semantics differ.

`--game breakout` selects the MinAtar v1 environment with its minimal action set by default. Other standard game names include `asterix`, `freeway`, `seaquest`, and `space_invaders`. Action counts are read from each v1 environment rather than assumed from this list. Use a full ID such as `--game MinAtar/Breakout-v0` to select the six-action v0 variant used by the original MinAtar baselines; v0 and v1 checkpoints are not interchangeable.

## Replay, learning, and search

Replay remains episode-based and stores the native observation history, selected action, real reward, raw MCTS visit distribution, MCTS root value, termination/truncation state, and policy-valid state. Warm-up uses random behavior with masked policy supervision while value, reward, representation, and dynamics learning remain active. Recurrent unrolls use recorded actions and absorbing zero targets after termination; they never cross into the next episode.

Training uses scalar reward/value MSE and policy cross-entropy against raw normalized visits. MCTS expands one latent edge per simulation, uses no environment queries below the root, adds root noise only during self-play, and chooses the most-visited action during evaluation. The implementation intentionally omits EfficientZero additions, reconstruction/consistency losses, target encoders, reanalyze, prioritized replay, recurrent sequence models, and distributional supports.

## Commands

Run from `teamproject_26_muzero/sprint4` with the project virtual environment. Use `--help` as the source of truth for optional tuning flags while the native CLI evolves.

```bash
# install/register the pinned native MinAtar dependency
../../.venv/bin/python -m pip install -r requirements.txt

# inspect the native trainer
../../.venv/bin/python -m muzero.train --help

# short end-to-end smoke run
../../.venv/bin/python -m muzero.train --game breakout \
  --episodes 2 --max-steps 50 --simulations 3 \
  --latent-channels 8 --batch-size 8 --updates-per-episode 1

# a first substantive Breakout-v1 run
../../.venv/bin/python -m muzero.train --game breakout \
  --episodes 500 --simulations 25 --batch-size 64 \
  --updates-per-transition 0.25 --min-updates-per-episode 5 \
  --max-updates-per-episode 100

# continue the exact actor/learner state from latest.pt
../../.venv/bin/python -m muzero.train --game breakout \
  --episodes 500 --simulations 25 --batch-size 64 \
  --updates-per-transition 0.25 --min-updates-per-episode 5 \
  --max-updates-per-episode 100 --reuse-checkpoint

# select another game; its input channels/actions and output folder are derived
../../.venv/bin/python -m muzero.train --game seaquest --episodes 500

# inspect/evaluate the best Breakout checkpoint
../../.venv/bin/python -m muzero.diagnose \
  --checkpoint checkpoints/muzero/breakout/best.pt \
  --game breakout \
  --output-json artifacts/muzero/diagnostics/breakout/best_seed0.json

# synchronized environment/native-channel recording
../../.venv/bin/python -m muzero.diagnose \
  --checkpoint checkpoints/muzero/breakout/best.pt \
  --game breakout \
  --record-gif artifacts/muzero/gifs/breakout/breakout_mcts_seed42.gif

# test the native package
../../.venv/bin/python -m pytest -q muzero/tests
```

`--episodes` means additional episodes when `--reuse-checkpoint` is present. Resume reads `latest.pt` and restores networks, optimizer, replay, histories, episode count, and random-number states. Architecture, game/action variant, observation history, discount, and reward scale are validated before training continues.

## Output layout

```text
checkpoints/muzero/<game>/
├── best.pt
├── latest.pt
└── final.pt

artifacts/muzero/
├── training/<game>/rewards.png
├── diagnostics/<game>/
└── gifs/<game>/<game>_<raw-or-mcts>_seed<seed>.gif
```

See the [artifact manifest](../artifacts/muzero/README.md), [GIF notes](../artifacts/muzero/gifs/README.md), and [checkpoint manifest](../checkpoints/muzero/README.md) before moving or comparing runs.

## Diagnostic GIFs

The GIF keeps the model input and played environment synchronized. The environment rendering is shown next to the exact native channel planes used for that decision. With longer histories, the panel includes every history slot and channel, including initial zero padding. A final frame shows the terminal-updated input. These are real native observations, not decoded or imagined latent states.

Raw-policy mode acts from the prediction logits directly. MCTS mode still uses those logits as priors and combines them with learned rewards, dynamics, and leaf values; evaluation recordings disable root noise. Consult `muzero.diagnose --help` for the currently supported mode, seed, size, and frame-rate flags.

## Fresh-checkpoint requirement

Start every native MinAtar game from a fresh native checkpoint. Legacy screenshot checkpoints encode five grayscale CartPole frames and have incompatible input semantics even when tensor dimensions happen to match. A checkpoint from another MinAtar game, another history length, a different native channel layout, or a different v0/v1 action set must also be rejected rather than partially loaded.

## Current limitations

- The MinAtar package/environment registrations must be installed in the project virtual environment before native commands can run.
- Default support targets MinAtar v1 minimal action sets. Full-action v0 is optional and only available if the adapter exposes an explicit compatible game ID.
- Games train independently; there is no multitask model, shared replay, or cross-game checkpoint transfer.
- `history_length=1` is intentionally minimal. Increasing it may help games whose native channels do not expose enough motion information, but changes the representation input and requires a fresh checkpoint.
- Rendering palettes and channel names are game-specific; diagnostics must preserve the raw channel order reported by the environment.
- Structural tests, falling loss, or a good fixed-seed GIF do not establish that a game is solved. Report fixed-batch learnability, MCTS improvement, and evaluation return separately.

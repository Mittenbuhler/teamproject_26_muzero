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

## Training parameter reference

Invoke training as `../../.venv/bin/python -m muzero.train [OPTIONS]`. The defaults below are the values used when an option is omitted.

### Run, environment, and output

| Option | Default | Meaning |
| --- | --- | --- |
| `--game GAME`, `--env GAME` | `breakout` | MinAtar short name or full environment ID. Short names select v1, for example `breakout` becomes `MinAtar/Breakout-v1`; use a full v0 ID explicitly when needed. |
| `--episodes N` | `10` | Number of episodes to train. With `--reuse-checkpoint`, this is the number of **additional** episodes. Must be positive. |
| `--max-steps N` | `2500` | Maximum environment transitions per training or evaluation episode. Reaching the limit truncates the episode. Must be positive. |
| `--seed N` | `0` | Base seed for Python, NumPy, PyTorch, the action space, and per-episode environment resets. |
| `--sticky-action-prob P` | `0.1` | MinAtar probability of repeating the previous action instead of applying the selected action. Must be in `[0, 1]`. |
| `--no-difficulty-ramping` | off | Disable MinAtar's built-in difficulty ramping. Ramping is enabled unless this flag is present. |
| `--checkpoint-path PATH` | `checkpoints/muzero/<game>/best.pt` | Path for the best evaluation checkpoint. Companion `latest` and `final` checkpoint names are derived from this path. |
| `--reward-plot-path PATH` | `artifacts/muzero/training/<game>/rewards.png` | Destination for the training/evaluation reward plot written when training finishes. |
| `--reuse-checkpoint` | off | Resume the full actor/learner state. Resolution order is the companion `latest` checkpoint, then `final`, then the specified best-checkpoint path. |

### Model, replay, and learner

| Option | Default | Meaning |
| --- | --- | --- |
| `--history-length N` | `1` | Number of consecutive native observations concatenated along the channel axis. Changing it changes the architecture and requires a fresh checkpoint. Must be positive. |
| `--latent-channels N` | `32` | Channel width of the latent representation and the small dynamics/prediction networks. Must be positive. |
| `--batch-size N` | `32` | Replay transitions sampled per learner update, capped by the number currently available. Must be positive. |
| `--buffer-capacity N` | `20000` | Nominal transition capacity of the replay buffer. It evicts whole old episodes and always retains at least one complete episode, even if that episode exceeds the limit. Must be positive. |
| `--warmup-episodes N` | `1` | Initial episodes that use random actions and mask the policy loss. Value, reward, representation, and dynamics learning remain active. May be zero. |
| `--updates-per-episode N` | `5` | Fixed learner updates after each episode. Used only when `--updates-per-transition` is `0` or less. May be zero. |
| `--updates-per-transition RATE` | `0.0` | If positive, replace the fixed schedule with `ceil(episode_steps * RATE)` learner updates. |
| `--min-updates-per-episode N` | `1` | Lower clamp for transition-relative updates. Used only with a positive `--updates-per-transition`; may be zero. |
| `--max-updates-per-episode N` | `50` | Upper clamp for transition-relative updates. Must be positive and at least the minimum. |
| `--learning-rate RATE` | `0.0003` | Adam learning rate. Must be positive. A supplied value overrides the saved optimizer rate when resuming. |
| `--weight-decay RATE` | `0.00001` | Adam weight decay. Must be nonnegative and overrides the saved value when resuming. |
| `--gradient-clip-norm VALUE` | `5.0` | Maximum global gradient norm before the optimizer step. Must be positive. |
| `--dynamics-gradient-scale SCALE` | `0.5` | Multiplier applied to gradients passed backward through each recurrent dynamics state. Must be in `[0, 1]`. |

### Targets and losses

| Option | Default | Meaning |
| --- | --- | --- |
| `--discount GAMMA` | `0.997` | Discount used for value targets and MCTS backups. Must be in `[0, 1]` and cannot change while resuming replay. |
| `--bootstrap-steps N` | `10` | Number of rewards in each n-step value target before bootstrapping from the stored MCTS root value. Must be positive. |
| `--unroll-steps N` | `5` | Number of recorded actions through which the dynamics model is recurrently unrolled during one learner update. Must be positive. |
| `--reward-scale SCALE` | `1.0` | Multiplier applied to rewards and value targets used by the model. Must be positive and cannot change while resuming replay. Reported episode scores remain unscaled. |
| `--policy-loss-weight VALUE` | `1.0` | Weight of policy cross-entropy in the total loss. Must be nonnegative. |
| `--value-loss-weight VALUE` | `1.0` | Weight of scalar value MSE in the total loss. Must be nonnegative. |
| `--reward-loss-weight VALUE` | `1.0` | Weight of scalar reward MSE in the total loss. Must be nonnegative. |

At least one of the three loss weights must be greater than zero.

### MCTS and self-play exploration

| Option | Default | Meaning |
| --- | --- | --- |
| `--simulations N` | `25` | MCTS simulations per environment decision. More simulations cost more model inference. Must be positive. |
| `--pb-c-base VALUE` | `19652` | Base constant in the PUCT exploration coefficient. Must be positive. |
| `--pb-c-init VALUE` | `1.25` | Initial constant in the PUCT exploration coefficient. Must be positive. |
| `--root-dirichlet-alpha ALPHA` | `0.25` | Dirichlet concentration used to generate root prior noise during self-play. Must be positive. |
| `--root-exploration-fraction P` | `0.25` | Fraction of the root prior replaced by Dirichlet noise during self-play. Must be in `[0, 1]`. Evaluation never adds root noise. |
| `--temperature-initial T` | `1.0` | Visit-count action-sampling temperature before the first temperature boundary. Must be nonnegative. |
| `--temperature-middle T` | `0.5` | Temperature between the two episode boundaries. Must be nonnegative. |
| `--temperature-final T` | `0.25` | Temperature at and after the second episode boundary. Must be nonnegative. |
| `--temperature-initial-episodes N` | `200` | Use the initial temperature while the zero-based global episode index is below `N`. May be zero. |
| `--temperature-middle-episodes N` | `500` | Switch from middle to final temperature at this zero-based global episode index. Must be at least `--temperature-initial-episodes`. |

Temperature boundaries use the total completed episode count, including episodes restored by `--reuse-checkpoint`. A temperature of zero chooses the most-visited action instead of sampling.

### Evaluation

| Option | Default | Meaning |
| --- | --- | --- |
| `--evaluation-interval N` | `10` | Run deterministic, noise-free MCTS evaluation every `N` completed episodes. The final episode is always evaluated. Must be positive. |
| `--evaluation-episodes N` | `3` | Number of episodes in each evaluation; their mean score determines whether `best.pt` is replaced. Must be positive. |

`-h` or `--help` prints the current command-line interface.

## Diagnostic and GIF parameter reference

Invoke diagnostics as `../../.venv/bin/python -m muzero.diagnose CHECKPOINT [OPTIONS]`, or omit the positional checkpoint and supply `--checkpoint PATH` instead. Diagnostics run on CPU and compare raw-policy argmax play with noise-free MCTS play on identical seeds.

| Option | Default | Meaning |
| --- | --- | --- |
| `CHECKPOINT` | required unless `--checkpoint` is used | Positional path to a native MinAtar MuZero checkpoint. |
| `--checkpoint PATH` | none | Named alternative to the positional checkpoint path. If both are supplied, this option takes precedence. |
| `--game GAME` | checkpoint metadata | Override the game name or verify it against the environment ID stored in the checkpoint. |
| `--replay PATH` | embedded checkpoint replay | Optional `torch.save`d `EpisodeReplayBuffer` used for policy-search and unroll diagnostics. If omitted, embedded replay is used when present; otherwise diagnostics use a blank root and omit unroll metrics. |
| `--simulations N` | `25` | MCTS simulations per evaluated or recorded decision. |
| `--episodes N` | `5` | Number of raw-policy and MCTS evaluation episodes. Both modes use the same seeds. |
| `--seed N` | `0` | First evaluation seed; subsequent episodes use `N+1`, `N+2`, and so on. Also supplies the GIF seed unless `--gif-seed` is set. |
| `--record-gif PATH` | none | Record one real-environment episode to this GIF. Parent directories are created automatically. |
| `--gif-mode {mcts,raw}` | `mcts` | Choose actions using noise-free MCTS visit counts or the prediction network's raw-policy argmax. |
| `--gif-seed N` | `--seed` | Environment and action-space seed for the recorded episode. |
| `--gif-max-steps N` | checkpoint training limit | Maximum recorded transitions. Falls back to `2500` for a checkpoint without a stored training limit. Must be positive. |
| `--gif-fps N` | `20` | GIF playback frame rate. Must be positive. |
| `--gif-width PIXELS` | `480` | Width of the RGB gameplay pane. The native model-input panel is added beside it, so the complete GIF is wider. Must be positive. |
| `--output-json PATH` | none | Save the same diagnostic report printed to standard output as formatted JSON. Parent directories are created automatically. |

GIF-only options have no effect unless `--record-gif` is supplied. `-h` or `--help` prints the current diagnostic interface.

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

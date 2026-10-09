# MuZero from CartPole's four state variables

[Project comparison](../README.md) · [Central commands](../COMMANDS.md) · [Trained models](../checkpoints/README.md) · [Development](../docs/DEVELOPMENT.md)

`cartpole_muzero` trains MuZero directly from CartPole-v1 observations:
`[cart position, cart velocity, pole angle, angular velocity]`. This is
**native-state MuZero, checkpoint v15**. The separate `minatar_muzero/` package
learns from screenshots in MinAtar games.

## What learns

All networks start from random weights and are optimized together:

```text
real four-number observation -> h -> learned latent vector
                                       |
                                       f -> policy logits, value
                                       |
                     latent + action -> g -> next latent, reward
                                                |
                                                f -> policy logits, value
```

- **Representation h:** a fully connected encoder, 4 -> hidden -> latent.
- **Dynamics g:** latent plus one-hot action -> hidden layers -> next latent
  and scalar reward.
- **Prediction f:** separate fully connected policy and value branches.

Defaults are a 64-dimensional latent and 128 hidden units. Each representation
and recurrent latent is min-max normalized per sample to [0, 1]. The latent
is not constrained to equal the four physical variables and is not necessarily
an information-compressing representation. No frame history is needed because
the native observation already contains both velocities.

The learner uses MCTS visit counts for policy targets, real environment rewards
for reward targets, and n-step rewards plus a later MCTS root-value bootstrap
for value targets. Gradients propagate through h, g, and f across a root and
`--unroll-steps` recurrent predictions. There is no heuristic controller,
expert-action dataset, exact-physics search, next-physical-state regression,
image reconstruction, or latent-consistency objective. Search only calls g/f
below the real root; it never steps the environment.

Warm-up uses random actions. Those transitions train reward/value and their
upstream networks, while policy loss is masked. The policy output starts at
uniform probabilities. After warm-up, MCTS controls behavior and supplies raw
normalized visit targets. Actor temperature affects action sampling only.

Replay contains complete episodes. Terminal, time-limit, and explicit
`--max-steps` boundaries end the episode; the existing finite-episode target
convention uses zero bootstrap and absorbing zero reward/value continuation
at those boundaries. Policy targets beyond the boundary are masked.

## Run

Run from `teamproject_26_muzero/sprint4` in the existing project environment:

```bash
conda activate state-mcts
python -m unittest discover -s cartpole_muzero/tests -v

# Execution check, not a trained agent: save into a separate smoke directory.
python -m cartpole_muzero.train \
  --episodes 3 --max-steps 40 --simulations 5 \
  --warmup-episodes 2 --updates-per-episode 3 \
  --batch-size 8 --unroll-steps 3 \
  --checkpoint-path checkpoints/cartpole_muzero/smoke_state4d/run.pt

python -m cartpole_muzero.train --help
```

A starting configuration for a learning experiment follows. Save each training
seed in its own directory and evaluate on fresh seeds.

```bash
python -m cartpole_muzero.train \
  --episodes 800 --max-steps 500 --simulations 50 \
  --latent-dim 64 --hidden-dim 128 \
  --batch-size 64 --buffer-capacity 50000 --warmup-episodes 20 \
  --updates-per-transition 0.25 --min-updates-per-episode 1 \
  --max-updates-per-episode 50 --learning-rate 1e-4 \
  --discount 0.99 --reward-scale 0.1 --unroll-steps 5 \
  --evaluation-interval 20 --evaluation-episodes 20 --seed 0 \
  --checkpoint-path checkpoints/cartpole_muzero/state4d_seed0/run.pt
```

To resume, repeat the original options and checkpoint path, add
`--reuse-checkpoint`, and set `--episodes` to the number of **additional**
episodes. Architecture, environment, discount, and reward scale must match.
Keep the original seed and episode-limit settings for a consistent experiment.
The optimizer retains its moments, with the explicitly requested learning
rate and weight decay applied to its parameter groups.

Training and numeric evaluation do not render the environment. Use
`--latent-dim` and `--hidden-dim`; the former image-size, stack-size, and
latent-channels options no longer apply.

## Evaluate and record

The diagnostic compares random actions, raw policy argmax (h/f without search),
and latent MCTS on identical seeds. It inherits the checkpoint's search budget
unless `--simulations` is supplied. Evaluation disables root noise, resets
search randomness for each seed, and preserves the actor's random state.

```bash
python -m cartpole_muzero.diagnose \
  checkpoints/inference/cartpole_state4d.pt \
  --episodes 20 --seed 20000 \
  --output-json artifacts/cartpole_muzero/state4d_seed0/evaluation.json

python -m cartpole_muzero.diagnose \
  checkpoints/inference/cartpole_state4d.pt \
  --episodes 1 --seed 20000 \
  --record-gif artifacts/cartpole_muzero/gifs/state4d_seed0_mcts_seed20000.gif
```

The report retains every episode score, evaluation seeds, search budget and
policy/search agreement. Depth-wise value/reward errors require saved replay,
so use the original checkpoint or `--replay` for those diagnostics. These replay errors are training-data diagnostics, not held-out proof
of generalization. The checkpoint's replay is used automatically; `--replay`
can instead supply a saved replay state, checkpoint, or replay buffer.

GIFs show real environment rendering beside the exact **four numeric inputs**.
Rendering is only for display and never enters the model. Use `--gif-mode raw`
for policy-only play; default `mcts` uses learned-model search. GIF creation
requires Gymnasium's CartPole rendering dependency (pygame).

## Checkpoints and interpretation

The default path is `checkpoints/cartpole_muzero/cartpole_state_v15.pt`:

- Base `.pt`: best scheduled MCTS evaluation.
- `_latest.pt`: resumable state, written at evaluation points and run end.
- `_final.pt`: end of the requested run.
- `_rewards.png`: raw training rewards, moving average, periodic evaluation.

Checkpoints include all networks, architecture and observation-mode metadata,
optimizer, replay, histories, configuration, and Python/NumPy/Torch CPU random
states. Writes are atomic. Older pre-v8 and screenshot v8-v14 checkpoints are
incompatible and rejected before loading their weights. Existing archive files
are retained; their scores are not evidence for the new 4D model.

Compare this experiment with `state_mcts` to study joint learning without
heuristic labels on the same task and observations. That comparison changes
training and search as well as representation; it is not an isolated
representation ablation. Compare fresh random/raw/MCTS scores and multiple
training seeds before claiming successful learning. Passing execution checks
or fitting a tiny batch does not establish that CartPole has been solved.

## Saved run

The completed 800-episode run is in
`checkpoints/cartpole_muzero/state4d_seed0_R9ncWC/`. It contains `run.pt`,
`run_latest.pt`, `run_final.pt`, `training.log` and `run_rewards.png`.
The best checkpoint was selected at episode 800. Evaluate it with `cartpole_muzero.diagnose`; the portable identical-weight copy
is `checkpoints/inference/cartpole_state4d.pt`. Original paths are preserved.

Checkpoint files retain their original weights, optimizer, replay and training
history. Embedded paths describe their original locations and do not need to
be rewritten. `image_observation.py` remains an unused historical helper;
the active trainer and diagnostic use `environment.py`.

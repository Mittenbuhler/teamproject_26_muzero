# Legacy image MuZero (vanilla, checkpoint v14)

This directory contains a deliberately small MuZero for discrete-action environments. CartPole uses five normalized 32x32 grayscale screenshots, but `observation_shape`, `stack_size`, and the environment's discrete `action_space.n` configure the networks, so the same core can later accept MinAtar observations.

## Data flow

Only a real observation history enters the representation function:

`o[t-4:t] -> h -> spatial s[t,0] -> f -> (policy logits, value)`

Every searched or trained hypothetical step uses the recorded/discrete action and dynamics:

`(s[t,k-1], one-hot action planes) -> g -> (s[t,k], reward) -> f -> (policy logits, value)`

Both representation and dynamics min-max normalize each latent sample safely to `[0,1]`. There is no reconstruction, latent-consistency, terminal, target-encoder, recurrent-RNN, reanalyze, or distributional head.

The prediction function preserves spatial position. Its policy branch applies a small 1x1 convolution, ReLU, flattens every latent cell, and maps the resulting vector to action logits. Its value branch independently applies a 1x1 convolution, ReLU, flattening, and a small hidden linear layer before producing an unconstrained scalar. It does not use global average pooling: averaging all positions made object presence visible but largely discarded where the object was, which is harmful for both CartPole screenshots and MinAtar grids.

The representation derives `(latent_channels, latent_height, latent_width)` from the configured input shape. Prediction-head linear dimensions are constructed from that derived latent shape; neither 32x32 observations, an 8x8 latent, a 10x10 grid, nor a particular action count is hard-coded. A CartPole-like 5x32x32 input and configurable-channel MinAtar-like 10x10 input therefore use the same network classes. They still train separate weights/checkpoints.

## Replay and targets

Replay stores complete episodes: stacked observation, selected action, real environment reward, raw normalized MCTS visit-count policy, a policy-valid flag, MCTS root value, and both `terminated` and `truncated`. Sampling never joins episodes. For valid post-warm-up targets, the visit-count policy is always `N(a) / sum_b N(b)`. Actor temperature is applied only to the separate behavior distribution used to sample the environment action; it never sharpens the supervised policy target.

Warm-up transitions use random behavior actions and remain useful learner data for representation, dynamics, reward, and value. Their policy-valid flag is false, so the policy cross-entropy receives no gradient from the arbitrary visit distribution of an untrained search. The final policy linear layer is initialized to zero, which gives every discrete action an exactly uniform raw prior until valid policy learning begins instead of injecting an arbitrary initialization-specific favorite. Once warm-up ends, MCTS controls behavior and its raw visit counts become valid policy targets. Warm-up data therefore teaches the model from real trajectories without teaching the policy an accidental initial search bias.

For a root time `t`, the learner evaluates `(pi_t,z_t)` at the root. Dynamics step 1 consumes `a_t` and learns `(r_{t+1},pi_{t+1},z_{t+1})`; step `K` consumes `a_{t+K-1}` and learns `(r_{t+K},pi_{t+K},z_{t+K})`. Thus there are exactly one root prediction and `K` recurrent predictions, with no root reward loss.

The scalar value target is

`z_t = r_(t+1) + gamma r_(t+2) + ... + gamma^(n-1) r_(t+n) + gamma^n nu_(t+n)`.

Bootstrap is zero after termination. Real rewards are used without the former `-25` penalty. Past a terminal/truncated transition, targets form an absorbing continuation: dummy action 0, reward/value zero, and masked policy. Warm-up policy masking is preserved at both root and recurrent depths when those transitions are sampled. The loss is mean policy cross-entropy plus scalar value/reward MSE; terms are divided by their valid counts. Weight decay, clipping, recurrent gradient scaling, and each loss weight are configurable.

`--reward-scale` optionally changes the units used by the learned model. Raw environment rewards remain in episode logs and plots. Replay reward targets and n-step reward sums use `raw_reward * reward_scale`; MCTS value bootstraps and network reward/value predictions stay in those same scaled units. There is no one-sided rescaling, categorical support, or nonlinear value transform.

## Search

Each simulation selects with PUCT and min-max-normalized observed edge Q, expands one unexpanded edge with exactly one `g` call, evaluates its state with one `f` call, and backs up predicted reward plus discounted leaf value. Nodes store prior `P`, count `N`, sum `W`, mean `Q`, edge reward `R`, and child latent state. Search never queries the environment. Training enables root Dirichlet noise. Its raw normalized visit counts become the replay policy target, while the temperature-adjusted visit distribution selects the behavior action. Evaluation disables noise and chooses the largest visit count.

## Commands

Run from `teamproject_26_muzero/sprint4` using the project environment:

```bash
../../.venv/bin/python -m pytest -q legacy_muzero/tests

# short smoke run
../../.venv/bin/python -m legacy_muzero.train_policy_value --episodes 2 --max-steps 20 --simulations 2 --updates-per-episode 1 --batch-size 2 --unroll-steps 2

# inspect every configurable value
../../.venv/bin/python -m legacy_muzero.train_policy_value --help

# fuller fixed-update training
../../.venv/bin/python -m legacy_muzero.train_policy_value --episodes 300 --simulations 50 --updates-per-episode 10

# continue the same v14 run for 100 additional episodes
../../.venv/bin/python -m legacy_muzero.train_policy_value --episodes 100 --simulations 50 --updates-per-episode 10 --checkpoint-path checkpoints/legacy_muzero/cartpole_policy_v14.pt --reuse-checkpoint

# evaluation and current diagnostics (raw and MCTS use identical seeds)
../../.venv/bin/python -m legacy_muzero.diagnose_latent_muzero checkpoints/legacy_muzero/cartpole_policy_v14.pt --episodes 10

# record the current v14 checkpoint as a composite MCTS GIF
../../.venv/bin/python -m legacy_muzero.diagnose_latent_muzero checkpoints/legacy_muzero/cartpole_policy_v14.pt --episodes 1 --simulations 50 --record-gif artifacts/legacy_muzero/gifs/cartpole_policy_v14_mcts_seed4200.gif --gif-mode mcts --gif-seed 4200

# the same composite recording remains available for the archived v12 checkpoint
../../.venv/bin/python -m legacy_muzero.diagnose_latent_muzero checkpoints/legacy_muzero/archive/v12/cartpole_spatial_v12.pt --episodes 1 --simulations 50 --record-gif artifacts/legacy_muzero/gifs/cartpole_spatial_v12_mcts_seed42.gif --gif-mode mcts --gif-seed 42

# record the policy network's raw argmax on the same seed for comparison
../../.venv/bin/python -m legacy_muzero.diagnose_latent_muzero checkpoints/legacy_muzero/archive/v12/cartpole_spatial_v12.pt --episodes 1 --simulations 50 --record-gif artifacts/legacy_muzero/gifs/cartpole_spatial_v12_raw_seed42.gif --gif-mode raw --gif-seed 42
```

The diagnostic reports raw-policy and visit entropy, their L1 change and argmax agreement, real-root value error (depth 0), recurrent value error/correlation, reward error and latent norms by depth, and raw-policy versus MCTS evaluation scores. Supply `--replay path.pt` for depth-wise target metrics.

`--record-gif` now creates a composite diagnostic frame by default. The left pane is the real RGB frame rendered by the environment, not a reconstruction or imagined latent state. The right pane is the exact normalized grayscale observation history passed to the representation network for that decision, arranged vertically from oldest to newest and labeled with padding/time offsets. It includes the initial zero-padded history, updates before every selected action, and shows the final stack alongside the terminal frame. This makes screenshot preprocessing and temporal context directly inspectable instead of inferring them from the RGB animation.

There is no separate composite-view switch: supplying `--record-gif PATH` enables the composite recording, while omitting `--record-gif` disables GIF creation entirely. `--gif-width` controls the width of the left RGB gameplay pane, so the complete composite GIF is wider than that value. `--gif-fps` controls playback and `--gif-max-steps` controls the recording limit; the latter defaults to the checkpoint's training limit. The destination's parent directory is created automatically, and generated recordings belong in `artifacts/legacy_muzero/gifs/`.

The overlay shows the step, accumulated raw environment score, selected integer action, raw policy probabilities, and predicted value. In `--gif-mode raw`, the policy argmax acts directly and neither dynamics nor search is used. In `--gif-mode mcts`, the policy probabilities are still used as MCTS priors together with learned dynamics, rewards, and leaf values; the most-visited root action acts in the environment. Recording is evaluation-like: root noise is disabled and no behavior temperature is applied. The MCTS overlay additionally shows normalized raw visit counts and the backed-up root value.

## Configuration

Environment and architecture defaults are `--env CartPole-v1`, `--max-steps 500`, `--image-size 32`, `--stack-size 5`, and `--latent-channels 32`. The discrete action count always comes from `env.action_space.n`.

Optimizer controls are `--learning-rate 3e-4`, `--weight-decay 1e-5`, `--gradient-clip-norm 5`, and `--dynamics-gradient-scale 0.5`. A resumed optimizer retains moments but every parameter group is explicitly updated to the learning rate and weight decay requested by the new command.

Replay/update controls are `--buffer-capacity 20000`, `--warmup-episodes 1`, `--updates-per-episode 5`, `--updates-per-transition 0`, `--min-updates-per-episode 1`, and `--max-updates-per-episode 50`. A nonpositive transition rate uses the fixed update count. A positive rate uses `ceil(trajectory_length * rate)`, clipped to the configured minimum and maximum. Warm-up uses random behavior actions but now performs the configured learner updates as soon as replay data is available. Its policy targets are masked; value and reward losses remain active and train all applicable model paths. The selected mode is printed and checkpointed.

Target/loss controls are `--discount 0.997`, `--bootstrap-steps 10`, `--unroll-steps 5`, `--reward-scale 1`, and the three `--*-loss-weight` arguments, each defaulting to 1.

Search controls are `--simulations 25`, `--pb-c-base 19652`, `--pb-c-init 1.25`, `--root-dirichlet-alpha 0.25`, and `--root-exploration-fraction 0.25`. Root noise is used only by the actor; evaluation always disables noise and uses temperature zero.

The actor temperature phases are controlled by `--temperature-initial`, `--temperature-middle`, `--temperature-final`, `--temperature-initial-episodes`, and `--temperature-middle-episodes`. Episode indices below the first boundary use the initial value, indices below the second use the middle value, and later indices use the final value. Defaults are `1.0/0.5/0.25` with boundaries `200/500`, deliberately keeping broad behavior exploration much longer than v13. These values affect behavior action sampling only, never the replay policy target. During random-action warm-up the scheduled temperature is reported but does not select the environment action.

Evaluation defaults to `--evaluation-interval 10 --evaluation-episodes 3`. All sizes and counts, PUCT values, noise settings, temperatures, optimizer controls, and phase boundaries are validated before environment construction.

## Stable CartPole example

```bash
../../.venv/bin/python -m legacy_muzero.train_policy_value \
  --env CartPole-v1 \
  --episodes 800 \
  --max-steps 500 \
  --image-size 32 \
  --stack-size 5 \
  --latent-channels 32 \
  --simulations 50 \
  --batch-size 64 \
  --buffer-capacity 50000 \
  --warmup-episodes 20 \
  --updates-per-transition 0.25 \
  --min-updates-per-episode 1 \
  --max-updates-per-episode 50 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --gradient-clip-norm 5 \
  --dynamics-gradient-scale 0.5 \
  --discount 0.99 \
  --bootstrap-steps 10 \
  --unroll-steps 5 \
  --reward-scale 0.1 \
  --policy-loss-weight 1 \
  --value-loss-weight 1 \
  --reward-loss-weight 1 \
  --pb-c-base 19652 \
  --pb-c-init 1.25 \
  --root-dirichlet-alpha 0.25 \
  --root-exploration-fraction 0.25 \
  --temperature-initial 1.0 \
  --temperature-middle 0.5 \
  --temperature-final 0.25 \
  --temperature-initial-episodes 200 \
  --temperature-middle-episodes 500 \
  --evaluation-interval 20 \
  --evaluation-episodes 20 \
  --checkpoint-path checkpoints/legacy_muzero/cartpole_policy_v14.pt
```

Checkpoint v14 stores the full derived latent shape, both position-preserving head configurations, `policy_target_mode=raw_visit_counts`, and the per-transition policy-valid state required by the masked warm-up. Version 13 has the same prediction architecture but its warm-up replay treats untrained MCTS visits as valid policy supervision and suppresses learner updates during warm-up. It must not be resumed into v14; start a fresh v14 run. Version 12 remains usable by the documented diagnostic/GIF commands for evaluation, while version 11 and older checkpoints remain architecturally incompatible. Training rejects incompatible versions before partially loading weights or replay. `<stem>_latest.pt` is the resumable v14 checkpoint and contains networks, optimizer, replay, complete configuration, raw scores, completed-episode count, and random states. `--reuse-checkpoint` finds that companion automatically and treats `--episodes` as additional episodes. Repeat the architecture, discount, reward scale, and warm-up semantics used by the original run; changing discount or reward scale would invalidate stored replay targets and is rejected. `<stem>.pt` remains the best-evaluation checkpoint and `<stem>_final.pt` records the end of the most recent run.

Every checkpoint is written to a temporary file in its destination directory, flushed, and atomically replaced so readers never observe a partially written PyTorch archive.

At the end of training, the trainer also writes `<checkpoint_stem>_rewards.png` beside the checkpoint. It plots raw episode reward, a 10-episode moving average, and periodic MCTS evaluation reward as separate series.

Historical checkpoints are preserved under `checkpoints/legacy_muzero/archive/`. See the [checkpoint archive manifest](../checkpoints/legacy_muzero/README.md) for experiment names, completed episodes, file roles, and compatibility. New policy-masked warm-up runs use distinct v14 names at the checkpoint root. Generated episode recordings belong in `artifacts/legacy_muzero/gifs/`.

Native MinAtar execution still needs a small observation adapter that converts its channel-last or channel-first multi-channel grid into the configured `[channels, height, width]` float tensor and bypasses screenshot rendering/frame stacking as appropriate. Environment selection must also expose each game's discrete `action_space.n`. No MinAtar-specific model architecture is required.

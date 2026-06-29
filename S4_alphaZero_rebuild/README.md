# CHATTest

Fresh CartPole rebuild that separates the pieces you asked for:

- `models.py`: dynamics model, policy network, value network, and a future CNN representation adapter.
- `mcts.py`: model-based MCTS. It does not clone Gym envs. It uses:
  - policy NN probabilities as action priors during selection,
  - the dynamics model to expand predicted next states and rewards,
  - the value NN to evaluate newly reached leaf states.
- `train_dynamics.py`: collects real CartPole vector transitions and trains the one-step dynamics model.
- `train_policy_value.py`: trains policy/value networks from MCTS visit-count and episode-return targets.
- `image_observation.py`: downscaled grayscale screenshot interface for the future CNN path.
- `cartpole_demo.py`: end-to-end demo that trains/loads checkpoints and writes visual artifacts.

## Setup

```bash
source .venv/bin/activate
```

If imports fail, make sure the active Python is this folder's venv:

```bash
which python
python -m pip install -r requirements.txt
```

You can also bypass activation:

```bash
./.venv/bin/python cartpole_demo.py --reuse-checkpoints
```

## Train Dynamics Explicitly

This learns the one-step CartPole dynamics model:

```bash
python train_dynamics.py \
  --collect-episodes 250 \
  --train-steps 1500 \
  --save-path checkpoints/dynamics_cartpole.pt
```

Checkpoint:

- `checkpoints/dynamics_cartpole.pt`

This model learns:

```text
state + action -> next_state + reward
```

## Train Policy/Value Explicitly

This requires a trained dynamics checkpoint first. It uses model-based MCTS to create policy targets from visit counts and value targets from episode returns:

```bash
python train_policy_value.py \
  --dynamics-path checkpoints/dynamics_cartpole.pt \
  --episodes 80 \
  --exploration-episodes 30 \
  --value-discount 0.99 \
  --simulations 50,30 \
  --save-best-checkpoint \
  --simulation-upgrade-reward-threshold 50 \
  --simulation-upgrade-window 10 \
  --checkpoint-eval-interval 5 \
  --checkpoint-eval-episodes 5 \
  --save-path checkpoints/policy_value_cartpole.pt \
  --loss-plot-path artifacts/policy_value_loss.png \
  --training-plot-path artifacts/policy_value_training_progress.png
```

Outputs:

- `checkpoints/policy_value_cartpole.pt`
- `artifacts/policy_value_loss.png`
- `artifacts/policy_value_training_progress.png`

The policy learns:

```text
state -> MCTS visit-count action probabilities
```

During search, these probabilities guide exploration but do not play simulated
rollouts. The value network directly evaluates each new leaf state instead.
Predicted rewards are normalized to the same scale as the value targets before
they are backed up through the tree.

The value network learns discounted return targets computed from the real rewards collected after each timestep:

```text
state_t -> normalized(reward_t + gamma*reward_t+1 + gamma^2*reward_t+2 + ...)
```

`--exploration-episodes` controls how long action selection stays stochastic during policy/value training. During those episodes, actions are sampled from the MCTS visit-count policy. After that, training switches to deterministic argmax actions. Increasing it can help avoid early collapse into a bad policy.

`--value-discount` is the gamma value used for those return targets.

`--simulations 50` uses 50 MCTS search iterations before every real CartPole
action. With best-checkpoint mode, `--simulations 50,30` starts training at 30
and permanently increases it to 50 once the recent average reward reaches the
configured threshold. `--simulation-upgrade-reward-threshold` sets that
threshold and `--simulation-upgrade-window` sets the averaging window.

### Best-checkpoint evaluation

`--save-best-checkpoint` changes `--save-path` from "save the final networks" to
"keep the best networks observed during training."

`--checkpoint-eval-interval 5` runs an evaluation after every five training
episodes: episodes 5, 10, 15, and so on. The last training episode is always
evaluated even when it is not a multiple of the interval.

`--checkpoint-eval-episodes 5` means that each evaluation plays five complete
CartPole episodes. These episodes:

- use deterministic MCTS action selection,
- do not add data to the replay buffer,
- do not update either network,
- reuse the same five fixed seeds at every checkpoint evaluation.

Using the same seeds makes scores from different points in training directly
comparable. Their rewards are averaged into one checkpoint score. If that mean
reward is higher than every earlier checkpoint score, the current policy/value
weights overwrite `--save-path`. A single unusually good episode therefore
cannot select the checkpoint by itself.

For example, evaluation rewards of `44, 53, 48, 51, 49` produce a checkpoint
score of `49`. The checkpoint is saved only if `49` exceeds the previous best
mean score.

Checkpoint evaluation uses the currently active MCTS simulation count. With
`--simulations 50,30`, evaluations use 30 simulations before the reward
threshold is reached and 50 afterward.

When `--save-best-checkpoint` is omitted, no periodic checkpoint evaluation is
performed and `--save-path` stores the networks from the final training episode.

## End-To-End Demo

This loads existing checkpoints when available, otherwise trains missing pieces:

```bash
python cartpole_demo.py --reuse-checkpoints --save-best-checkpoint
```

With `--save-best-checkpoint`, the demo loads the best version from the normal
`policy_value_cartpole.pt` file for its final evaluation playthrough.

The final playthrough uses seed `234`, the best seed found in the earlier
20-run evaluation. Use `--playthrough-seed` to select a different seed.

Policy/value checkpoints created by the earlier policy-rollout search are
detected as incompatible and retrained automatically.

Artifacts currently written by the demo:

- `artifacts/downscaled_grayscale_observations.png`: first 84x84 grayscale observations from the evaluation playthrough.
- `artifacts/cartpole_playthrough.gif`: animation of the evaluation playthrough.
- `artifacts/policy_value_loss.png`: policy/value loss plot, when policy/value training runs.
- `artifacts/policy_value_training_progress.png`: episode rewards, 10-episode average reward, and policy/value losses.

Use `--gif-fps` to control the playback speed of the generated GIF.

The current algorithm still uses vector CartPole states as the model state. The screenshot path is intentionally present as a clean boundary for the next step: replace direct vector states with `ImageRepresentationNetwork(screenshot)` latent states, then train dynamics/policy/value on those latents.

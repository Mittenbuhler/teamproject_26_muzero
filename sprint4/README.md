# Image-Latent MuZero CartPole

This implementation learns entirely from rendered CartPole observations. Gym's
four-value state vector is not given to the networks.

```text
x[t-4:t] -> representation -> s[t]
s[t] -> policy, value
s[t], action -> dynamics -> s[t+1], reward
```

There is no RNN. The representation input is a fixed stack of five grayscale
`32x32` frames. At the beginning of an episode, unavailable history is filled
with zero frames:

```text
t=0: [0, 0, 0, 0, x0]
t=1: [0, 0, 0, x0, x1]
t=2: [0, 0, x0, x1, x2]
...
t=4: [x0, x1, x2, x3, x4]
```

## Networks

The representation function follows the supplied architecture, adjusted to the
required five-frame input:

```text
Conv2d(5 -> 8, kernel=3, stride=2)
ReLU
Conv2d(8 -> 16, kernel=3, stride=2)
ReLU
Flatten
Linear(16*7*7 -> 64)
ReLU
Linear(64 -> 32)
```

The resulting 32-dimensional latent state is the only input used by the policy,
value, and dynamics networks.

## MCTS

The tree logic remains model-based:

1. The policy network supplies action priors during expansion.
2. Selection combines predicted reward, backed-up mean value, and the prior
   exploration bonus.
3. The dynamics model predicts every child latent state and immediate reward.
4. The value network evaluates newly reached latent leaves.
5. Value and predicted rewards are backed up through the selected path.
6. Root visit counts form the search policy used to choose the real action.

The real environment decides episode termination. Imagined latent states do not
currently have a terminal head.

## Joint Training

Training starts with a short random-action warm-up so the replay buffer contains
varied visual transitions before MCTS depends on the initially random model.
After warm-up, action temperature anneals from `1.0` but cannot fall below
`0.5` until deterministic checkpoint evaluation reaches reward `20`. Once the
gate unlocks, temperature gradually anneals from its current value toward
`0.25`. Checkpoint evaluation itself always uses temperature `0`.

One replay sample contains:

```text
frame stack, action, next frame stack,
MCTS visit policy, full-episode value target, shaped learning reward
```

All four networks are optimized together:

- **Policy loss:** predicted policy versus MCTS visit counts.
- **Value loss:** predicted value versus the discounted shaped rewards from that
  state through the actual end of the episode, scaled by the maximum discounted
  return. This target does not bootstrap from an estimated MCTS value.
- **Reward loss:** dynamics reward prediction versus the shaped transition reward.
- **Latent consistency loss:** predicted next latent versus the representation
  of the real next five-frame observation.

The consistency target is detached, while policy, value, reward, and transition
gradients train the representation function itself.

### Failure Reward And Discount

CartPole still reports `+1` on the transition where the pole falls. For
learning, that final reward is replaced with `-10` when Gym reports
`terminated=True`. Ordinary transitions remain `+1`. A time-limit
`truncated=True` transition is not penalized because reaching the limit is a
successful run.

The default value discount is `0.997`. Together, the high discount and negative
failure reward make a late failure more valuable than an early failure and give
the dynamics reward head a direct signal for dangerous transitions. Plots,
evaluation, simulation-upgrade thresholds, and best-checkpoint selection still
use the unmodified real episode reward, so their numbers remain episode length.

Full-episode returns are the default (`--value-target-mode full-episode`). They
are calculated backward after each episode, so its eventual failure or success
directly affects every preceding state. The previous 10-step MCTS-bootstrap
target remains available for comparison with `--value-target-mode n-step` and
`--bootstrap-steps 10`.

## Setup

From `sprint4`:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train Directly

```bash
python train_policy_value.py \
  --episodes 800 \
  --warmup-episodes 20 \
  --exploration-episodes 600 \
  --minimum-temperature 0.25 \
  --temperature-hold 0.5 \
  --temperature-unlock-reward-threshold 20 \
  --learning-rate 0.00015 \
  --value-discount 0.997 \
  --terminal-penalty -10 \
  --value-target-mode full-episode \
  --simulations 50 \
  --batch-size 64 \
  --updates-per-episode 10 \
  --save-best-checkpoint \
  --checkpoint-eval-interval 20 \
  --checkpoint-eval-episodes 20 \
  --save-path checkpoints/latent_muzero_cartpole.pt \
  --loss-plot-path artifacts/latent_muzero_loss.png \
  --training-plot-path artifacts/latent_muzero_training_progress.png
```

The first latent run should use a fixed search budget because there are no
reliable image-based reward statistics yet. Adaptive search remains available:
`--simulations 60,30` starts at 30 and switches to 60 when the configured reward
threshold and averaging window are reached. Its default upgrade threshold is
now `20`; `--simulations 50` is fixed at 50 for the entire run.

## End-to-End Demo

Train missing networks, select the best unified checkpoint, and produce the
annotated side-by-side GIF:

```bash
python cartpole_demo.py \
  --train \
  --episodes 400 \
  --simulations 50 \
  --eval-simulations 90
```

Play an existing compatible latent checkpoint without retraining:

```bash
python cartpole_demo.py
```

An existing `checkpoints/latent_muzero_cartpole.pt` is loaded by default. The
demo prints its absolute path, number of saved training episodes, best
evaluation, and value-target mode before the run. Use `--train` only when you
intend to train a new model and allow the checkpoint file to be overwritten.
The older `--reuse-checkpoints` spelling remains accepted as a compatibility
alias.

The demo writes:

- `checkpoints/latent_muzero_cartpole.pt`: representation, dynamics, policy,
  value, architecture metadata, and training history.
- `artifacts/latent_muzero_loss.png`: policy, value, reward, and latent losses.
- `artifacts/latent_muzero_training_progress.png`: rewards and total joint loss.
- `artifacts/latent_cartpole_side_by_side.gif`: current grayscale input beside
  Gym RGB, annotated with frame-history count, learned value, policy
  probabilities, MCTS visit shares, action, and cumulative reward.

## Best Checkpoint

Best-checkpoint evaluation remains appropriate for latent learning because the
complete system is evaluated together: representation, dynamics, policy, and
value. Every evaluation uses the same 20 fixed seeds and deterministic MCTS.
The log reports the ordinary reward mean and sample standard deviation, and the
training plot shows the same values as error bars. When mean real-environment
reward improves, the single latent checkpoint is overwritten. Evaluation
episodes never update the networks or replay buffer.

Old vector-state dynamics and policy/value checkpoints are intentionally
incompatible with this pipeline and are not loaded.

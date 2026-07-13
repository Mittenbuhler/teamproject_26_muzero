# Modular real-state CartPole MCTS

## Command overview

Run these commands from `teamproject_26_muzero/sprint4`:

```bash
# Show every CLI option
python -m state_mcts.experiment --help

# Exact-dynamics MCTS baseline: UCT + random rollouts
python -m state_mcts.experiment --models none

# Train and evaluate exactly one learned component
python -m state_mcts.experiment --models dynamics
python -m state_mcts.experiment --models policy
python -m state_mcts.experiment --models value

# Train all components and evaluate only the complete system
python -m state_mcts.experiment --models all

# Train all components and evaluate all eight component combinations
python -m state_mcts.experiment --models all --ablation

# Re-evaluate compatible existing checkpoints without retraining
python -m state_mcts.experiment --models all --ablation --reuse-checkpoints

# More reliable evaluation using fixed seeds and a separate report path
python -m state_mcts.experiment \
  --models all \
  --ablation \
  --eval-episodes 50 \
  --eval-seed 10000 \
  --report-path artifacts/state_mcts/state_mcts_ablation_report.json

# Change the global MCTS lookahead for any model combination. When dynamics is
# trained, its multi-step horizon automatically uses the same value.
python -m state_mcts.experiment --models dynamics --search-depth 40
python -m state_mcts.experiment --models policy,value --search-depth 40

# Delay learned value bootstrapping until after 10 simulated tree steps
python -m state_mcts.experiment --models value --bootstrap-after 10

# Sidecar diagnostic: train policy and/or value from vanilla-MCTS statistics
python -m state_mcts.mcts_distillation --models policy
python -m state_mcts.mcts_distillation --models value
python -m state_mcts.mcts_distillation --models policy,value --bootstrap-after 10

# Fast smoke run
python -m state_mcts.experiment \
  --models all \
  --train-samples 1000 \
  --train-epochs 5 \
  --simulations 24 \
  --eval-episodes 2

# Run focused tests or the complete state/legacy regression suite
python -m unittest -v state_mcts.tests.test_experiment state_mcts.tests.test_dashboard
python -m unittest -v state_mcts.tests.test_experiment state_mcts.tests.test_dashboard legacy_muzero.tests.test_pipeline

# Rebuild and open the read-only dashboard
python -m state_mcts.dashboard
open artifacts/state_mcts/diagnostics_dashboard.html

# Run editable mode (persistent run names)
python -m state_mcts.dashboard_server
# The editable dashboard opens automatically in your default browser.
```

Stop the local dashboard server with `Ctrl+C`. Reports are append-only by
default; use a distinct `--report-path` when you want a separate experiment
family rather than another run in the existing history.

This experiment is deliberately separate from the image/latent MuZero path. It
uses the real four-value observation returned by `CartPole-v1`:

`[cart position, cart velocity, pole angle, pole angular velocity]`

The baseline is ordinary MCTS with:

- Gym-equivalent deterministic CartPole transitions;
- ordinary UCT selection with no learned action prior;
- random rollouts for leaf evaluation.

Each learned model replaces exactly one baseline function:

| Toggle | Baseline | Learned replacement |
|---|---|---|
| `dynamics` | exact CartPole equations | `DynamicsModel.predict` |
| `policy` | UCT exploration | PUCT with `PolicyNetwork.action_probs` |
| `value` | random rollout | `ValueNetwork.value` |

The models share collected Gym transitions but have separate targets,
optimizers, validation metrics, and checkpoint files. Their training processes
therefore do not depend on one another.

## Dynamics training

Dynamics training uses contiguous real Gym trajectories, not independently
shuffled transition targets. Its maximum training rollout is always equal to
`--search-depth`. For the default MCTS depth of 30, the curriculum is:

```text
5 steps -> 10 steps -> 20 steps -> 30 steps
```

At every stage the model starts from a real state, consumes the real recorded
action sequence, and recursively feeds its predicted state into the next
transition. Losses after the real episode boundary are masked. The objective
keeps a strong one-step anchor and downweights increasingly distant bands:

| Predicted step | Smooth-L1 band weight |
|---|---:|
| 1 | 1.0 |
| 2-5 | 0.5 |
| 6-10 | 0.25 |
| 11-20 | 0.125 |
| 21-search depth | 0.0625 |

Each band is averaged before weighting, so longer horizons do not inflate the
loss merely by containing more time steps. Train/validation splits use whole
episodes to prevent neighboring-transition leakage. Validation reports open-
loop state MAE at horizons 1, 5, 10, 20, and the configured search depth.

CartPole's immediate reward is always `+1`, so learned-dynamics MCTS hardcodes
that reward and trains only state rollout quality. The model's reward head is
not used. The analytic CartPole position/angle bounds determine termination
from each predicted real state; there is no termination head.

Changing `--search-depth` changes the maximum dynamics training horizon. A
checkpoint trained at a different depth or with the older one-step scheme is
rejected by `--reuse-checkpoints`, forcing an explicit retraining rather than a
silent mismatch.

`--search-depth` is a global search parameter and works with every `--models`
combination. It controls MCTS tree lookahead and the length of random leaf
rollouts. If dynamics is selected for training, it additionally controls the
dynamics multi-step horizon. It does **not** alter policy labels or value
training targets. Value targets use the fixed episode horizon (`--max-steps`)
and are converted onto the current search-depth scale only when MCTS evaluates
a leaf.

## Policy learning

The policy network is supervised imitation learning, not policy-gradient or
self-play training. Every real Gym state receives an action label from this
fixed stabilizing controller:

```text
score = pole_angle
      + 0.45 * pole_angular_velocity
      + 0.02 * cart_position
      + 0.08 * cart_velocity

label = right if score > 0, otherwise left
```

Dataset collection follows that label with probability
`--expert-probability` (default `0.5`) and otherwise executes a random action.
This gives the dataset broader state coverage, but the supervised label is
always the heuristic action, regardless of which action Gym actually received.

`PolicyNetwork` maps the four real state values through two ReLU hidden layers
and a two-output softmax. It is optimized with Adam and negative log likelihood
(equivalent here to cross-entropy against the left/right label). The dataset is
split by complete episodes, batches are reshuffled each epoch, and validation
reports classification accuracy on held-out episodes. The loss graph records
mean negative log likelihood per epoch.

Inside MCTS, the predicted probabilities do not directly select the played
action. They replace uniform UCT exploration with a PUCT prior:

```text
exploration = c * policy_probability * sqrt(parent_visits)
              / (1 + child_visits)
```

MCTS still chooses the root action with the largest visit count. Therefore a
high policy accuracy means the network reproduced the heuristic labels; it does
not by itself prove that search performance improved. `--search-depth` changes
how long MCTS searches, but it does not change policy targets or policy
training.

## Value learning

The value network also uses supervised targets rather than Monte Carlo returns
from MCTS. For each real Gym state, the exact CartPole simulator follows the
same stabilizing heuristic for at most `--max-steps` steps. Its target is:

```text
value_target = survived_steps / max_steps
```

Targets therefore lie in `[0, 1]` and represent the fraction of the configured
episode horizon that the heuristic controller is expected to survive from that
state. They are not immediate rewards, search visit counts, or targets produced
by the policy network. The target is generated independently from the fixed
heuristic even when only the value model is selected.

`ValueNetwork` maps the four state values through two ReLU hidden layers to one
`tanh` output. Training uses Adam and mean squared error against the normalized
survival target. Validation reports mean absolute error on held-out episodes,
while the loss graph records mean squared training error per epoch. A reused
value checkpoint must have the same `--max-steps` target horizon.

At an MCTS leaf, the network output is clipped to `[0, 1]` and used directly as
a broad state-quality estimate:

```text
leaf_value = clipped_value
```

This replaces the baseline random leaf rollout. The important separation is:
`--search-depth` decides the maximum MCTS tree depth, while the value network
remains a general estimate of state quality over the configured episode
horizon. `--bootstrap-after` optionally sets the minimum simulated tree depth
before a learned value evaluator may be used. Its default is `0`, preserving
the immediate-bootstrap behavior. For example, `--bootstrap-after 10` means MCTS
must simulate at least ten CartPole transition steps before asking the value
network for a leaf value, unless it reaches a terminal state earlier.
`--search-depth` and `--bootstrap-after` do not change value training targets.
`--train-epochs` controls optimization passes for both policy and value and is
independent of search depth.

## MCTS-teacher distillation

`mcts_distillation.py` is the non-heuristic teacher path. It does not change the
default heuristic-target training used by `state_mcts.experiment`; it is a
separate diagnostic for asking whether policy and/or value can reproduce
vanilla MCTS search statistics.

The teacher is exact-dynamics vanilla MCTS with uniform priors and random leaf
rollouts. During each teacher episode, every root search gives the policy
target:

```text
policy_target = root child visits / total root child visits
```

The policy target is the soft visit distribution rather than only the most
visited action, so the network learns how strongly MCTS preferred each action.

The value target is the normalized discounted return-to-go from the rest of the
completed teacher episode:

```text
value_target = (r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...)
               /
               (1 + gamma + gamma^2 + ... over value_horizon)
```

This is the intended AlphaZero-style split: the policy learns from search visit
counts, while the value learns from what actually happened after the teacher
acted from that state. For CartPole this is normalized future survival time.
With `--discount 1.0`, a state with 250 future alive steps and
`--value-horizon 500` gets target `0.5`; with `--discount 0.99`, near-term
survival is weighted a bit more strongly.

Toggle which network is trained and evaluated with `--models`:

```bash
python -m state_mcts.mcts_distillation --models policy
python -m state_mcts.mcts_distillation --models value
python -m state_mcts.mcts_distillation \
  --models value \
  --discount 0.99 \
  --value-horizon 500
python -m state_mcts.mcts_distillation \
  --models policy,value \
  --teacher-simulations 64 \
  --search-depth 30 \
  --bootstrap-after 10 \
  --train-samples 10000
```

It writes separate policy/value checkpoints and appends each distillation run to
its own report history:

- `checkpoints/state_mcts/state_policy_mcts_teacher.pt`
- `checkpoints/state_mcts/state_value_mcts_teacher.pt`
- `artifacts/state_mcts/mcts_distillation_report.json`
- selected model loss SVGs under `artifacts/state_mcts/losses/<run_id>/`

The older visit-weighted tree-value target was intentionally removed from this
path because it did not match how we want the value network to generalize as a
leaf evaluator.

## Value-vs-MCTS action diagnostic

`value_mcts_diagnostic.py` checks whether the current value checkpoint agrees
with vanilla MCTS on local action ordering. It regenerates teacher-MCTS states,
runs a fresh root search, and compares:

```text
MCTS action  = root child with most visits
value action = argmax_a reward(s, a) / search_depth + V(next_state(s, a))
```

Run it after a value distillation run:

```bash
python -m state_mcts.value_mcts_diagnostic \
  --samples 500 \
  --teacher-simulations 64 \
  --search-depth 30 \
  --seed 0
```

To keep mostly decisive examples in the raw JSON records while still computing
summary metrics over all sampled states:

```bash
python -m state_mcts.value_mcts_diagnostic \
  --samples 5000 \
  --teacher-simulations 64 \
  --search-depth 30 \
  --seed 0 \
  --store-records 100 \
  --store-record-mode high-confidence \
  --store-confidence-threshold 0.70
```

It appends to `artifacts/state_mcts/value_mcts_action_diagnostic.json`.
Low agreement means the scalar value network may have learned reasonable global
returns while still giving the wrong local left/right ordering that value-only
MCTS needs for control.

## Run

From this directory:

```bash
python -m state_mcts.experiment --models none
python -m state_mcts.experiment --models dynamics
python -m state_mcts.experiment --models policy
python -m state_mcts.experiment --models value
python -m state_mcts.experiment --models all --ablation
```

`--models all --ablation` trains each model once and evaluates all eight model
subsets. Use `--reuse-checkpoints` to repeat evaluation without retraining.
Each run appends a timestamped entry to the JSON report, by default
`artifacts/state_mcts/state_mcts_report.json`. Existing runs are never overwritten. A run
is saved before training, after training, after each ablation, and on failure or
interruption, so long dynamics evaluations still leave a useful partial record.

Per-epoch optimization losses are written separately from the main report:

```text
artifacts/state_mcts/losses/<run_id>/dynamics.svg
artifacts/state_mcts/losses/<run_id>/policy.svg
artifacts/state_mcts/losses/<run_id>/value.svg
```

Only selected models receive a graph. Each self-contained SVG plots mean
training loss against epoch and includes the loss definition, final validation
metrics, and relevant training settings. Hover over a point for its exact epoch
and loss. Change the location with `--loss-dir`. No JSON files are created in
this loss folder. With `--reuse-checkpoints`, the stored checkpoint history is
plotted; older checkpoints without history receive a graph explaining that the
curve is unavailable.

For a quick smoke test:

```bash
python -m state_mcts.experiment \
  --models all \
  --ablation \
  --train-samples 1000 \
  --train-epochs 5 \
  --simulations 24 \
  --eval-episodes 2
```

For less noisy comparisons, use at least 20 evaluation episodes and retain the
same `--eval-seed` across configurations.

## Visual dashboard

Build the self-contained dashboard for `artifacts/state_mcts/state_mcts_report.json`:

```bash
python -m state_mcts.dashboard
```

Then open `artifacts/state_mcts/diagnostics_dashboard.html`. The horizontal mean-reward
chart makes learning runs easy to compare, while each run also gets an episode-
reward bar chart for variance and outliers. Clicking either graph opens that
run's exact JSON. The complete `state_mcts_report.json` remains available in one
collapsed section. Older MuZero diagnostic files are intentionally excluded.
Re-run the command whenever the state report changes; it has no external
package or network dependency.

Opening the HTML file directly is read-only. To edit run display names and save
them back into `state_mcts_report.json`, start the localhost-only dashboard
server instead:

```bash
python -m state_mcts.dashboard_server
# Open: http://127.0.0.1:8000/
```

Edit a name under **Editable run names** and press **Save name** (or Enter).
The server updates only the run's optional `display_name`; its stable `run_id`
is preserved. The JSON update is atomic and the dashboard is regenerated
automatically. An empty display name restores the generated fallback label.
Stop the server with `Ctrl+C`.

The **Folder organization** section creates, renames, deletes, and reorders
folders. Assign each run with the folder selector beside its editable name.
Folder definitions and assignments are saved in `dashboard_folders` and
`dashboard_run_folders`. Deleting a folder moves its runs to **Unfiled**. Each
folder heading is collapsible in the editable-name, mean-comparison, reward,
and full-configuration sections, keeping large experiment histories compact.
Each panel's open/closed state is retained in browser storage, so saving a name,
moving runs, or reordering folders does not collapse the panels you were using.
Runs can be selected with the checkboxes shown in editable-name rows and reward
tiles. The **Bulk run actions** bar mirrors that selection across both views and
can move all selected runs into one folder or back to **Unfiled**. **Select all**
and **Clear** provide quick selection controls; selection itself is temporary,
while the resulting folder assignments are persisted to the report.

Runs are sorted alphabetically by editable display name inside each folder.
Unnamed runs use their model/evaluation label and stable run ID as fallbacks.
The same alphabetical order is used for editable names, mean-reward
comparisons, reward graphs, and full configurations. A new run has no folder
assignment, so it appears alphabetically under **Unfiled** until you organize
it. Dashboard names and folder metadata are preserved if an experiment writes
to the report while the server is open.

Use `python -m state_mcts.dashboard_server --no-open-browser` when you do not want the
browser window to open automatically.

## Tests

```bash
python -m unittest -v state_mcts.tests.test_experiment state_mcts.tests.test_dashboard
```

The tests verify the exact simulator against actual Gym transitions, verify
that each toggle replaces only its intended interface, exercise all three
independent trainers and checkpoint loaders, and check a controlled MCTS
problem where longer survival must be selected.

## Interpreting codependence

The training procedures are intentionally independent. Search efficacy is not:

- Dynamics errors compound with tree depth and change the states seen by both
  policy and value networks.
- A policy prior can reduce the number of simulations needed, but a bad prior
  can prevent useful branches from receiving enough visits.
- A value model removes expensive rollouts, but its error is trusted at every
  leaf. A nearly constant value is less informative than random rollouts.
- Policy and value models trained on real states face distribution shift when
  learned dynamics generates imperfect states.

For that reason, validation loss alone is insufficient. The eight-way
ablation is the actual system-level test; single-model runs identify which
interface causes a failure.

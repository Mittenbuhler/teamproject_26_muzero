# Commands: setup, test, evaluate and train

[Verification results and environment](docs/VERIFICATION.md)

All commands below run from **`teamproject_26_muzero/sprint4`**, unless a different
starting directory is explicitly named. Use `python -m package.module`; running
package files directly bypasses their relative imports.

## 1. Environment

From the repository root:

```bash
cd sprint4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows use `py -3 -m venv .venv` and `.venv\Scripts\Activate.ps1` instead.
An existing compatible environment is also fine (`conda activate state-mcts`
was used for the original runs). The environment must match the machine's CPU
architecture; an Intel-only Conda Python may not execute on an ARM Mac.
Use the sprint4 dependency list, not the historical repository-root list.
For the verified Python 3.13 environment, optionally constrain installation with
`python -m pip install -r requirements.txt -c requirements-verified.txt`.
The recorded training reports used PyTorch 2.2.2, NumPy 1.26.4 and Gymnasium
1.3.0; the current verification environment is listed in
[requirements-verified.txt](requirements-verified.txt). Reproduction across
library versions/platforms need not give bit-identical trajectories.

## 2. Check model files and run tests

```bash
# Requires only Python's standard library; checks portable inference copies.
python -m scripts.check_checkpoints --verify

# Also check original training files and historical checkpoints, if available.
python -m scripts.check_checkpoints --originals --verify

# Complete current regression suite.
python -m unittest -v cartpole_muzero.tests.test_pipeline minatar_muzero.tests.test_pipeline state_mcts.tests.test_experiment state_mcts.tests.test_dashboard
```

The originals check intentionally fails in a clone without the large training
files, or if an original is cloud-backed and has no verified hash.
See [checkpoint locations and recovery](checkpoints/README.md). Tests
exercise small temporary training runs; they do not retrain the saved models.

## 3. Evaluate existing trained models (no training)

One fresh-seed episode per evaluation method:

```bash
python -m cartpole_muzero.diagnose checkpoints/inference/cartpole_state4d.pt --episodes 1 --seed 20000 --output-json artifacts/verification/cartpole.json
python -m minatar_muzero.diagnose --checkpoint checkpoints/inference/minatar_breakout.pt --game breakout --episodes 1 --seed 20000 --simulations 25 --output-json artifacts/verification/breakout.json
```

For a fuller evaluation, replace `--episodes 1` with `--episodes 20`.
CartPole inherits 50 simulations from its saved configuration; Breakout's saved
run used 25, supplied explicitly above. Reports here are fresh diagnostics,
separate from the curated results. Repeating a command replaces its diagnostic
JSON; change the output filename to keep both.

For embedded-replay diagnostics, substitute these **original checkpoint paths**:

```bash
python -m cartpole_muzero.diagnose checkpoints/cartpole_muzero/state4d_seed0_R9ncWC/run.pt --episodes 20 --seed 20000 --output-json artifacts/verification/cartpole_with_replay.json
python -m minatar_muzero.diagnose --checkpoint checkpoints/minatar_muzero/breakout_screenshots/best.pt --game breakout --episodes 20 --seed 20000 --simulations 25 --output-json artifacts/verification/breakout_with_replay.json
```

CartPole compares random actions, raw policy and MCTS. MinAtar's diagnostic
compares raw policy and MCTS; the result collector also supplies a random
baseline. Without replay, policy/search diagnostics use fallback roots and
omit replay-unroll errors. Episode evaluation still uses real environments.
For exact published result reproduction, use the collector in section 5: its
seed/reset protocol is recorded in each result folder, including the MinAtar
sticky-action reset caveat.

## 4. Record gameplay

```bash
python -m cartpole_muzero.diagnose checkpoints/inference/cartpole_state4d.pt --episodes 1 --seed 20006 --record-gif artifacts/verification/cartpole_mcts.gif
python -m minatar_muzero.diagnose --checkpoint checkpoints/inference/minatar_breakout.pt --game breakout --episodes 1 --seed 20000 --simulations 25 --record-gif artifacts/verification/breakout_mcts.gif
```

Add `--gif-mode raw` to record the policy without search. CartPole's GIF displays
its four numeric inputs; MinAtar's diagnostic displays the screenshot history.
The full render dependencies are included in `requirements.txt`.

## 5. Rebuild the curated reports

```bash
# Uses only existing result JSON; replaces the curated figures and result README.
python -m scripts.collect_muzero_results --model all --plots-only

# Re-evaluates original checkpoints, records GIFs and replaces curated reports.
# Requires the original best and latest files; may take several minutes.
python -m scripts.collect_muzero_results --model cartpole
python -m scripts.collect_muzero_results --model breakout
```

The collector checks checkpoint hashes before/after and makes no optimizer
updates. Preserve custom edits to generated result READMEs before regenerating.

## 6. Short end-to-end training checks

These create new models, not new evidence about the saved agents. The `mktemp`
commands below create unused directories on macOS/Linux. On Windows create a
new unique directory and use its path as the checkpoint destination.

```bash
CART_RUN=$(mktemp -d checkpoints/cartpole_muzero/smoke_XXXXXX)
python -m cartpole_muzero.train --episodes 2 --max-steps 20 --simulations 3 --warmup-episodes 1 --updates-per-episode 1 --batch-size 4 --unroll-steps 2 --evaluation-episodes 1 --checkpoint-path "$CART_RUN/run.pt"

MINATAR_RUN=$(mktemp -d checkpoints/minatar_muzero/smoke_XXXXXX)
python -m minatar_muzero.train --game breakout --episodes 2 --max-steps 20 --simulations 3 --latent-channels 8 --warmup-episodes 1 --batch-size 4 --updates-per-episode 1 --unroll-steps 2 --evaluation-episodes 1 --checkpoint-path "$MINATAR_RUN/best.pt" --reward-plot-path "$MINATAR_RUN/rewards.png"
```

## 7. Longer experiments and resume

New training with the saved runs' main settings, each in a fresh directory:

```bash
CART_RUN=$(mktemp -d checkpoints/cartpole_muzero/experiment_XXXXXX)
python -m cartpole_muzero.train --episodes 800 --max-steps 500 --simulations 50 --latent-dim 64 --hidden-dim 128 --batch-size 64 --buffer-capacity 50000 --warmup-episodes 20 --updates-per-transition 0.25 --min-updates-per-episode 1 --max-updates-per-episode 50 --learning-rate 1e-4 --discount 0.99 --reward-scale 0.1 --unroll-steps 5 --evaluation-interval 20 --evaluation-episodes 20 --seed 0 --checkpoint-path "$CART_RUN/run.pt"

MINATAR_RUN=$(mktemp -d checkpoints/minatar_muzero/experiment_XXXXXX)
python -m minatar_muzero.train --game breakout --episodes 1500 --image-size 32 --history-length 4 --latent-channels 32 --simulations 25 --batch-size 32 --updates-per-transition 0.25 --min-updates-per-episode 5 --max-updates-per-episode 100 --seed 0 --checkpoint-path "$MINATAR_RUN/best.pt" --reward-plot-path "$MINATAR_RUN/rewards.png"
```

To resume **your new experiment**, repeat its exact command with the same
checkpoint path, add `--reuse-checkpoint`, and change `--episodes` to the number
of **additional** episodes. The loader resolves latest, then final, then best.
Keep architecture, input, environment/action set, discount, reward scale and
other training settings consistent. Do not resume an inference copy.

To continue an original saved run while preserving it, first copy its full run
directory to a new directory and pass the copy's best-checkpoint path. Reuse
the settings in its `training_history.json` under `artifacts/.../results/`;
embedded old `legacy_muzero`/`muzero` path strings are historical metadata.
Resuming writes companion files in the chosen destination.

## 8. Modular MCTS testing version

Exact-physics baseline (no learned model training):

```bash
python -m state_mcts.run_state_mcts_experiment --models none --simulations 24 --search-depth 30 --eval-episodes 2 --report-path artifacts/verification/state_mcts_baseline.json
```

Train components once and test all eight combinations in a fresh directory:

```bash
STATE_RUN=$(mktemp -d checkpoints/state_mcts/experiment_XXXXXX)
python -m state_mcts.run_state_mcts_experiment --models all --ablation --train-samples 1000 --train-epochs 5 --simulations 24 --eval-episodes 2 --checkpoint-dir "$STATE_RUN" --loss-dir artifacts/verification/state_mcts_losses --report-path artifacts/verification/state_mcts_ablation.json
```

Use `--models dynamics`, `policy`, `value`, or `policy,value` to isolate components.
For evaluation of existing complete compatible files, add `--reuse-checkpoints`
and supply their directory; keep `--search-depth` and `--max-steps` consistent
with saved dynamics/value targets. **Missing component files cause training even
with this flag.** The default original files are in `checkpoints/state_mcts/`.

Teacher and value-alignment diagnostics (training writes only to new test paths):

```bash
python -m state_mcts.train_policy_value_from_mcts --models policy,value --policy-checkpoint-path checkpoints/state_mcts/teacher_test/state_policy.pt --value-checkpoint-path checkpoints/state_mcts/teacher_test/state_value.pt --report-path artifacts/verification/teacher_report.json --loss-dir artifacts/verification/teacher_losses
python -m state_mcts.diagnose_value_action_alignment --help
python -m state_mcts.dashboard
python -m state_mcts.dashboard_server
```

The static dashboard writes `artifacts/state_mcts/diagnostics_dashboard.html`.
The server opens an editable dashboard; stop it with Ctrl+C. See the
[testbed README](state_mcts/README.md) for target semantics and specialist flags.

## 9. Export/check portable models and inspect options

The curated inference files already exist. To recreate separate copies from
original checkpoints (read-only sources, destinations must not already exist):

```bash
python -m scripts.export_inference_checkpoints --output-dir artifacts/verification/inference_copy
python -m scripts.check_checkpoints --verify
python -m cartpole_muzero.train --help
python -m cartpole_muzero.diagnose --help
python -m minatar_muzero.train --help
python -m minatar_muzero.diagnose --help
python -m state_mcts.run_state_mcts_experiment --help
python -m state_mcts.train_policy_value_from_mcts --help
python -m state_mcts.diagnose_value_action_alignment --help
```

Exports include source hashes and verify equality of every network tensor.
They omit replay, optimizer and RNG state. Existing sources are never rewritten.

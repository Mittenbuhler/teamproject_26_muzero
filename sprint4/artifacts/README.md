# Experiment results

Start here for the MuZero analysis, training histories, performance comparisons
and gameplay recordings. Each model's results live beside its other artifacts.
The checkpoint directories contain the trained weights and original training
outputs; the folders below contain reproducible analysis of those saved runs.

| Model | Analysis and statistics | Combined PNG | Six-second gameplay |
| --- | --- | --- | --- |
| CartPole MuZero | [Results](cartpole_muzero/results/state4d_seed0_R9ncWC/README.md) | [History and comparison](cartpole_muzero/results/state4d_seed0_R9ncWC/overview.png) | [GIF](cartpole_muzero/results/state4d_seed0_R9ncWC/gameplay_6s.gif) |
| MinAtar Breakout MuZero | [Results](minatar_muzero/results/breakout_screenshots/README.md) | [History and comparison](minatar_muzero/results/breakout_screenshots/overview.png) | [GIF](minatar_muzero/results/breakout_screenshots/gameplay_6s.gif) |

## Contents of each results folder

- `README.md`: statistics, interpretation, evaluation setup and reproduction.
- `overview.png`: training history and performance comparison, 3200 x 1800.
- `training_history.png` and `.svg`: the learning curve by itself.
- `performance_comparison.png` and `.svg`: random, policy-only and MCTS scores.
- `evaluation.json` and `episode_scores.csv`: summary statistics and every seed's score.
- `training_history.json`, `training_scores.csv` and `periodic_evaluations.csv`: numeric training data extracted from the saved checkpoint.
- `gameplay_6s.gif` and `gameplay.json`: gameplay plus timing and episode-selection details.

The comparison evaluates the best checkpoint on 20 fresh environment seeds.
Training histories cover the full run, so the final history checkpoint can
differ from the best evaluation checkpoint. Each report identifies both files.
These are single-run results, and the games' reward scales differ.

## Recreate the results

From sprint4 in the project environment:

```bash
python -m scripts.collect_muzero_results
python -m scripts.collect_muzero_results --model cartpole
python -m scripts.collect_muzero_results --model breakout --plots-only
```

Collection performs no optimizer updates and checks that the source checkpoints
remain unchanged. Figures and source data in these curated results directories
are available to Git; large trained checkpoints retain their existing ignore rules.

## Testbed and outdated data

[Modular MCTS reports](state_mcts/README.md) are component experiments.
[Older CartPole plots](cartpole_muzero/training_progress/README.md) are explicitly
outdated. Both stay at their existing paths; see [the historical index](../docs/OUTDATED.md).

For environment setup and all workflows use [COMMANDS.md](../COMMANDS.md).

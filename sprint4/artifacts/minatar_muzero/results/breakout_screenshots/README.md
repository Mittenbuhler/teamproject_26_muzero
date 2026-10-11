# MuZero on MinAtar Breakout: saved-run results

[Combined figure](overview.png) | [Training history](training_history.png) | [Performance comparison](performance_comparison.png) | [Six-second gameplay GIF](gameplay_6s.gif)

Checkpoint: `checkpoints/minatar_muzero/breakout_screenshots/best.pt`, selected at episode 990.

Evaluation uses 20 fresh seeds (20000-20019), 25 search simulations per move and no root exploration noise. Standard deviation is across environment seeds.

| Method | Mean reward | Sample SD | Median | Range |
| --- | ---: | ---: | ---: | --- |
| Random actions | 0.25 | 0.44 | 0.00 | 0-1 |
| Trained policy without search | 2.55 | 1.88 | 2.00 | 0-5 |
| MuZero with search | 4.30 | 1.89 | 5.00 | 0-6 |

Search changes mean reward by **+1.75**. Across matched seeds: 9 wins, 11 ties and 0 losses.

Search improves average Breakout performance for this checkpoint.

## Training history

The full history has 1,500 episodes. Mean actor reward is 0.11 in the first 100 episodes and 2.86 in the last 100. Exploration changes during training.

The best scheduled evaluation is 4.67. The final scheduled evaluation is 3.00. These use fixed seeds and are distinct from the fresh evaluation above.

## Gameplay

The GIF shows seed 20000, score 5, 62 environment steps in exactly six seconds.

Breakout uses the first evaluation seed. Its 62-step episode ends before step 200, so the full episode is uniformly retimed.

## Data and reproduction

- [Evaluation statistics and checkpoint identity](evaluation.json)
- [Per-seed scores](episode_scores.csv)
- [Full training history](training_history.json)
- [Training rewards](training_scores.csv)
- [Periodic evaluations](periodic_evaluations.csv)
- [GIF timing and selection](gameplay.json)

Recreate with `python -m scripts.collect_muzero_results --model breakout` from sprint4. Use `--plots-only` to redraw existing results without running evaluation.

The collector makes no optimizer updates and verifies checkpoint bytes are unchanged. These results concern one training run. Reward scales differ between environments.

## Reset protocol

This comparison reproduces the original procedure: one environment per method, with seeds 20000-20019 evaluated in order. The installed MinAtar version retains its previous sticky action across resets, so environment lifecycle is part of the protocol.

A [separate check with a newly constructed environment per seed](reset_protocol_check.json) gives policy-only mean 2.85 and MCTS mean 4.30. The policy score differs on seed 20018. These are different reset procedures, not different model weights.

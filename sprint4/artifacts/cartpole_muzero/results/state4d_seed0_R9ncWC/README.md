# MuZero on CartPole: saved-run results

[Combined figure](overview.png) | [Training history](training_history.png) | [Performance comparison](performance_comparison.png) | [Six-second gameplay GIF](gameplay_6s.gif)

Checkpoint: `checkpoints/cartpole_muzero/state4d_seed0_R9ncWC/run.pt`, selected at episode 800.

Evaluation uses 20 fresh seeds (20000-20019), 50 search simulations per move and no root exploration noise. Standard deviation is across environment seeds.

| Method | Mean reward | Sample SD | Median | Range | Reached 500 |
| --- | ---: | ---: | ---: | --- | ---: |
| Random actions | 22.00 | 14.79 | 17.00 | 9-62 | 0/20 |
| Trained policy without search | 406.25 | 76.97 | 374.50 | 287-500 | 7/20 |
| MuZero with search | 406.55 | 77.02 | 377.00 | 287-500 | 7/20 |

Search changes mean reward by **+0.30**. Across matched seeds: 5 wins, 11 ties and 4 losses.

Additional search has almost no effect on this CartPole evaluation.

## Training history

The full history has 800 episodes. Mean actor reward is 115.87 in the first 100 episodes and 438.44 in the last 100. Exploration changes during training.

The best scheduled evaluation is 484.30. The final scheduled evaluation is 484.30. These use fixed seeds and are distinct from the fresh evaluation above.

## Gameplay

The GIF shows seed 20006, score 500, 500 environment steps in exactly six seconds.

CartPole uses a successful example. The first three seconds show steps 0-200, and the final three show steps 201-500.

## Data and reproduction

- [Evaluation statistics and checkpoint identity](evaluation.json)
- [Per-seed scores](episode_scores.csv)
- [Full training history](training_history.json)
- [Training rewards](training_scores.csv)
- [Periodic evaluations](periodic_evaluations.csv)
- [GIF timing and selection](gameplay.json)

Recreate with `python -m scripts.collect_muzero_results --model cartpole` from sprint4. Use `--plots-only` to redraw existing results without running evaluation.

The collector makes no optimizer updates and verifies checkpoint bytes are unchanged. These results concern one training run. Reward scales differ between environments.

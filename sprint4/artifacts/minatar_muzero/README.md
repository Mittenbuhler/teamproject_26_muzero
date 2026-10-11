# Screenshot MuZero artifacts

The saved Breakout run's analysis is in
[results/breakout_screenshots](results/breakout_screenshots/README.md), including
training history, performance comparisons, raw statistics, a combined PNG and
the six-second gameplay GIF.

Outputs from `minatar_muzero/` belong here; state-MCTS and historical artifacts stay
in their own directories. The current model consumes RGB screenshots from
MinAtar's renderer, not native feature planes.

```text
artifacts/minatar_muzero/
├── training/<game-or-run>/rewards.png
├── diagnostics/<game-or-run>/<report>.json
└── gifs/<game-or-run>/<game>_<raw-or-mcts>_seed<seed>.gif
```

The saved run’s available training figure is
[results/breakout_screenshots/training_history.png](results/breakout_screenshots/training_history.png).
The older `training/breakout_screenshots/rewards.png` path is not present in this
checkout. Full numeric history is also in the result JSON; full history and replay
remain embedded in the original checkpoints. Use `minatar_muzero.diagnose` to write fresh
evaluation reports into a separate diagnostics directory.

Training plots distinguish actor reward, moving average, and periodic fixed-seed
evaluation. Diagnostic reports should identify the exact checkpoint, screenshot
preprocessing, game/action variant, history, search budget, and all evaluation
seeds. Keep per-episode scores when comparing agents. One GIF or the best score
across many evaluations does not establish general performance.

Use distinct run names when changing the observation format, history, game,
seed, or training budget. The binaries are ignored by Git; preserve the raw
reports and checkpoints separately before any cleanup.

# Native MinAtar diagnostic GIFs

Store recordings by game:

```text
artifacts/muzero/gifs/<game>/<game>_<raw-or-mcts>_seed<seed>.gif
```

For example:

```text
artifacts/muzero/gifs/breakout/breakout_mcts_seed42.gif
artifacts/muzero/gifs/breakout/breakout_raw_seed42.gif
```

Each recording is synchronized at the decision boundary:

- the environment pane shows the real MinAtar state being played;
- the input pane shows the exact native channel planes passed to the representation network for that action;
- a history longer than one shows all time slots in oldest-to-newest order, including initial zero padding;
- the last image shows the terminal-updated observation beside the final environment frame.

The native-channel panel is not a grayscale screenshot, reconstruction, or visualization of the latent dynamics. It exposes the semantic binary/normalized planes in their actual configured order. Because meanings differ by game, retain channel labels or ordering metadata in the recording/diagnostic whenever the adapter provides them.

Create a Breakout recording from `teamproject_26_muzero/sprint4`:

```bash
../../.venv/bin/python -m muzero.diagnose \
  --checkpoint checkpoints/muzero/breakout/best.pt \
  --game breakout \
  --record-gif artifacts/muzero/gifs/breakout/breakout_mcts_seed42.gif
```

Use the same seed and checkpoint for raw-policy versus MCTS comparisons. MCTS recordings should use evaluation behavior: no Dirichlet root noise and the most-visited action. Check `../../.venv/bin/python -m muzero.diagnose --help` for supported mode, seed, frame-rate, and size switches.

GIF binaries should not be treated as training data or proof of solved performance. Report their seed together with multi-episode evaluation statistics.


# Legacy MuZero episode GIFs

Generated diagnostic episode recordings belong in this directory. GIF binaries are ignored by Git; this README keeps the intended location and naming convention visible.

Use names of the form:

```text
<checkpoint-name>_<raw-or-mcts>_seed<seed>.gif
```

For example:

```text
cartpole_spatial_v12_mcts_seed42.gif
cartpole_spatial_v12_raw_seed42.gif
```

Recordings contain real environment RGB frames. They are diagnostic visualizations, not decoded latent-state predictions.

New recordings are composite by default:

- The left pane contains the real RGB environment rendering and the policy/value overlay.
- The right pane contains the exact normalized grayscale frame stack supplied to the representation network, vertically ordered from oldest to newest and labeled as padding or by time offset.
- The first decisions visibly include the zero-padded history. The panel then updates before each selected action and once more for the final terminal frame.

The composite panel is automatically enabled whenever `--record-gif` is present. There is no separate option for an RGB-only GIF; omit `--record-gif` when no recording should be created. `--gif-width` controls the left gameplay pane rather than the total composite width.

Run from `teamproject_26_muzero/sprint4`, for example:

```bash
../../.venv/bin/python -m legacy_muzero.diagnose_latent_muzero \
  checkpoints/legacy_muzero/cartpole_policy_v14.pt \
  --episodes 1 \
  --simulations 50 \
  --record-gif artifacts/legacy_muzero/gifs/cartpole_policy_v14_mcts_seed4200.gif \
  --gif-mode mcts \
  --gif-seed 4200
```

The resulting file is:

```text
artifacts/legacy_muzero/gifs/cartpole_policy_v14_mcts_seed4200.gif
```

Existing single-pane GIF files are not changed retroactively; rerun their diagnostic command to create them with the current composite layout.

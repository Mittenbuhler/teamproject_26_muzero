# Native MinAtar MuZero checkpoints

Native checkpoints are isolated by game and must not be mixed with `checkpoints/legacy_muzero` or `checkpoints/state_mcts`.

## Layout and roles

```text
checkpoints/muzero/<game>/
├── best.pt
├── latest.pt
└── final.pt
```

- `best.pt` is the checkpoint with the best periodic fixed-seed, noise-free MCTS evaluation.
- `latest.pt` is the resumable actor/learner state and should contain networks, optimizer, replay, histories, completed-episode count, configuration, and random-number states.
- `final.pt` records the end of the most recent training invocation even when it is worse than `best.pt`.

Related plots and recordings belong under `artifacts/muzero/`, never beside the model archives.

## Required metadata

A native checkpoint needs enough metadata to reconstruct and validate all networks:

- canonical game name and concrete environment ID/version;
- native channel count, height, width, channel order, and `history_length`;
- discrete action count and whether the environment uses the v1 minimal or v0 full action set;
- latent shape and network-head configuration;
- discount, reward scaling, unroll/bootstrap lengths, and search configuration;
- checkpoint/schema version and replay policy-target semantics.

Loading must fail clearly before applying weights when any structural or semantic field is incompatible. Matching tensor sizes alone are insufficient: two games can have channels or actions with different meanings.

## Fresh runs and resume

Create a Breakout run from `teamproject_26_muzero/sprint4`:

```bash
../../.venv/bin/python -m muzero.train --game breakout
```

The trainer should place its outputs in `checkpoints/muzero/breakout/` by default. Use the explicit reuse/resume option exposed by `../../.venv/bin/python -m muzero.train --help` to continue `latest.pt`; do not resume from `best.pt` if optimizer and replay continuity matter.

```bash
../../.venv/bin/python -m muzero.train --game breakout \
  --episodes 500 --reuse-checkpoint
```

Native checkpoints use schema version 1 and the format marker
`native_minatar_vanilla_muzero`. A legacy screenshot checkpoint is rejected
before any weights are applied, even if it happens to contain version number 1.

Always start fresh when changing any of the following:

- game or native channel semantics;
- v1 minimal versus v0 full action set;
- native observation dimensions or channel order;
- history length or latent/network architecture;
- a target/reward transformation that changes stored replay units.

Legacy CartPole screenshot checkpoints are intentionally incompatible. Cross-game weight transfer and partial head loading are not supported. Checkpoints should be written atomically so interruption cannot leave a file that appears valid but is only partially serialized.

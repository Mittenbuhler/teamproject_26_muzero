# Native MuZero artifacts

This tree is reserved for outputs from the native MinAtar package. Keep it separate from `artifacts/legacy_muzero` and `artifacts/state_mcts`: those implementations use different observations, model semantics, and checkpoints.

## Layout

```text
artifacts/muzero/
├── README.md
├── training/
│   └── <game>/
│       └── rewards.png
├── diagnostics/
│   └── <game>/
│       └── <diagnostic outputs>
└── gifs/
    ├── README.md
    └── <game>/
        └── <game>_<raw-or-mcts>_seed<seed>.gif
```

Use lowercase canonical game names such as `breakout`, `asterix`, `freeway`, `seaquest`, and `space_invaders`. Do not mix v0 full-action and v1 minimal-action results in one unnamed experiment; include the environment variant in an additional filename or subdirectory whenever v0 support is explicitly used.

Training plots should distinguish raw episodic reward, a documented moving average, and fixed-seed evaluation reward. Diagnostic JSON should record the game/environment ID, native observation shape and channel order, history length, action count, checkpoint identity, search settings, and seeds needed to reproduce it.

Generated binaries may be ignored by Git, but the directory convention and small explanatory metadata should remain versioned. A high reward from one seed or a visually convincing recording is not a substitute for aggregate evaluation.


# Sprint 4 layout

Sprint 4 contains two deliberately separate implementations. Run all commands
below from this directory so Python resolves the intended package.

```text
sprint4/
├── state_mcts/          real four-value CartPole state experiment
│   ├── run_state_mcts_experiment.py    training, checkpointing, ablations, evaluation
│   ├── mcts_search.py        modular state-space MCTS
│   ├── train_policy_value_dynamic_from_data.py      independent dynamics, policy, and value trainers
│   ├── models.py        state-only network definitions
│   ├── dashboard.py     state report dashboard generator
│   ├── dashboard_server.py
│   ├── tests/
│   └── README.md        full experiment documentation
├── legacy_muzero/       earlier screenshot/latent MuZero implementation
│   ├── mcts.py
│   ├── models.py
│   ├── train_policy_value.py
│   ├── diagnostics and demo scripts
│   └── tests/
├── artifacts/
│   ├── state_mcts/
│   └── legacy_muzero/
└── checkpoints/
    ├── state_mcts/
    └── legacy_muzero/
```

The two packages do not import one another. In particular, `state_mcts` owns
its MCTS and network definitions; it cannot accidentally import the earlier
latent `mcts.py` or `models.py`.

## Real-state MCTS

```bash
python -m state_mcts.run_state_mcts_experiment --models none
python -m state_mcts.run_state_mcts_experiment --models all --ablation
python -m state_mcts.dashboard_server
python -m unittest -v state_mcts.tests.test_experiment state_mcts.tests.test_dashboard
```

See [state_mcts/README.md](state_mcts/README.md) for training targets, search
semantics, dashboard usage, and all command options.

## Earlier latent/image implementation

```bash
python -m legacy_muzero.cartpole_demo
python -m legacy_muzero.train_policy_value --help
python -m legacy_muzero.diagnose_latent_muzero --help
python -m unittest -v legacy_muzero.tests.test_pipeline
```

See [legacy_muzero/README.md](legacy_muzero/README.md) for the legacy entry
points. Module-style commands are required; do not run package files directly.


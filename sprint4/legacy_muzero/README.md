# Earlier latent/image MuZero implementation

This package contains the Sprint 4 implementation that observes rendered
CartPole frames, learns a latent representation and dynamics model, and plans
with `ModelBasedMCTS`. It is retained for comparison and diagnostics; it is not
used by the newer `state_mcts` experiment.

Run commands from the `sprint4` directory:

```bash
# End-to-end demo (reuses the default latent checkpoint when available)
python -m legacy_muzero.cartpole_demo

# Training entry points
python -m legacy_muzero.train_dynamics --help
python -m legacy_muzero.train_policy_value --help

# Diagnostics
python -m legacy_muzero.diagnose_latent_muzero --help
python -m legacy_muzero.diagnose_policy_behavior --help
python -m legacy_muzero.diagnose_mcts_failure --help
python -m legacy_muzero.diagnose_search_ablations --help
python -m legacy_muzero.probe_terminal_latent --help

# Regression tests
python -m unittest -v legacy_muzero.tests.test_pipeline
```

Default outputs are isolated under:

```text
artifacts/legacy_muzero/
checkpoints/legacy_muzero/
```

The implementation is intentionally self-contained inside this package. Any
new work on real Gym states belongs in `state_mcts`, not here.


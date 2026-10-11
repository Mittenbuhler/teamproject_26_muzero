# Modular MCTS testing checkpoints

These are separate dynamics, policy and value models for the retained
[component-testing experiment](../../state_mcts/README.md), not the final
CartPole MuZero representation/dynamics/prediction checkpoint.

- `state_dynamics.pt`, `state_policy.pt`, `state_value.pt`: default component files.
- `state_policy_mcts_teacher.pt`, `state_value_mcts_teacher.pt`: teacher variants.
- `20261004_175041/`: retained timestamped component set.

Existing files remain at their original paths. Reuse requires compatible
training horizons: dynamics search depth and value target episode horizon are
checked. Missing files trigger training even with `--reuse-checkpoints`.
Use [central commands](../../COMMANDS.md) and fresh directories for new tests.

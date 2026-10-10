# Native-state MuZero checkpoints (v15)

[Model inventory and portable copies](../README.md) · [Central commands](../../COMMANDS.md)

`cartpole_muzero/` now consumes CartPole's four native state variables and jointly
trains fully connected representation, dynamics, and prediction networks.
See the [current training guide](../../cartpole_muzero/README.md).

New runs use distinct paths such as `state4d_seed0/run.pt`. The default trainer
path is `cartpole_state_v15.pt`. In each run, the base `.pt` is the best
scheduled-evaluation checkpoint, `_latest.pt` is resumable, `_final.pt` is the
last model, and `_rewards.png` contains raw training and evaluation rewards.
A short smoke run checks execution rather than learning quality.

The completed 800-episode run is stored in `state4d_seed0_R9ncWC/`, including
best/latest/final checkpoints, its original training log and reward graph.
Directory renaming did not change the saved checkpoint contents.

## Historical inventory

The inventory checked on 4 October 2026 contains two old files:

- `archive/pre_v8/dynamics_cartpole.pt`
- `archive/pre_v8/policy_value_cartpole.pt`

These take four physical state variables directly and have no representation
network. They are architecturally incompatible with the new v15 latent model.
Their files have not been altered by this conversion.

Screenshot v8-v14 checkpoints named in older documentation are absent in this
checkout. Historical notes mention v11/v12/v13/v14 runs, but their reported
scores cannot be freshly verified without recovering those files. The current
v15 loader rejects all earlier formats; old screenshot diagnostics require
the corresponding historical implementation.

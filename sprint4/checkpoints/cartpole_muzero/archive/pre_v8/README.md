# OUTDATED: pre-v8 CartPole checkpoint format

`dynamics_cartpole.pt` and `policy_value_cartpole.pt` are retained historical
files from the separate state-model approach. They have no learned observation
representation and cannot load with the current CartPole v15 or MinAtar v2
MuZero loaders. Do not rename them to impersonate a current checkpoint.

Use [the model inventory](../../../README.md) for current runnable models and
[the historical index](../../../../docs/OUTDATED.md) for development context.

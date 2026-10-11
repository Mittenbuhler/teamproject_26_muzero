# Portable trained models: inference only

These are compact copies of the saved best CartPole and MinAtar Breakout
checkpoints. All representation, dynamics and prediction weights are unchanged;
`provenance.json` identifies the original files and records tensor equality.

Use the [central evaluation/recording commands](../../COMMANDS.md). Replay,
optimizer and random-number state are omitted, so these files cannot resume
training. Full training history is in the result JSON and original checkpoints.

[Model inventory and recovery paths](../README.md) · [Results](../../artifacts/README.md)

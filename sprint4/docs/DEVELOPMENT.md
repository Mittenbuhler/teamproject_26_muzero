# Development guide and source map

This is the current reading guide for the `MuZero-for-MinAtar` result branch.
Start with the [overview](../README.md), [commands](../COMMANDS.md), and
[saved results](../artifacts/README.md). Older notes record development;
they are not the specification of the final agents.

## Development stages

1. **Learning material and early state-model experiments.** Repository-root
   `tutorials/` and `example_code/` contain exercises. `S4_alphaZero_rebuild/`
   contains earlier CartPole code with separately trained dynamics and
   policy/value networks. These are historical reference material.
2. **Latent CartPole MuZero.** Earlier work added screenshot preprocessing,
   representation learning and joint recurrent training. Git later named that
   package `legacy_muzero`. Screenshot formats v8–v14 are historical.
3. **Modular MCTS as a parallel testing experiment.** `state_mcts` was added to
   isolate effects of dynamics, policy and value. It is not the ancestor of all
   latent MuZero code. Commit `1f4529b` records the split; the earlier latent
   implementation already existed at that point.
4. **MinAtar migration.** The old `muzero` package used native MinAtar feature
   grids (format v1). Today's `minatar_muzero` uses only rendered RGB screenshot
   histories (format v2).
5. **Final result packages.** `cartpole_muzero` now encodes the four CartPole
   values with fully connected networks (v15). `minatar_muzero` uses
   convolutional networks on screenshot history. Both train `h`, `g` and `f`
   jointly and search through learned latent transitions.

The final package split and saved-run reports were already working-tree changes
when this documentation cleanup began. Do not infer a separate Git commit for
every checkpoint version. [Historical notes](history/README.md) preserve earlier
accounts, including claims that no longer describe the code or available files.

## Follow one decision and one training update

Read these files in either main package in this order:

| File | Question it answers |
| --- | --- |
| `environment.py` | What exact observation reaches the agent? |
| `models.py` | How do representation `h`, dynamics `g`, and prediction `f` transform it? |
| `mcts.py` | How do priors, predicted rewards/values, visits and action selection interact? |
| `buffers.py` | What is stored, sampled and padded across recurrent training steps? |
| `train.py` | How do self-play, targets, joint updates and checkpoints fit together? |
| `diagnose.py` | How are random actions, policy-only play and search compared? |
| `tests/test_pipeline.py` | Which input, search, replay and checkpoint contracts are checked? |

The packages own their implementations independently. Similar names do not mean
shared classes. Read both when comparing a specific behavior.

## Representation is the main input distinction

CartPole's `[x, x_dot, theta, theta_dot]` is a fully observed four-value input.
Its encoder maps `[B,4]` to `[B,64]` by default. Rendering is only for GIFs.
`cartpole_muzero/image_observation.py` is an unused historical helper, not part
of the active agent.

MinAtar calls `render()` after reset and every real step. RGB pixels are resized
with nearest-neighbor interpolation, normalized by 255 and stacked oldest
first. Four frames give `[B,12,32,32]`; the default encoder produces
`[B,32,8,8]`. Initial missing frames are zeros. Native boolean feature planes
and `info` fields are never representation inputs. History provides motion
evidence, but does not guarantee a fully Markovian observation.

Both use latent/action dynamics and policy/value prediction below the root.
Neither predicts screenshots as its learning objective. Their parameters,
environments, search budgets and reward scales differ, so cross-environment
scores cannot isolate the effect of an encoder.

## Why retain modular MCTS?

`state_mcts` makes each learned component independently replaceable. Its
baseline uses exact CartPole dynamics, UCT and random rollouts. The dynamics,
policy and value switches test state prediction, policy priors and leaf
estimation separately or in all eight combinations. Default policy/value
supervision uses a stabilizing heuristic; a separate MCTS-teacher route exists.

This testbed helps inspect model error and interactions in a simpler setting.
It has no learned representation and is not the final MuZero agent. Its scores
and historical plots remain component experiments.
See [its detailed guide](../state_mcts/README.md).

## Read results together with provenance

Current result folders contain checkpoint identities, all seed scores, training
history and recording protocols. `scripts/collect_muzero_results.py` rebuilds
these curated reports. The best-checkpoint evaluation and full-run history may
come from different files.

Use [the checkpoint inventory](../checkpoints/README.md) to distinguish portable
inference copies, original resumable state and incompatible pre-v8 archives.
Inference exports retain all three network state dictionaries and model
metadata, with a source hash, and omit training state. They are the same trained
models. Embedded original paths are provenance, not a reason to move or rewrite
old checkpoints.

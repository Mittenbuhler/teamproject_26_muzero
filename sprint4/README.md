# MuZero on CartPole and MinAtar

The final project has two MuZero agents: **CartPole from four physical state
values** and **MinAtar Breakout from RGB screenshot histories**. Both learn a
representation, dynamics, reward, policy and value together, then use MCTS to
plan in the learned latent state. The modular `state_mcts` experiment remains
available for testing individual planning components.

**Start here:** [all commands](COMMANDS.md) · [saved results](artifacts/README.md)
· [trained models](checkpoints/README.md) · [development and code map](docs/DEVELOPMENT.md)
· [outdated material](docs/OUTDATED.md)

## What differs between the implementations?

| | CartPole MuZero | MinAtar MuZero | Modular MCTS testbed |
| --- | --- | --- | --- |
| Package | [`cartpole_muzero`](cartpole_muzero/README.md) | [`minatar_muzero`](minatar_muzero/README.md) | [`state_mcts`](state_mcts/README.md) |
| Real input | `[position, velocity, pole angle, angular velocity]` | Four 32×32 RGB rendered frames, oldest first | The same four CartPole state values |
| Input to representation `h` | `[batch, 4]`; no frame history | `[batch, 12, 32, 32]`; zero padding at episode start | No learned representation |
| Encoder / latent | Fully connected / 64-value vector by default | Convolutional / 32×8×8 spatial latent by default | Physical four-value state |
| Search transitions | Learned latent dynamics `g` | Learned latent dynamics `g` | Exact physics or an independently trained state predictor |
| Purpose | Main state-input result | Main visual-input result | Controlled component tests and ablations |

CartPole already supplies velocities. MinAtar's screenshot history supplies
visual motion cues. The MinAtar adapter **discards native semantic feature
grids**; the CartPole agent **does not use rendered pixels**. These inputs lead
to different encoders and latent shapes, while both agents use the same core
MuZero learning pattern. They are separate implementations, not interchangeable
checkpoints or an isolated experiment changing only representation.

```text
real observation -> h -> latent -> f -> policy, value
                          |
                 latent + action -> g -> next latent, reward
                                            |
                                            f -> policy, value
```

Only the root receives a real observation. Hypothetical MuZero search steps use
`g` and `f`, without querying the environment. MCTS visits supervise policy;
real rewards and bootstrapped returns supervise reward and value.

## Try the trained agents

Run from this `sprint4` directory with the environment in [COMMANDS.md](COMMANDS.md).
Small inference checkpoints retain the original trained network weights and are
included in the shareable project files. Large original checkpoints remain at
their existing paths for resume and replay diagnostics.

```bash
python -m scripts.check_checkpoints
python -m cartpole_muzero.diagnose checkpoints/inference/cartpole_state4d.pt --episodes 1 --seed 20000 --output-json artifacts/verification/cartpole.json
python -m minatar_muzero.diagnose --checkpoint checkpoints/inference/minatar_breakout.pt --game breakout --episodes 1 --seed 20000 --simulations 25 --output-json artifacts/verification/breakout.json
```

These commands evaluate without training. Inference copies omit replay,
optimizer and random-number state; use original checkpoints for resumed
training or diagnostics that need replay.

## Saved results

Existing reports evaluate one trained run per environment on 20 fresh seeds
(20000–20019). These are mean raw rewards; the games have different reward scales.

| Saved run | Episodes trained | Random | Policy without search | MuZero with search | Report |
| --- | ---: | ---: | ---: | ---: | --- |
| CartPole | 800 | 22.00 | 406.25 | 406.55 | [Results and gameplay](artifacts/cartpole_muzero/results/state4d_seed0_R9ncWC/README.md) |
| MinAtar Breakout | 1,500 | 0.25 | 2.55 | 4.30 | [Results and gameplay](artifacts/minatar_muzero/results/breakout_screenshots/README.md) |

CartPole used 50 simulations per move; Breakout used 25. Search made little
difference for this CartPole checkpoint and improved this Breakout checkpoint.
These are single-training-run observations. The Breakout report also records
the reset/sticky-action protocol and a separate fresh-environment check.

## Where to look

```text
sprint4/
├── cartpole_muzero/      main result: four-value input, vector latent
├── minatar_muzero/       main result: RGB frame history, spatial latent
├── state_mcts/           retained modular component-testing experiment
├── checkpoints/         inventory, inference copies, original runs
├── artifacts/           results, figures, recordings, historical data
├── scripts/             checkpoint checking/export and result collection
├── docs/                development map and explicitly outdated notes
├── COMMANDS.md          setup, tests, evaluation, recording and training
└── requirements.txt     dependencies for the current implementations
```

For a human or an LLM reading the project, follow
[the development guide](docs/DEVELOPMENT.md) before interpreting older folders.
Old data is retained in place with status labels so existing paths stay valid.

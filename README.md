# MuZero team project: final results

The main result branch is `MuZero-for-MinAtar`. Start in **[sprint4](sprint4/README.md)**:
CartPole learns from four physical state values; MinAtar Breakout learns from a
history of rendered RGB screenshots. Both jointly train representation,
dynamics and prediction networks and use latent-state MCTS.

- [Main overview and implementation comparison](sprint4/README.md)
- [All commands: setup, tests, trained-model evaluation and training](sprint4/COMMANDS.md)
- [Results, figures and gameplay](sprint4/artifacts/README.md)
- [Model inventory and portable trained weights](sprint4/checkpoints/README.md)
- [Development story and source-code reading order](sprint4/docs/DEVELOPMENT.md)
- [Outdated material and preserved historical paths](sprint4/docs/OUTDATED.md)

## Repository map

| Directory | Role |
| --- | --- |
| `sprint4/cartpole_muzero/` | **MAIN RESULT**: four-value input, fully connected representation, vector latent |
| `sprint4/minatar_muzero/` | **MAIN RESULT**: RGB screenshot histories, convolutional representation, spatial latent |
| `sprint4/state_mcts/` | **TESTING VERSION**: independently replaceable dynamics, policy and value components |
| `S4_alphaZero_rebuild/` | **OUTDATED implementation**, preserved as development history |
| `example_code/`, `tutorials/` | **HISTORICAL learning material**, with existing notebooks/data retained |
| `slides/` | **HISTORICAL presentations**, representing different project stages |

For a human or an LLM exploring the repository, use the current overview and
development guide first. Earlier READMEs are labeled where their instructions
or results are outdated. Use `sprint4/requirements.txt` for the final project;
the repository-root requirements belong to earlier exercises.

Original training checkpoints remain at their existing paths. Small inference
copies let readers run the trained agents without the multi-GB replay buffers.
See the inventory for which originals need a local copy or team backup.

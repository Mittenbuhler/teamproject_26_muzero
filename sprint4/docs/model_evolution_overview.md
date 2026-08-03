# From `state_mcts` to legacy MuZero to native MuZero

Status reviewed on 3 August 2026.

## Executive summary

The repository does not contain a literal `state_mcts -> legacy_muzero`
conversion. The chronology is the reverse: the screenshot/latent MuZero code
already existed when `state_mcts` was introduced. Commit `1f4529b` then split
the two implementations into parallel packages and described them as:

- `state_mcts`: a modular, real-state CartPole MCTS testbed;
- `legacy_muzero`: the earlier screenshot/latent MuZero implementation.

The actual successor relationship is `legacy_muzero -> muzero`. The new
`muzero` package keeps the legacy model's small vanilla MuZero algorithm, but
replaces rendered CartPole screenshots with native MinAtar feature grids and
adds strict game-specific configuration, checkpointing, and diagnostics.

```text
Earlier MCTS / policy-value / dynamics experiments
                         |
                         +-- image/latent CartPole MuZero (29 June)
                         |          |
                         |          +-- legacy_muzero (named on 6 July)
                         |                    |
                         |                    +-- native MinAtar muzero (26 July)
                         |
                         +-- modular real-state state_mcts (6 July)
                                    (parallel diagnostic testbed)
```

## Evidence-based timeline

| Date | Commit or working-tree evidence | Change |
| --- | --- | --- |
| 27 June | `ce92a16`, `New MuZero setup` | Added the first Sprint 4 MuZero artifacts and checkpoints. |
| 29 June | `9836b81`, `add rebuild with internal states + new model with complete learning` | Added the source for the screenshot-based latent model: image preprocessing, representation, learned dynamics, policy/value prediction, replay, MCTS, and joint training. |
| 6 July | `d775dab`, `initialise modular MCTS for rigorous testing` | Added the separate real-state CartPole MCTS, independent dynamics/policy/value trainers, ablation runner, tests, diagnostics, and dashboard. |
| 6 July | `1f4529b`, `split between state and legacy MCTS` | Moved the real-state experiment into `state_mcts/` and the already-existing latent implementation into `legacy_muzero/`. The packages were deliberately isolated and did not import one another. |
| 13 July | `b2e2bcb`, `updated value NN` | Changed the state-MCTS value leaf to a direct clipped state-quality estimate, added `--bootstrap-after`, MCTS-teacher policy/value distillation, and a value-versus-MCTS action diagnostic. |
| 14 July | `5ea5281` and `e46c431` | Renamed/organized entry points and stored artifacts and neural-network checkpoints under package-specific directories. |
| 24-26 July | current uncommitted `legacy_muzero` files/checkpoints | Progressed through checkpoint formats v11-v14 and stabilized the small vanilla image MuZero training pipeline. |
| 26 July | current untracked `muzero/` file timestamps | Added native MinAtar observations, native diagnostics, strict checkpoint format v1, and the MinAtar training package. |
| 26-27 July | current native checkpoint metadata | Ran three Breakout-v1 experiments with history lengths 1, 5, and 2. |

The branch currently points at commit `e46c431`. The v14 legacy changes and the
entire native `muzero/` package are working-tree changes, so Git cannot provide
a more precise commit-by-commit history for those updates.

## 1. What `state_mcts` was doing

`state_mcts` was designed to answer controlled questions about learned pieces
of MCTS. It searches CartPole's real four-number state:

```text
[cart position, cart velocity, pole angle, pole angular velocity]
```

Its baseline combines exact CartPole equations, uniform UCT exploration, and
random leaf rollouts. Three independent switches replace one interface at a
time:

| Switch | Baseline behavior | Learned replacement |
| --- | --- | --- |
| dynamics | Exact CartPole transition and termination equations | Predict the next real four-number state; reward stays the known CartPole `+1` |
| policy | Uniform UCT exploration | Policy-network prior used by PUCT |
| value | Random leaf rollout | Value-network leaf estimate |

The default policy and value networks are supervised independently from a
fixed stabilizing CartPole heuristic. A sidecar teacher path instead trains the
policy on vanilla-MCTS visit distributions and the value on normalized
discounted return-to-go. The learned dynamics uses a multi-step curriculum up
to the configured search depth. The eight-way ablation tests every combination
of exact and learned pieces.

This made `state_mcts` interpretable and useful for diagnosing compounding
dynamics error, policy-prior effects, value calibration, and distribution
shift. It was not full MuZero: it had no learned observation representation,
the three models were not jointly optimized through recurrent imagined
unrolls, and its default targets did not come from end-to-end self-play.

## 2. What changed conceptually in the legacy image model

Although the legacy model did not descend from `state_mcts` in Git, the
conceptual difference between them is the step from independently replaceable
real-state components to a joint latent planning system.

| Concern | `state_mcts` | `legacy_muzero` v14 |
| --- | --- | --- |
| Observation | Real CartPole state vector | Five normalized 32x32 grayscale screenshots |
| Search state | Real or predicted four-number state | Learned spatial latent state |
| Representation | None | `h(observation history) -> latent` |
| Dynamics | Optional next-real-state model | Always-used `g(latent, action) -> next latent, reward` |
| Prediction | Separate optional policy and value networks | Joint `f(latent) -> policy logits, value` |
| Search target | Heuristic supervision by default; optional MCTS teacher | Raw normalized MCTS visit counts |
| Value target | Heuristic survival or teacher return | Discounted n-step real rewards plus a later MCTS root-value bootstrap |
| Optimization | Independent component trainers | Joint root plus recurrent-unroll training |
| Environment in search | Exact environment model may be used | Never queried below the real root |

The important legacy checkpoint iterations were:

- pre-v8: separate dynamics and policy/value checkpoints;
- v8: first unified flattened-latent checkpoint;
- v9: terminal-head and latent-consistency experiments;
- v11: spatial latent with globally pooled prediction heads;
- v12: position-preserving prediction heads, but temperature-sharpened replay
  policies;
- v13: raw visit-count targets, but untrained warm-up MCTS visits were still
  treated as valid policy labels and learning was suppressed during warm-up;
- v14: raw visit-count targets plus a per-transition `policy_valid` mask.
  Warm-up actions are random, policy loss is masked, and representation,
  dynamics, reward, and value learning remain active.

The current v14 checkpoint has 1,600 CartPole episodes. Its stored best
20-episode, fixed-seed MCTS evaluation is 500.0; its evaluation at episode
1,600 is 116.65. This illustrates why `best.pt`, `latest.pt`, and `final.pt`
have separate meanings.

## 3. What changed from `legacy_muzero` to native `muzero`

The core learning algorithm was retained. The main migration was from an image
wrapper around CartPole to native, semantic MinAtar observations:

1. **Native observations.** `MinAtarAdapter` converts MinAtar's boolean
   `[H,W,C]` observation to contiguous `float32 [C,H,W]`. It no longer renders
   an RGB frame, downsamples it, or converts it to grayscale for model input.
2. **Configurable observation history.** A positive `history_length` is
   zero-padded at the start of an episode and concatenated along the channel
   axis. The default is one native frame; there is no history RNN.
3. **Runtime-derived architecture.** Channel count, height, width, latent
   shape, and discrete action count come from the selected environment. The
   model is not hard-coded to 10x10 grids or a particular action count.
4. **Generic representation naming.** `ImageRepresentationNetwork` became
   `RepresentationNetwork`; the convolutional `h`, `g`, and `f` design is
   otherwise essentially the same as the current legacy version.
5. **Game/version-aware environments.** Short names select MinAtar v1 minimal
   action sets. Explicit `MinAtar/...-v0` IDs select the full-action variant.
   Sticky-action probability and difficulty ramping are configurable.
6. **Strict fresh checkpoints.** Native checkpoints carry format marker
   `native_minatar_vanilla_muzero`, schema version 1, environment ID, native
   shape/channel ordering, history length, action variant, architecture,
   optimizer, replay, histories, and RNG states. Legacy, cross-game,
   cross-history, and v0/v1 mismatches are rejected before weights are loaded.
7. **Native diagnostics.** GIFs place the actual environment rendering beside
   every native feature plane and history slot consumed by the model. JSON
   diagnostics compare raw-policy and MCTS behavior and report errors through
   recurrent depths.
8. **Per-game output layout.** Best, latest, and final checkpoints and their
   plots/diagnostics are isolated under game-specific directories.

No CartPole checkpoint was transferred. Every MinAtar run started with fresh
weights because the observation and action semantics are incompatible.

## 4. What the current native MuZero does

For every real environment decision:

1. The adapter converts the native grid to channel-first floats and the
   history stack produces `[history_length * C,H,W]`.
2. The representation network `h` encodes that real observation history into
   a min-max-normalized spatial latent state.
3. The prediction network `f` produces root policy logits and a scalar value.
4. MCTS runs PUCT in latent space. Each simulation expands one edge by calling
   `g(latent, action)` once for a predicted next latent and scalar reward, then
   calls `f` once for new priors and a leaf value.
5. Search backs up predicted rewards plus discounted values. The real
   environment is not queried during simulated lookahead.
6. During training, Dirichlet noise is mixed into root priors. After warm-up,
   the behavior action is sampled from a temperature-adjusted visit
   distribution, while the replay policy target remains the unsharpened
   normalized visit counts. Evaluation disables root noise and plays the
   most-visited action.
7. Replay stores complete episodes and never constructs an unroll across an
   episode boundary. It stores real observations/actions/rewards,
   termination/truncation, MCTS visits/root values, and policy validity.
8. Value labels are discounted n-step real rewards plus a later stored MCTS
   root-value bootstrap. After termination, recurrent padding uses dummy
   action zero and zero reward/value targets.
9. A learner update encodes one real root with `h`, predicts root targets with
   `f`, then applies `g` and `f` for the configured recurrent unroll. It jointly
   minimizes policy cross-entropy plus scalar value and reward MSE, with
   gradient clipping and reduced recurrent dynamics gradients.
10. Periodic fixed-seed, noise-free MCTS evaluation controls `best.pt`.
    `latest.pt` is resumable and `final.pt` records the end of the latest
    invocation.

This is intentionally vanilla MuZero. It does not implement EfficientZero,
reconstruction or latent-consistency losses, target encoders, reanalysis,
prioritized replay, recurrent observation models, categorical/distributional
value supports, multitask training, or cross-game transfer.

## 5. Latest native results

The current stored checkpoints are all Breakout-v1 with four native 10x10
channels, three minimal actions, 32 latent channels, 50 MCTS simulations,
five-step recurrent training unrolls, and a transition-relative update rate of
0.25 (clipped to 5-100 updates per episode).

| Run | Episodes | Best periodic MCTS evaluation | Final periodic evaluation | Last-100 actor mean | Maximum actor episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| history 1 | 2,500 | 7.6 | 5.2 | 6.18 | 17 |
| history 2 | 2,000 | **8.6** | **8.0** | **8.81** | 14 |
| history 5 | 1,000 | 7.0 | 6.6 | 5.52 | 9 |

The history-2 checkpoint is the strongest stored run on these measures, but
the episode budgets differ, so this is not a controlled history-length
ablation. There is no current history-0 checkpoint, and the implementation
rejects non-positive history lengths.

For the history-1 best checkpoint, the two saved diagnostics report:

- seed-42 evaluation: raw policy 4.6 versus MCTS 4.7; the recorded episode
  itself scored 0;
- seed-50 evaluation: raw policy 6.0 versus MCTS 9.0; the recorded episode
  scored 9.

Those examples show that search can improve over the raw policy, but they do
not establish that Breakout is solved. The project still needs controlled,
multi-seed aggregate evaluation across histories and games.

## 6. Current limitations and next useful checks

- Only Breakout has stored native checkpoints; the code supports Asterix,
  Freeway, Seaquest, and Space Invaders, but no results are present for them.
- Games train independently and checkpoints cannot be shared across games,
  history lengths, native channel layouts, or action-set variants.
- The current diagnostics cover only the history-1 Breakout checkpoint.
- The history comparison needs equal episode/update/search budgets and
  multiple seeds before drawing a conclusion about the best history length.
- The native package and its checkpoints are not committed on the current
  branch, so the latest implementation history is not reproducible from Git
  alone yet.
- The existing `muzero_migration_medium.md` draft says that history 5 was best,
  describes a history-0 experiment, and mixes CartPole and MinAtar results.
  The checkpoint metadata does not support those claims.

## Source map

- Real-state experiment: [`state_mcts/README.md`](../state_mcts/README.md)
- Legacy image MuZero: [`legacy_muzero/README.md`](../legacy_muzero/README.md)
- Native package overview: [`muzero/README.md`](../muzero/README.md)
- Native environment/history: [`muzero/environment.py`](../muzero/environment.py)
- Native `h`, `g`, and `f`: [`muzero/models.py`](../muzero/models.py)
- Native latent search: [`muzero/mcts.py`](../muzero/mcts.py)
- Episode replay: [`muzero/buffers.py`](../muzero/buffers.py)
- Actor/learner/checkpoints: [`muzero/train.py`](../muzero/train.py)
- Native diagnostics: [`muzero/diagnose.py`](../muzero/diagnose.py)
- Checkpoint semantics: [`checkpoints/muzero/README.md`](../checkpoints/muzero/README.md)

At review time, all current regression tests passed: 22 `state_mcts` tests and
63 combined native/legacy tests plus 7 subtests.

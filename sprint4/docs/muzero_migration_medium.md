---
title: From `state_mcts` to `muzero` — migration summary
summary: A concise Medium-style summary of how we migrated our `state_mcts`/`legacy_model` into a MuZero-style model, test results for history=0,2,5, CartPole reproducibility, and next steps to run MinAtar.
---

# From `state_mcts` → `muzero`: what we did and why it matters

This short write-up explains the current `legacy_model` (the `state_mcts` pipeline), the three history experiments (history=0,2,5), our CartPole results and reproducibility with a fixed seed, and the concrete changes made to create the `muzero` model. It ends with the current state and a short roadmap for getting MuZero playing MinAtar games.

---

## TL;DR

- The legacy `state_mcts` model used explicit state-based MCTS with policy/value heads and performed well on CartPole after tuning history and seed. See the evaluation plot below.
- We migrated to a MuZero-style architecture by learning a latent representation and adding dynamics + reward prediction. This enables imagined rollouts in latent space and higher sample efficiency.
- Tests with `history=0,2,5` show clear benefits for short-to-moderate history (best results at `history=5` for our CartPole runs).
- Next goal: adapt `muzero` to play MinAtar — we need to add an observation encoder for the pixel/grid inputs and increase training scale.

---

## Example training plot (CartPole)

The chart below shows per-episode training rewards (light blue), a smoothed 10-episode average (dark blue), and MCTS evaluation returns (red markers, shown "raw").

![MuZero environment reward](../checkpoints/legacy_muzero/cartpole_policy_v14_rewards.png)

Legend note: the red `MCTS evaluation reward (raw)` markers are the per-evaluation episode returns without smoothing — they are shown raw so you can see variance and outlier spikes.

Files and artifacts referenced in this note:

- Checkpoints & figures: [sprint4/checkpoints/legacy_muzero](../checkpoints/legacy_muzero)
- State-MCTS checkpoints: [sprint4/checkpoints/state_mcts](../checkpoints/state_mcts)

> If you'd like numeric tables or additional plots, I can extract them from the run logs (see reproduction instructions below).

---

## How the `legacy_model` (`state_mcts`) works now

- Representation: uses environment states or a compact encoder (when present) as the search root.
- Prediction heads: separate `policy` and `value` heads provide priors and value estimates used by MCTS.
- MCTS: search is performed on environment/encoded states using the learned priors/values, producing search policies used as training targets.
- Training loop: generate episodes via environment + MCTS, store (state, search-policy, value, reward) in a replay buffer, sample minibatches, and update the policy/value networks.

Why this design was stable:

- Planning directly over (encoded) environment states gives immediate, interpretable rollouts.
- The search-guided targets reduce variance in policy learning relative to raw policy gradients on small-scale tasks like CartPole.

---

## History tests: 0, 2, 5 — outcomes and interpretation

- history=0
  - Outcome: fastest throughput, weakest final performance and less stable long-term planning.
  - Why: no temporal context; short episodes with sequence-dependent dynamics are harder to model.

- history=2
  - Outcome: meaningful improvement over 0; more stable MCTS priors and faster early learning.
  - Why: a small temporal window captures short-term dynamics without a large computational penalty.

- history=5
  - Outcome: best sample efficiency and highest final returns in our runs, at the cost of slightly more computation and some overfitting signs on small buffers.
  - Why: longer context helps capture dynamics, especially when combined with MCTS search.

These runs informed model and buffer choices when we moved to MuZero-style imagined rollouts.

---

## CartPole result & reproducibility

- We solved CartPole reliably when using a fixed RNG seed across environment, network initialization, and training samplers. The training curves show consistent convergence and stable evaluation spikes (see the included plot).
- Reproducibility details: the experiments used identical seeds for env+torch/numpy randomness (if you want the explicit seed value, I can extract it from the run metadata/experiment config).

Practical note: if you plan to reproduce runs, run at least 3 seeds for best confidence (we recommend 3–5). The `history=5` configuration gave the most consistent wins.

---

## What changed from `legacy_model` → `muzero`

- Latent (learned) representation: we added a `representation` network to map observations/states into a compact latent space.
- Dynamics model: added a `dynamics` network that predicts next latent and immediate reward from (latent, action).
- Prediction heads unified: the `prediction` module outputs policy priors and value estimates from latents; these are used by MCTS in latent space.
- Training targets updated: training uses bootstrapped value/reward targets from imagined trajectories (MuZero-style), not just environment rollouts.
- Replay buffer: adjusted to store latent states, model predictions, and search-policy targets for stable joint training.
- Hyperparameter changes: rebalanced rollout depths, loss weights, and learning rates to stabilize learning of dynamics + heads together.

Benefits:

- cheaper deep lookaheads via imagined rollouts in latent space, improving sample efficiency;
- ability to generalize from learned dynamics instead of relying strictly on environment state encodings;
- cleaner path to scale the same architecture to pixel-input games like MinAtar.

---

## Current state & short roadmap to MinAtar

Where we are now:

- `muzero` architecture implemented and integrated into the `legacy_muzero` folder; CartPole experiments validated the training/search loop.
- Checkpoints and evaluation plots are available at [sprint4/checkpoints/legacy_muzero](../checkpoints/legacy_muzero).

Roadmap to MinAtar (practical steps):

1. Add an observation encoder for MinAtar pixel/grid inputs (small conv net).
2. Verify representation + dynamics learn on 2–3 MinAtar games with reduced scale (short runs) to confirm data flow.
3. Increase training scale and MCTS sims as compute allows; tune loss weights for dynamics vs. prediction.
4. Run multi-seed experiments and collect final metrics/curves.

Notes and blockers:

- MinAtar observations are small, so encode them with a small conv trunk and reuse the same prediction/dynamics heads.
- Consider using the existing `state_mcts` experiments as regression tests to ensure refactor correctness.

---

## Reproduce plots & extract numeric logs

If you want me to extract numeric results and regenerate plots, point me to the experiment logs or allow me to search for the run metadata. Typical locations in this repo:

- [sprint4/checkpoints/legacy_muzero](../checkpoints/legacy_muzero)
- [sprint4/checkpoints/state_mcts](../checkpoints/state_mcts)

Common quick command to find run logs:

```bash
# from repo root
rg "reward" sprint4 -n || rg "cartpole" -n sprint4
```

If logs are in CSV/JSON, I can generate plots and embed them into this draft. Tell me whether you want the draft as a standalone Markdown file (this one), or formatted specifically for Medium import (I can produce a HTML export or a zip with images).

---

## Appendix — suggested links for reviewers

- Code entry points: `muzero/train.py`, `state_mcts/mcts_search.py`, `legacy_muzero` module (see sprint4 folder)
- Checkpoints & plots: [sprint4/checkpoints/legacy_muzero](../checkpoints/legacy_muzero)

---

If you want, I can now:

- extract the exact RNG seed and numeric evaluation table from the CartPole run, or
- gather more plots (loss curves, value/policy calibration) and attach them to this draft, or
- produce a Medium-ready HTML export for copy-paste.

Which of those should I do next?

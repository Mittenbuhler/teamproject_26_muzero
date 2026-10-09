# Cleanup verification — 9 October 2026

The final CartPole and MinAtar implementations and the modular MCTS testbed were
checked after the documentation and model-packaging cleanup. No training or
search algorithm was changed by this cleanup.

## Checks completed

- All **88 existing tests passed** across `cartpole_muzero.tests.test_pipeline`,
  `minatar_muzero.tests.test_pipeline`, `state_mcts.tests.test_experiment` and
  `state_mcts.tests.test_dashboard`.
- Both new inference files load through the current model loaders. Every tensor
  in all three networks equals its counterpart in the original best checkpoint.
  Export source files were hashed before and after and were unchanged.
- The central quick-start diagnostic commands completed with both inference
  files on seed 20000. CartPole: random 9, policy 364, MCTS 363 (50 simulations).
  Breakout: policy 1, MCTS 5 (25 simulations). These are execution checks on one
  seed, not replacement performance estimates for the curated reports.
- The two main best/latest pairs match the original hashes stored in the
  existing evaluation/training-history JSON reports.
- Fourteen locally readable original `.pt` files and both new inference files
  pass SHA-256 verification. All 16 original `.pt` paths, byte sizes and
  modification times are unchanged.
- All local links in current/index documentation resolve. Historical documents
  retained verbatim under `docs/history/` intentionally contain old paths.
- Documented command flags were checked against the actual CLI declarations.
  Dependency consistency (`pip check`) and whitespace checks passed.

Fresh diagnostic outputs are under the ignored `artifacts/verification/`
directory. Existing curated figures, numeric results and checkpoint contents
were not regenerated or overwritten.

## Limits

Two originals were cloud-backed placeholders, so their contents could not be
read for hash verification:

- `checkpoints/minatar_muzero/breakout_screenshots/final.pt`
- `checkpoints/state_mcts/20261004_175041/state_value.pt`

They remain in place with the same size and modification time. The manifest
records `sha256: null` and an explanation rather than claiming a verified hash.
`python -m scripts.check_checkpoints --originals --verify` therefore exits with
status 1 for these two entries; the portable-only check succeeds. Original best
and latest files for both main result runs are available and verified.

No empty project directories were found, so no directories or existing data
were deleted. Historical material was labeled in place to preserve references.
The previously uncommitted package reorganization was retained; deletions of
old package paths visible in Git already existed before this cleanup. The two
previously deleted historical prose documents were recovered into
`docs/history/` with outdated/correction notices.

## Environment

Verification used Python 3.13.0 on macOS ARM64 with PyTorch 2.14.1, NumPy 2.5.3,
Gymnasium 1.4.0 and MinAtar 1.0.15. Full versions are recorded in
[requirements-verified.txt](../requirements-verified.txt). The original Conda
Python was Intel-only and could not execute on this machine, so a separate,
ignored `sprint4/.venv` was created.

Original result reports record PyTorch 2.2.2, NumPy 1.26.4 and Gymnasium 1.3.0.
Cross-version/platform trajectories need not be bit-identical. The weight
identity check is exact, and this cleanup does not revise the published scores.

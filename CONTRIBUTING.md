# Contributing to CorSeaCare_yolo

Thanks for helping! This is a **citizen-science** project of the
[CorSeaCare](https://www.marevivu.com) expedition (association *Mare Vivu*, Corsica). Three
kinds of contribution are welcome:

## 1. Photos & annotations (most valuable)
The single biggest lever on accuracy is **more annotated, distinct sieves**. If you have
sieve photos from a Manta-net sample:

- Photograph the sieve **top-down**, frame-filled, with a **scale ruler** in shot, ~12 MP
  JPEG (see the README — you don't need 48 MP RAW).
- Annotate in Label Studio (see [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md)) using
  the classes `fragment`, `fibre`, `film`, `mousse`, `pellet`, `autre`.
- Record each photo in `samples.csv` (one physical sieve = one `sample_id`; note the
  `sieve_mm` mesh size).

**Privacy:** your photos, annotations, datasets and weights live under `data/` and `runs/`,
which are **git-ignored** — they are *never* committed. Only share imagery you have the
right to share. If you want to donate data to the shared training set, open an issue first.

## 2. Code
- Set up: `./scripts/setup.sh` (or `make setup`), then `make test`.
- This repo is **test-driven**: add or update a test in `tests/` for any behaviour change,
  and keep `uv run pytest` green. Match the style of the surrounding code.
- Keep modules small and focused; the engine (`src/corseacare/`) stays free of heavy
  model imports so it can be tested without PyTorch (see the `Fake*` backends).
- Open a PR with a short description and the test plan. One logical change per PR.

## 3. Issues
Bug reports, dataset ideas, and accuracy observations (especially on **unseen** sieves) are
all useful. Please include your OS, how you installed, and the exact command.

## License
By contributing you agree your contributions are licensed under the project's
**AGPL-3.0-or-later** (see [LICENSE](LICENSE)).

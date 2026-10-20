# MyFirstBDH

Training the graph version of BDH (Dragon Hatchling, Kosowski et al. 2025,
arXiv:2509.26507): Eq. 6 / Table 1, with the 5 graphs Gx_e, Gx_i, Gy_e, Gy_i, Gs.
First dataset: a tiny "toy world" whose answers are only in the story,
so the model must use its memory sigma.

## Setup
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run (from the repo root)
    python -m scripts.make_toyworld --level 1
    python -m scripts.train
    python -m scripts.evaluate checkpoints/tiny.pt
    python -m scripts.record
    python -m viz.rounds traces/example.npz
    python -m pytest

## Layout
- bdhgraph/   the model: config, the 5 G's and the 4 rounds, data, tracing
- scripts/    things you run: make_toyworld, train, evaluate, sample, record
- viz/        things you look at: rounds (2D), graph3d (3D), weights
- tests/      small checks done by hand
- reference/  the paper and notes
- data/ checkpoints/ traces/ figures/   generated files (not in git)

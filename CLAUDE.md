# MyFirstBDH

Learning project: train the **graph version** of BDH (Dragon Hatchling, Kosowski et al. 2025,
arXiv:2509.26507), i.e. Eq. 6 / Table 1 of the paper, with its 5 graphs, on a MacBook Air.
The paper only trains the GPU version (BDH-GPU); this repo trains the graph version directly.

## How to work with me

- I am learning this. **Explain before you code**, slowly, step by step.

- Change **one thing at a time** (one round, one function). Show the diff, then explain it.
- Keep code simple and readable. No clever tricks, no extra abstractions.
- Explanations in English.
- Ask before: adding a dependency, changing the folder layout, running anything longer
  than ~2 minutes (use `steps=20` for quick checks).
- If something in the paper is ambiguous, say so and show the options. Don't guess silently.

## The model (source of truth)

n neurons, L layers. For each letter, for each layer l, in this order:

```python
x     = x + relu((Gx_e - Gx_i) @ y)        # Y pushes X     (Table 1, round 4l+3)
a     = sigma[l] @ x                        # read memory    (round 4l)
sigma[l] = u * (sigma[l] + outer(y, x) * Gs)   # write memory (round 4l+1), uses y from previous layer
y     = relu((Gy_e - Gy_i) @ a) * x         # A pushes Y     (round 4l+2)
```

- `a` is read from the memory **before** this letter writes into it (sigma at t-1).
- The write uses `y` from the previous layer and `x` from this layer.
- Table 1 and Eq. 6 name the two indices of sigma in opposite order.
  **Follow the matrix form above** (`outer(y, x)`, `sigma @ x`), not the table's letters.

### What is trained vs not

| Name | Shape | Trained? | Meaning |
|---|---|---|---|
| Gx_e, Gx_i | n x n, >= 0 | yes | Y -> X wires (excitatory / inhibitory) |
| Gy_e, Gy_i | n x n, >= 0 | yes | A -> Y prods (excitatory / inhibitory) |
| Gs | n x n, >= 0 | no by default (fixed mask) | which memory connections may exist |
| embed | 256 x n | yes | letter -> neurons |
| head | n -> 256 | yes | neurons -> next-letter scores |
| sigma[l] | n x n per layer | **no** | memory of the current text, starts at 0 |

- Keep the G's >= 0: call `model.clamp_()` after every optimizer step.
- Gs modes (config `gs`): `"dense"` (all ones, like BDH-GPU), `"groups"` (block diagonal),
  `"random"` (fixed 0/1 mask with density `gs_density`).

### Our own choices (not in the paper)

- Input: `y = relu(embed(letter))`, `x = 0` at the start of each letter.
- Output: `head(x)` after the last layer (the paper reads out through X).
- Decay `u` is one fixed scalar (like ALiBi).
- With a batch, sigma has shape `(B, n, n)` per layer.
- Normalization: if values blow up (loss = nan), add a scale-only norm
  `z / sqrt(mean(z**2) + eps)`, which keeps values >= 0. Say so when you add it.

### Later (not now)

- "Trick 1": replace each pair `G_e - G_i` with one free matrix W, split it back after training.
- "Trick 2": with dense Gs, compute the memory read as causal linear attention (parallel, fast).
  Then add a test that the parallel and loop versions give the same numbers.
- Trainable Gs.

## Sizes

| Config | n | L | T (letters/chunk) | batch | trained numbers | RAM |
|---|---|---|---|---|---|---|
| TINY | 128 | 4 | 64 | 8 | ~148k | ~0.5 GB |
| SMALL | 256 | 4 | 64 | 8 | ~0.5M | ~2 GB |

- Start loss ≈ ln(256) ≈ 5.5. A model using only the previous letter gets roughly 2.5.
  Below that, the memory is helping.
- Run on CPU first. Apple GPU (`mps`) is optional, probably not faster at this size.

## Layout

```
bdhgraph/   config.py, model.py, data.py, trace.py   (the model)
scripts/    train.py, sample.py, record.py           (things you run)
viz/        rounds.py (2D), graph3d.py (3D), weights.py   (things you look at)
tests/      test_rounds.py                           (hand-checked examples)
reference/  paper PDF, notes.md
data/ checkpoints/ traces/ figures/                  (generated, not in git)
```

Flow: `train.py` → `checkpoints/` → `record.py` → `traces/` → `viz/*` → `figures/`.
The model never imports viz. It only calls `tracer.record(...)` when a tracer is passed.

## Visualizations (planned)

- `viz/rounds.py`: X, Y, A matrices (one row per letter-and-layer, one column per neuron),
  sigma heatmap per layer (rows = from i, columns = to j), and which G is used at each round.
- `viz/graph3d.py`: plotly 3D, neurons as points, edges for Gx, Gy (excitatory / inhibitory
  colors) and sigma (thickness = strength). Draw only the top-k strongest edges.
- `viz/weights.py`: heatmaps and degree distributions of the G's (paper Section 5).

## Commands (from the repo root)

```bash
source .venv/bin/activate
python -m scripts.train
python -m scripts.sample
python -m scripts.record
python -m viz.rounds traces/example.npz
python -m viz.graph3d checkpoints/tiny.pt
python -m pytest
```

Always use `python -m ...` from the root so `import bdhgraph` works.

## References

- Paper: `reference/2509.26507v1.pdf` (Table 1, Eq. 6, Fig. 3, Appendix B.2 and E)
- Official BDH-GPU code: https://github.com/pathwaycom/bdh
  Their names are flipped vs the paper: `encoder` = Dx, `encoder_v` = Dy, `decoder` = E.

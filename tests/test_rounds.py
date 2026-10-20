"""Hand-checked example: 3 neurons, 1 layer, 2 letters.
Compute x, a, sigma, y on paper, then check the model gives the same numbers.
Run: python -m pytest
"""

from dataclasses import replace

import torch

from bdhgraph.config import TINY
from bdhgraph.model import BDHGraph


def t(rows):
    return torch.tensor(rows, dtype=torch.float32)


def make_model():
    cfg = replace(TINY, n=3, L=1, u=0.5, gs="dense", norm_a=False)
    m = BDHGraph(cfg)
    # Wx = Gx_e - Gx_i = [[1,0,0],[0,0,1],[1,1,-1]]
    m.Gx_e.data = t([[1, 0, 0], [0, 0, 1], [1, 1, 0]])
    m.Gx_i.data = t([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    # Wy = Gy_e - Gy_i = [[1,1,0],[1,0,0],[0,0,-1]]
    m.Gy_e.data = t([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    m.Gy_i.data = t([[0, 0, 0], [0, 0, 0], [0, 0, 2]])
    return m


def test_one_layer_by_hand():
    m = make_model()
    sigma = torch.zeros(1, 3, 3)

    # letter 1: y = [1, 0, 2], x = 0, sigma = 0
    x, a, y, sigma = m.layer(torch.zeros(1, 3), t([[1, 0, 2]]), sigma)
    assert torch.allclose(x, t([[1, 2, 0]]))
    assert torch.allclose(a, t([[0, 0, 0]]))  # empty memory
    assert torch.allclose(sigma, t([[[.5, 1, 0], [0, 0, 0], [1, 2, 0]]]))
    assert torch.allclose(y, t([[0, 0, 0]]))

    # letter 2: x and y restart, sigma carries over
    x, a, y, sigma = m.layer(torch.zeros(1, 3), t([[0, 2, 1]]), sigma)
    assert torch.allclose(x, t([[0, 1, 1]]))
    assert torch.allclose(a, t([[1, 0, 2]]))  # read what letter 1 wrote
    assert torch.allclose(sigma, t([[[.25, .5, 0], [0, 1, 1], [.5, 1.5, .5]]]))
    assert torch.allclose(y, t([[0, 1, 0]]))  # neuron 0 gated by x[0] = 0

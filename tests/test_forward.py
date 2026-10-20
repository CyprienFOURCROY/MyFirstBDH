"""Tests for the full forward pass: shapes, hand example, causality, memory off, padding.
Run: python -m pytest
"""

from dataclasses import replace

import torch

from bdhgraph.config import TINY
from bdhgraph.model import BDHGraph
from test_rounds import make_model, t  # the 3-neuron model with the hand-chosen G's


def small_model(**changes):
    torch.manual_seed(0)
    return BDHGraph(replace(TINY, n=8, L=2, **changes))


def test_shapes():
    m = small_model()
    idx = torch.randint(256, (3, 5))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 5, 256)
    assert loss.shape == ()
    assert m(idx)[1] is None  # no targets -> no loss


def test_hand_example_through_forward():
    m = make_model()  # n=3, L=1, u=0.5, G's from the hand example
    # make the embedding give y = [1,0,2] for letter 5 and y = [0,2,1] for letter 6
    m.embed.weight.data[5] = t([1, 0, 2])
    m.embed.weight.data[6] = t([0, 2, 1])
    head = torch.randn(256, 3)
    m.head.weight.data = head

    logits, _ = m(torch.tensor([[5, 6]]))
    # from tests/test_rounds.py: x = [1,2,0] after letter 1, x = [0,1,1] after letter 2
    assert torch.allclose(logits[0, 0], head @ t([1, 2, 0]), atol=1e-6)
    assert torch.allclose(logits[0, 1], head @ t([0, 1, 1]), atol=1e-6)


def test_norm_a_gives_unit_scale():
    m = small_model(norm_a=True)
    sigma = torch.rand(2, 8, 8)
    x, a, y, sigma = m.layer(torch.rand(2, 8), torch.rand(2, 8), sigma)
    assert torch.allclose((a**2).mean(dim=-1), torch.ones(2), atol=1e-4)
    assert (a >= 0).all()


def test_causality():
    m = small_model()
    idx = torch.randint(256, (2, 6))
    other = idx.clone()
    other[:, -1] = (idx[:, -1] + 1) % 256  # change only the last letter
    a, _ = m(idx)
    b, _ = m(other)
    assert torch.equal(a[:, :-1], b[:, :-1])  # earlier positions did not change
    assert not torch.equal(a[:, -1], b[:, -1])  # the last one did


def test_memory_off_sees_only_current_letter():
    m = small_model()
    m.Gs.zero_()
    a, _ = m(torch.tensor([[3, 7, 9]]))
    b, _ = m(torch.tensor([[4, 7, 1]]))  # same letter 7 at position 1, different past
    assert torch.allclose(a[:, 1], b[:, 1], atol=1e-6)


def test_memory_on_sees_the_past():
    m = small_model()
    a, _ = m(torch.tensor([[3, 7, 9]]))
    b, _ = m(torch.tensor([[4, 7, 1]]))
    assert not torch.allclose(a[:, 1], b[:, 1], atol=1e-6)


def test_padding_does_not_change_the_loss():
    m = small_model()
    story = torch.tensor(list(b"ab.\n"))
    x, y = story[:-1].unsqueeze(0), story[1:].unsqueeze(0)  # no padding, T = 3

    xp = torch.zeros(1, 8, dtype=torch.int64)  # padded to T = 8
    yp = torch.full((1, 8), -100, dtype=torch.int64)
    xp[0, :3], yp[0, :3] = x[0], y[0]

    _, loss = m(x, y)
    _, loss_padded = m(xp, yp)
    assert torch.allclose(loss, loss_padded, atol=1e-6)


def test_tracer_records_without_changing_the_result():
    from bdhgraph.trace import Tracer

    m = small_model()
    idx = torch.randint(256, (2, 5))
    plain, _ = m(idx)
    tracer = Tracer()
    traced, _ = m(idx, tracer=tracer)
    assert torch.equal(plain, traced)
    assert len(tracer.steps) == 5 * 2  # T letters x L layers
    assert tracer.steps[0]["x"].shape == (8,)
    assert tracer.steps[-1]["sigma"].shape == (8, 8)  # only the last letter keeps sigma
    assert tracer.steps[0]["sigma"] is None
    assert tracer.steps[1]["y_in"].shape == (8,)  # the y that entered layer 1 ...
    assert torch.equal(torch.tensor(tracer.steps[1]["y_in"]), torch.tensor(tracer.steps[0]["y"]))  # ... is layer 0's output

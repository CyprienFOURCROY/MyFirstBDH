"""Check the gradient formula of the round "Y pushes X" against loss.backward().

    z     = (Gx_e - Gx_i) @ y
    x_new = x + relu(z)
    L     = (c * x_new).sum()            so dL/dx_new = c

    g_z            = c * [z > 0]          (the ReLU gate: no gradient where z <= 0)
    grad Gx_e(j,i) = g_z(j) * y(i)
    grad Gx_i(j,i) = -g_z(j) * y(i)

Run: python -m pytest
"""

import torch


def t(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_push_x_gradient():
    Gx_e = t([[1, 0, 0], [0, 0, 1], [1, 1, 0]]).requires_grad_()
    Gx_i = t([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).requires_grad_()
    y = t([1, 0, 2])
    x = t([0, 0, 0])
    c = t([1, -2, 3])

    z = (Gx_e - Gx_i) @ y
    assert torch.equal(z, t([1, 2, -1]))  # z[2] < 0: the ReLU gate is closed there
    x_new = x + torch.relu(z)
    loss = (c * x_new).sum()
    loss.backward()

    g_z = c * (z > 0).float()  # [1, -2, 0]
    expected = torch.outer(g_z, y)  # entry (j, i) = g_z[j] * y[i]
    assert torch.allclose(Gx_e.grad, expected)
    assert torch.allclose(Gx_i.grad, -expected)
    assert torch.all(Gx_e.grad[2] == 0)  # the closed gate gives a zero row

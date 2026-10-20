"""BDH graph model (paper Eq. 6 / Table 1).

Trained (fixed while reading a text):
  Gx_e, Gx_i : Y -> X wires (excitatory, inhibitory)   n x n, >= 0
  Gy_e, Gy_i : A -> Y prods (excitatory, inhibitory)   n x n, >= 0
  embed      : letter -> neurons
  head       : neurons -> next-letter scores
Fixed at first:
  Gs         : which memory connections may exist     n x n, >= 0

Not trained (rebuilt for every text):
  sigma[l]   : memory of layer l, n x n, starts at 0

For each letter: y = relu(embed(letter)), x = 0. Then for each layer:
  x     = x + relu((Gx_e - Gx_i) @ y)        # Y pushes X
  a     = sigma @ x                           # read memory
  sigma = u * (sigma + outer(y, x) * Gs)      # write memory (Hebbian)
  y     = relu((Gy_e - Gy_i) @ a) * x         # A pushes Y
Prediction of the next letter: head(x) after the last layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def scale_norm(z, eps=1e-6):
    """z / sqrt(mean(z**2) + eps), over the n neurons. Keeps values >= 0, no learned gain."""
    return z / torch.sqrt((z**2).mean(dim=-1, keepdim=True) + eps)


class BDHGraph(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n = cfg.n

        # trained G's: random values in [0, 1/sqrt(n)), so they start >= 0
        # and (Gx_e - Gx_i) @ y does not grow with n
        self.Gx_e = nn.Parameter(torch.rand(n, n) / n**0.5)
        self.Gx_i = nn.Parameter(torch.rand(n, n) / n**0.5)
        self.Gy_e = nn.Parameter(torch.rand(n, n) / n**0.5)
        self.Gy_i = nn.Parameter(torch.rand(n, n) / n**0.5)

        # Gs is a fixed mask: a buffer is saved in checkpoints but not trained
        self.register_buffer("Gs", self.make_gs(cfg))

        self.embed = nn.Embedding(cfg.vocab, n)
        self.head = nn.Linear(n, cfg.vocab, bias=False)

    @staticmethod
    def make_gs(cfg):
        n = cfg.n
        if cfg.gs == "dense":
            return torch.ones(n, n)
        if cfg.gs == "random":
            return (torch.rand(n, n) < cfg.gs_density).float()
        raise NotImplementedError(f"gs mode '{cfg.gs}' comes in step 12")

    def layer(self, x, y, sigma):
        """One layer for one letter. x, y: (B, n). sigma: (B, n, n).
        Returns x, a, y, sigma after the four rounds."""
        Wx = self.Gx_e - self.Gx_i
        Wy = self.Gy_e - self.Gy_i

        x = x + F.relu(y @ Wx.T)                    # Y pushes X
        a = (sigma @ x.unsqueeze(2)).squeeze(2)     # read memory (old sigma)
        if self.cfg.norm_a:                         # without it, values blow up
            a = scale_norm(a)                       # (not in the paper, see CLAUDE.md)
        outer = y.unsqueeze(2) * x.unsqueeze(1)     # outer[b, i, j] = y_i * x_j
        sigma = self.cfg.u * (sigma + outer * self.Gs)   # write memory
        y = F.relu(a @ Wy.T) * x                    # A pushes Y
        return x, a, y, sigma

    def forward(self, idx, targets=None, tracer=None):
        """idx, targets: (B, T) letters. Returns logits (B, T, vocab) and the loss."""
        B, T = idx.shape
        n = self.cfg.n

        # memory of each layer: starts at 0 for each chunk, kept from letter to letter
        sigma = [torch.zeros(B, n, n, device=idx.device) for _ in range(self.cfg.L)]

        logits = []
        for t in range(T):
            y = F.relu(self.embed(idx[:, t]))         # restart at every letter
            x = torch.zeros(B, n, device=idx.device)
            for l in range(self.cfg.L):
                y_in = y
                x, a, y, sigma[l] = self.layer(x, y, sigma[l])
                if tracer is not None:  # one record per (letter, layer); sigma only at the last letter
                    tracer.record(t, l, "layer", x, y, a, sigma[l] if t == T - 1 else None, y_in)
            logits.append(self.head(x))               # x after the last layer
        logits = torch.stack(logits, dim=1)           # (B, T, vocab)

        loss = None
        if targets is not None:
            # -100 marks padding: the loss skips those positions
            loss = F.cross_entropy(
                logits.reshape(B * T, -1), targets.reshape(B * T), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def clamp_(self):
        """Keep the 4 trained G's >= 0 (call after each optimizer step)."""
        for G in (self.Gx_e, self.Gx_i, self.Gy_e, self.Gy_i):
            G.clamp_(min=0)

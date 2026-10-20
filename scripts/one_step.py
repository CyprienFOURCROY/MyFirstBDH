"""One gradient descent step by hand (no optimizer), on one fixed batch.
Run from the repo root: python -m scripts.one_step --lr 0.1
"""

import argparse

import torch

from bdhgraph.config import TINY
from bdhgraph.data import get_batch
from bdhgraph.model import BDHGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(TINY.seed)
    model = BDHGraph(TINY)
    x, y = get_batch("train", TINY)

    _, loss = model(x, y)
    print(f"loss before: {loss.item():.4f}")
    loss.backward()  # fills p.grad for every parameter

    old = {name: p.detach().clone() for name, p in model.named_parameters()}

    with torch.no_grad():
        for p in model.parameters():
            p -= args.lr * p.grad  # the descent step
    model.clamp_()  # G's back to >= 0

    with torch.no_grad():
        _, loss_after = model(x, y)  # SAME batch
    print(f"loss after:  {loss_after.item():.4f}  (lr = {args.lr})")

    print(f"{'param':14s} {'changed':>8s} {'biggest change':>15s} {'exactly 0':>10s}")
    for name, p in model.named_parameters():
        diff = (p.detach() - old[name]).abs()
        zeros = int((p == 0).sum()) if name.startswith("G") else "-"
        print(f"{name:14s} {int((diff > 0).sum()):8d} {diff.max().item():15.6f} {zeros:>10}")


if __name__ == "__main__":
    main()

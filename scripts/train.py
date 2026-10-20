"""Train the graph model. Run from the repo root: python -m scripts.train [--steps 200]

For now: plain gradient descent written by hand, no checkpoint.
Every step is appended to checkpoints/runs/<date_time>.jsonl (one file per run),
which the dashboard (python3 viewer/app.py) reads, live or later.
The weights go to checkpoints/runs/<date_time>.pt (every EVAL_EVERY steps and at the end).
TODO later: AdamW instead of the hand-written update, save checkpoints/tiny.pt.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import torch

from bdhgraph.config import TINY
from bdhgraph.data import get_batch
from bdhgraph.model import BDHGraph

LR = 0.1  # step size of plain gradient descent (worked in scripts/one_step.py)
EVAL_EVERY = 10  # val loss every this many steps
RUNS_DIR = "checkpoints/runs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)  # 20 = quick check
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--level", type=int, default=TINY.level)
    parser.add_argument("--n", type=int, default=TINY.n)          # neurons
    parser.add_argument("--L", type=int, default=TINY.L)          # layers
    parser.add_argument("--u", type=float, default=TINY.u)        # forgetting factor of sigma
    parser.add_argument("--batch", type=int, default=TINY.batch)
    parser.add_argument("--seed", type=int, default=TINY.seed)
    parser.add_argument("--gs", default=TINY.gs, choices=["dense", "random"])
    parser.add_argument("--gs_density", type=float, default=TINY.gs_density)
    parser.add_argument("--no_norm_a", action="store_true")       # switch the norm on a off
    args = parser.parse_args()

    cfg = replace(TINY, steps=args.steps, level=args.level, n=args.n, L=args.L, u=args.u,
                  batch=args.batch, seed=args.seed, gs=args.gs, gs_density=args.gs_density,
                  norm_a=not args.no_norm_a)
    torch.manual_seed(cfg.seed)
    model = BDHGraph(cfg)

    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, time.strftime("%Y-%m-%d_%H-%M-%S") + ".jsonl")
    log = open(path, "w")  # one new file per run, old runs are kept
    print("logging to", path)
    model_path = path.replace(".jsonl", ".pt")

    def save(step):
        torch.save({"cfg": asdict(cfg), "state": model.state_dict(), "step": step}, model_path)

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()  # so the dashboard sees it right away

    write({"type": "start", "steps": cfg.steps, "n": cfg.n, "L": cfg.L, "lr": args.lr, "level": cfg.level,
           "cfg": asdict(cfg)})
    t0 = time.time()

    for step in range(cfg.steps):
        x, y = get_batch("train", cfg)  # a new batch at each step
        model.zero_grad()  # backward() adds to .grad, so empty it first
        _, loss = model(x, y)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= args.lr * p.grad
        model.clamp_()
        print(f"step {step:2d}  loss {loss.item():.4f}")
        write({"type": "step", "step": step, "loss": loss.item(), "elapsed": time.time() - t0})

        if (step + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                _, val_loss = model(*get_batch("val", cfg))
            write({"type": "val", "step": step, "loss": val_loss.item()})
            save(step + 1)

    save(cfg.steps)
    write({"type": "done"})
    log.close()


if __name__ == "__main__":
    main()

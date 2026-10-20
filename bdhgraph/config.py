"""Model, data and training sizes."""

from dataclasses import dataclass


@dataclass
class Config:
    # data
    data: str = "toyworld"   # "toyworld" or "shakespeare"
    level: int = 1           # toy world level: 1, 2 or 3
    # model
    n: int = 128             # neurons
    L: int = 4               # layers
    vocab: int = 256         # raw bytes
    u: float = 0.97          # forgetting factor for sigma
    gs: str = "dense"        # "dense" | "groups" | "random"
    gs_density: float = 0.5  # used when gs == "random"
    norm_a: bool = True      # scale-only norm on the memory read a (not in the paper)
    # training
    T: int = 96              # letters per chunk (one toy story fits)
    batch: int = 8           # chunks per step
    lr: float = 1e-3
    steps: int = 2000
    seed: int = 1337


TINY = Config()
SMALL = Config(n=256)

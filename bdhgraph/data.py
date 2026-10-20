"""Load the training text and cut it into chunks.

cfg.data == "toyworld":    data/toyworld_l{level}_train.txt and _val.txt
                           (make them first: python -m scripts.make_toyworld --level 1)
                           one chunk = one story (one line, with its "\n"),
                           padded with byte 0 up to cfg.T. Padded targets are -100.
cfg.data == "shakespeare": data/input.txt (downloaded, 90% train / 10% val)
                           one chunk = a random window of cfg.T letters
"""

import os
import urllib.request

import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
IGNORE = -100  # target value that the loss skips (padding)

_cache = {}  # (data, level, split) -> text as bytes, so we read each file once


def load(split, cfg):
    """Return the text of this split ("train" or "val") as bytes."""
    key = (cfg.data, cfg.level, split)
    if key in _cache:
        return _cache[key]

    if cfg.data == "toyworld":
        path = f"data/toyworld_l{cfg.level}_{split}.txt"
        with open(path, "rb") as f:
            text = f.read()
        # every story must fit in one chunk of cfg.T letters
        longest = max(len(line) for line in text.splitlines(keepends=True))
        print(f"longest story ({split}, level {cfg.level}): {longest} bytes, T = {cfg.T}")
        if longest > cfg.T:
            raise ValueError(f"a story has {longest} bytes but T = {cfg.T}: raise T")
    elif cfg.data == "shakespeare":
        path = "data/input.txt"
        if not os.path.exists(path):
            os.makedirs("data", exist_ok=True)
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
        with open(path, "rb") as f:
            everything = f.read()
        cut = int(0.9 * len(everything))
        text = everything[:cut] if split == "train" else everything[cut:]
    else:
        raise ValueError(f"unknown data: {cfg.data}")

    _cache[key] = text
    return text


def get_batch(split, cfg):
    """Return x, y of shape (cfg.batch, cfg.T). y is x shifted by one letter."""
    text = load(split, cfg)

    if cfg.data == "toyworld":
        lines = text.splitlines(keepends=True)  # each story keeps its "\n"
        x = torch.zeros(cfg.batch, cfg.T, dtype=torch.int64)             # padding = byte 0
        y = torch.full((cfg.batch, cfg.T), IGNORE, dtype=torch.int64)    # padding = ignored
        for b, k in enumerate(torch.randint(len(lines), (cfg.batch,))):
            story = torch.tensor(list(lines[k]), dtype=torch.int64)
            m = len(story)
            x[b, : m - 1] = story[:-1]  # the last byte ("\n") is only a target
            y[b, : m - 1] = story[1:]
        return x, y

    data = torch.tensor(list(text), dtype=torch.int64)
    picks = torch.randint(len(data) - cfg.T, (cfg.batch,))
    chunks = torch.stack([data[s : s + cfg.T + 1] for s in picks])  # (batch, T+1)
    return chunks[:, :-1], chunks[:, 1:]

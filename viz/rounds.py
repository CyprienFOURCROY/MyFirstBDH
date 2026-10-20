"""2D view of the rounds, from a trace file.

Shows, step by step:
  X matrix : one row per (letter, layer), one column per neuron
  Y matrix : same for y
  A matrix : same for a
  sigma    : n x n heatmap per layer (rows = from i, columns = to j)
and which G is used at the current round.
Usage: python -m viz.rounds traces/example.npz
"""


def main():
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()

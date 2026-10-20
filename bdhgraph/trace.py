"""Record what happens at every round, for the visualizations.

For each (letter t, layer l, round r) we keep:
  x, y, a : vectors of size n
  sigma   : n x n, only after the write round (to save space)
Saved as traces/<name>.npz, read by viz/rounds.py and viz/graph3d.py.
"""


class Tracer:
    def __init__(self):
        self.steps = []

    def record(self, t, l, round_name, x, y, a, sigma=None, y_in=None):
        """Keep numpy copies of the first chunk of the batch (x, y, a, y_in: n numbers; sigma: n x n).
        y_in is the y that entered the layer; y is the y that came out."""
        self.steps.append({
            "t": t, "l": l, "round": round_name,
            "x": x[0].detach().cpu().numpy().copy(),
            "y": y[0].detach().cpu().numpy().copy(),
            "a": a[0].detach().cpu().numpy().copy(),
            "y_in": None if y_in is None else y_in[0].detach().cpu().numpy().copy(),
            "sigma": None if sigma is None else sigma[0].detach().cpu().numpy().copy(),
        })

    def save(self, path):
        # TODO: np.savez_compressed(path, ...)
        raise NotImplementedError

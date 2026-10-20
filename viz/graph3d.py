"""3D view of the neuron graph (plotly), saved to figures/graph3d.html.

Neurons = points in 3D (on a sphere, or a spectral layout of Gx + Gy).
Edges:
  Gx_e / Gx_i : Y -> X wires (excitatory / inhibitory colors)
  Gy_e / Gy_i : A -> Y prods
  sigma       : memory connections, thickness = strength (from a trace)
Draw only the strongest edges (top k), or it becomes a hairball.
Usage: python -m viz.graph3d checkpoints/tiny.pt [traces/example.npz]
"""


def main():
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()

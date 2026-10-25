# Is the grokked MLP using a Fourier solution (Gromov 2023 analytic solution; Nanda's "key frequencies")?
# For each hidden neuron, take the DFT over residues of its input weights for a (W1[:, :p]) and b (W1[:, p:]).
# Report: share of each neuron's power in its single strongest frequency (k and p−k folded), the frequency it uses
# for a vs b (same frequency expected), and how many distinct frequencies the network uses.
# Run: <venv>/bin/python -I fourier.py <weights.npz>
import sys, numpy as np
d = np.load(sys.argv[1]); W1, p = d["W1"], int(d["p"])
def spec(M):  # power per folded frequency k = 1..(p-1)/2 (k=0, the mean, excluded)
    F = np.abs(np.fft.rfft(M, axis=1)) ** 2; return F[:, 1:(p - 1) // 2 + 1]
Sa, Sb = spec(W1[:, :p]), spec(W1[:, p:])
ka, kb = Sa.argmax(1) + 1, Sb.argmax(1) + 1
fa = Sa.max(1) / Sa.sum(1); fb = Sb.max(1) / Sb.sum(1)
print(f"p={p} neurons={len(W1)}")
print(f"share of power in the top frequency: median a={np.median(fa):.3f} b={np.median(fb):.3f}; "
      f"neurons with >85% in one frequency: a={np.mean(fa > .85):.2f} b={np.mean(fb > .85):.2f}")
print(f"same top frequency for a and b: {np.mean(ka == kb):.2f}")
u, c = np.unique(ka, return_counts=True); order = np.argsort(-c)
print("frequencies used (k: number of neurons):", ", ".join(f"{u[i]}:{c[i]}" for i in order[:15]), f"| distinct={len(u)} of {(p - 1) // 2}")
r = np.random.default_rng(0).normal(size=W1.shape); Sr = spec(r[:, :p])
print(f"baseline (random weights): median top-frequency share={np.median(Sr.max(1) / Sr.sum(1)):.3f}")

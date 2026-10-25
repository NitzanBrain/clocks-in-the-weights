# Own test: does a 2-layer MLP grok modular addition? Setup after Gromov 2023 / Kumar et al. 2023:
# input = concat one-hot(a), one-hot(b) (D=2p); h = W1 x / sqrt(D); z = act(h); out = W2 z / N; MSE vs one-hot(a+b mod p).
# Full-batch GD, no weight decay (arm "gd"); or full-batch AdamW with weight decay (arm "adamw").
# Run: python3 -I grok.py <p> <train frac> <N> <act: quad|relu> <arm: gd|adamw> <lr> <epochs> [wd] [seed]
import sys, numpy as np
p, frac, N, act, arm, lr, E = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], float(sys.argv[6]), int(sys.argv[7])
wd = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0; seed = int(sys.argv[9]) if len(sys.argv) > 9 else 0
rng = np.random.default_rng(seed); D = 2 * p
a, b = np.meshgrid(np.arange(p), np.arange(p)); a, b = a.ravel(), b.ravel()
X = np.zeros((p * p, D)); X[np.arange(p * p), a] = 1; X[np.arange(p * p), p + b] = 1
Y = np.zeros((p * p, p)); Y[np.arange(p * p), (a + b) % p] = 1
perm = rng.permutation(p * p); ntr = int(frac * p * p); tr, te = perm[:ntr], perm[ntr:]
W1 = rng.normal(0, 1, (N, D)); W2 = rng.normal(0, 1, (p, N))
f = (lambda h: h * h, lambda h: 2 * h) if act == "quad" else (lambda h: np.maximum(h, 0), lambda h: (h > 0) * 1.0)
mo = [np.zeros_like(W1), np.zeros_like(W2)]; vo = [np.zeros_like(W1), np.zeros_like(W2)]
def fwd(idx):
    h = X[idx] @ W1.T / np.sqrt(D); z = f[0](h); return h, z, z @ W2.T / N
def acc(idx): return (fwd(idx)[2].argmax(1) == Y[idx].argmax(1)).mean()
marks = sorted(set(int(x) for x in np.unique(np.logspace(0, np.log10(E), 25).astype(int))) | set(range(E // 100, E + 1, E // 100)))
for t in range(1, E + 1):
    h, z, o = fwd(tr); g = 2 * (o - Y[tr]) / len(tr)               # dL/do, L = mean over samples of sum_k (o-y)^2
    gW2 = g.T @ z / N; gz = g @ W2 / N; gW1 = (gz * f[1](h)).T @ X[tr] / np.sqrt(D)
    if arm == "gd":
        W1 -= lr * gW1; W2 -= lr * gW2
    else:
        for i, (P, G) in enumerate(((W1, gW1), (W2, gW2))):
            mo[i] = 0.9 * mo[i] + 0.1 * G; vo[i] = 0.98 * vo[i] + 0.02 * G * G
            P -= lr * ((mo[i] / (1 - 0.9 ** t)) / (np.sqrt(vo[i] / (1 - 0.98 ** t)) + 1e-8) + wd * P)
    if t in marks:
        print(f"epoch {t:7d} train acc {acc(tr):.3f} test acc {acc(te):.3f} |W1| {np.linalg.norm(W1):.1f} |W2| {np.linalg.norm(W2):.1f}", flush=True)
import os
if os.environ.get("SAVE"):  # weights + split, for the Fourier analysis (fourier.py)
    np.savez(os.environ["SAVE"], W1=W1, W2=W2, tr=tr, te=te, p=p)

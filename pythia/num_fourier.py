# Own test: do Pythia-160m's number-token embeddings carry periodic (Fourier) structure, as reported for other LLMs
# (Zhou et al. 2024; Kantamneni & Tegmark 2025: periods 2, 5, 10, 100)? Numbers 0..499 (all single no-space tokens;
# first gap is 531). DFT over the number index of each embedding dimension (mean removed); power per period
# T = 500/k, summed over dims; compared with the same numbers in shuffled order (null: no periodicity).
# Run: <venv>/bin/python -I num_fourier.py <model dir>
import sys, json, numpy as np
md = sys.argv[1]
from safetensors.numpy import load_file
W = load_file(md + "/model.safetensors"); v = json.load(open(md + "/tokenizer.json"))["model"]["vocab"]
PRE = "Ġ" if sys.argv[2:] == ["space"] else ""  # "space": leading-space tokens, as in Zhou 2024 / Kantamneni 2025
ids = [v[PRE + str(n)] for n in range(500)]; Nn = 500  # 500 so periods 2, 5, 10, 100 fall on exact frequency bins
for name in ("gpt_neox.embed_in.weight", "embed_out.weight"):
    E = W[name][ids].astype(np.float64); E -= E.mean(0)
    P = (np.abs(np.fft.rfft(E, axis=0)) ** 2).sum(1); P /= P[1:].sum()
    rng = np.random.default_rng(0); null = []
    for _ in range(200):
        Es = E[rng.permutation(Nn)]; q = (np.abs(np.fft.rfft(Es, axis=0)) ** 2).sum(1); null.append(q / q[1:].sum())
    null = np.array(null); top = np.argsort(-P[1:])[:8] + 1
    print(f"{name}: top frequencies (k → period 500/k, power share, ×null mean, null 99.5th pct)")
    for k in top:
        print(f"   k={k:3d} T={Nn / k:7.2f}  {P[k]:.4f}  x{P[k] / null[:, k].mean():5.1f}  null99.5={np.quantile(null[:, k], .995):.4f}")
    for T in (2, 5, 10, 100):
        k = round(Nn / T); print(f"   period {T}: k={k} (T={Nn / k:.2f}) share {P[k]:.4f} = x{P[k] / null[:, k].mean():.1f} null")

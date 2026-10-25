# Causal test: does Pythia-160m USE the periodic directions of its number embeddings (run5/run6) for a task it can do?
# Task: continue a counting sequence " 37, 38, 39," → " 40" (n from 20..479, 4 numbers given), greedy, exact match.
# Ablation: in the INPUT embedding of number tokens " 0".." 499" (leading-space, as in the prompt), remove the subspace
# spanned by the period-T Fourier directions u_c = Σ_n E_n cos(2πnT⁻¹), u_s = Σ_n E_n sin(...) (centred E, orthonormalised).
# Control: remove the same number of random directions. Also report "same last digit" accuracy (mod-10 behaviour).
# Run: <venv>/bin/python -I count_ablate.py <model dir>
import sys, numpy as np, torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
md = sys.argv[1]; tok = AutoTokenizer.from_pretrained(md); m = GPTNeoXForCausalLM.from_pretrained(md, torch_dtype=torch.float32).eval()
ids = [tok.convert_tokens_to_ids("Ġ" + str(n)) for n in range(500)]
E0 = m.gpt_neox.embed_in.weight.data.clone(); En = E0[ids].double().numpy(); mu = En.mean(0); C = En - mu
def fourier_dirs(periods):
    V = []
    for T in periods:
        ang = 2 * np.pi * np.arange(500) / T
        V += [np.cos(ang) @ C, np.sin(ang) @ C]
    Q, _ = np.linalg.qr(np.array(V).T); return Q  # D × k orthonormal
def run(Q):
    W = m.gpt_neox.embed_in.weight.data; W.copy_(E0)
    if Q is not None:
        Qt = torch.tensor(Q, dtype=torch.float32); X = E0[ids] - torch.tensor(mu, dtype=torch.float32)
        W[ids] = E0[ids] - (X @ Qt) @ Qt.T  # project out (relative to the mean number embedding)
    rng = np.random.default_rng(0); ok = dig = 0; N = 150
    for n in rng.integers(20, 480, N):
        p = ",".join(f" {k}" for k in range(n - 4, n)) + ","
        with torch.no_grad(): t = m(tok(p, return_tensors="pt")["input_ids"]).logits[0, -1].argmax().item()
        s = tok.decode([t]).strip(); ok += s == str(n); dig += s.isdigit() and int(s) % 10 == n % 10
    return ok / N, dig / N
k_all = None
for name, per in [("none", None), ("T=10", [10]), ("T=100", [100]), ("T=2,5,10,100", [2, 5, 10, 100])]:
    Q = None if per is None else fourier_dirs(per)
    a, d = run(Q); line = f"{name:14s} dirs={0 if Q is None else Q.shape[1]:2d}  exact={a:.3f}  last-digit={d:.3f}"
    if Q is not None:
        R, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(Q.shape[0], Q.shape[1]))); ra, rd = run(R)
        line += f"   | random {R.shape[1]} dirs: exact={ra:.3f} last-digit={rd:.3f}"
    print(line, flush=True)

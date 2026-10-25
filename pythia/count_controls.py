# Stronger controls for the counting ablation (article §5): the 7 periodic directions (T=2 cos; T=5,10,100
# cos+sin) vs (a) 10 independent draws of 7 random directions, (b) the top-7 principal components of the centred
# number embeddings (high-variance, not chosen for periodicity), (c) the principal components 8–14 (next 7).
import sys, numpy as np, torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
md = sys.argv[1]; tok = AutoTokenizer.from_pretrained(md); m = GPTNeoXForCausalLM.from_pretrained(md, torch_dtype=torch.float32).eval()
ids = [tok.convert_tokens_to_ids("Ġ" + str(n)) for n in range(500)]
E0 = m.gpt_neox.embed_in.weight.data.clone(); En = E0[ids].double().numpy(); mu = En.mean(0); C = En - mu
def run(Q, N=150):
    W = m.gpt_neox.embed_in.weight.data; W.copy_(E0)
    Qt = torch.tensor(Q, dtype=torch.float32); W[ids] = E0[ids] - ((E0[ids] - torch.tensor(mu, dtype=torch.float32)) @ Qt) @ Qt.T
    rng = np.random.default_rng(0); ok = 0
    for n in rng.integers(20, 480, N):
        p = ",".join(f" {k}" for k in range(n - 4, n)) + ","
        with torch.no_grad(): t = m(tok(p, return_tensors="pt")["input_ids"]).logits[0, -1].argmax().item()
        ok += tok.decode([t]).strip() == str(n)
    return ok / N
V = []
for T in (2, 5, 10, 100):
    ang = 2 * np.pi * np.arange(500) / T; V.append(np.cos(ang) @ C)
    if T != 2: V.append(np.sin(ang) @ C)
Qp, _ = np.linalg.qr(np.array(V).T); print(f"periodic 7 dirs: {run(Qp):.3f}", flush=True)
U, S, Vt = np.linalg.svd(C, full_matrices=False)
var_p = np.linalg.norm(C @ Qp) ** 2 / np.linalg.norm(C) ** 2
print(f"  (share of number-embedding variance in the periodic subspace: {var_p:.3f}; top-7 PCs hold {(S[:7]**2).sum()/(S**2).sum():.3f})")
print(f"top-7 PCs: {run(Vt[:7].T):.3f}", flush=True); print(f"PCs 8-14: {run(Vt[7:14].T):.3f}", flush=True)
r = [run(np.linalg.qr(np.random.default_rng(s).normal(size=(C.shape[1], 7)))[0]) for s in range(10)]
print(f"random 7 dirs, 10 draws: mean {np.mean(r):.3f}, min {min(r):.3f}, max {max(r):.3f}")

# Follow-up to count_ablate.py: with the period-2/5/10/100 directions removed (T=2: cos only, since sin(πn)=0 —
# fixes an arbitrary extra direction in count_ablate.py), what do the wrong answers look like?
import sys, collections, numpy as np, torch
sys.argv += [] ; md = sys.argv[1]
from transformers import GPTNeoXForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(md); m = GPTNeoXForCausalLM.from_pretrained(md, torch_dtype=torch.float32).eval()
ids = [tok.convert_tokens_to_ids("Ġ" + str(n)) for n in range(500)]
E0 = m.gpt_neox.embed_in.weight.data.clone(); En = E0[ids].double().numpy(); mu = En.mean(0); C = En - mu
V = []
for T in (2, 5, 10, 100):
    ang = 2 * np.pi * np.arange(500) / T; V.append(np.cos(ang) @ C)
    if T != 2: V.append(np.sin(ang) @ C)
Q, _ = np.linalg.qr(np.array(V).T); Qt = torch.tensor(Q, dtype=torch.float32)
W = m.gpt_neox.embed_in.weight.data; W[ids] = E0[ids] - ((E0[ids] - torch.tensor(mu, dtype=torch.float32)) @ Qt) @ Qt.T
rng = np.random.default_rng(0); kinds = collections.Counter(); ok = 0; N = 150; ex = []
for n in rng.integers(20, 480, N):
    p = ",".join(f" {k}" for k in range(n - 4, n)) + ","
    with torch.no_grad(): t = m(tok(p, return_tensors="pt")["input_ids"]).logits[0, -1].argmax().item()
    s = tok.decode([t]).strip()
    if s == str(n): ok += 1; continue
    if not s.isdigit(): kinds["non-number"] += 1; ex.append((int(n), s)); continue
    d = int(s) - n; kinds[f"off by {d:+d}" if abs(d) < 12 or d % 10 == 0 else "other"] += 1; ex.append((int(n), s))
print(f"dirs={Q.shape[1]} exact={ok / N:.3f}; errors: {dict(kinds.most_common())}"); print("examples (target, output):", ex[:15])

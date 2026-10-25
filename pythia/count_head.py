# Is the successor head L5H9 (run13: succession score 0.636; next best 0.111) needed for counting? Zero its output
# (forward hook on the attention output before the dense projection), compare with 10 random other heads (one at a time)
# and with the mean over them. Same 150 counting prompts as count_ablate.py.
import sys, numpy as np, torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
md = sys.argv[1]; tok = AutoTokenizer.from_pretrained(md); m = GPTNeoXForCausalLM.from_pretrained(md, torch_dtype=torch.float32, attn_implementation="eager").eval()
hd = 64
def run(l=None, h=None, N=150):
    hook = None
    if l is not None:
        def pre(mod, args):
            x = args[0].clone(); x[..., h * hd:(h + 1) * hd] = 0; return (x,) + args[1:]
        hook = m.gpt_neox.layers[l].attention.dense.register_forward_pre_hook(pre)
    rng = np.random.default_rng(0); ok = 0; offs = []
    for n in rng.integers(20, 480, N):
        p = ",".join(f" {k}" for k in range(n - 4, n)) + ","
        with torch.no_grad(): t = m(tok(p, return_tensors="pt")["input_ids"]).logits[0, -1].argmax().item()
        s = tok.decode([t]).strip(); ok += s == str(n)
        if s.isdigit() and s != str(n): offs.append(int(s) - n)
    if hook: hook.remove()
    return ok / N, offs
a, _ = run(); print(f"none: {a:.3f}")
a, offs = run(5, 9); print(f"L5H9 zeroed: {a:.3f}; error offsets: {sorted(set(offs))[:12]}, -1 count {offs.count(-1)}, +10 count {offs.count(10)}")
rng = np.random.default_rng(3); r = []
for _ in range(10):
    l, h = int(rng.integers(0, 12)), int(rng.integers(0, 12))
    if (l, h) == (5, 9): continue
    r.append((f"L{l}H{h}", run(l, h)[0]))
print("random single heads:", r, "mean", round(float(np.mean([x for _, x in r])), 3))

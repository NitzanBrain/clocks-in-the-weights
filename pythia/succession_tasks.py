# Gould et al.'s other succession tasks on Pythia-160m: days, months, letters, ordinal words. Same direct-path score
# as succession.py, but for each task the argmax is over that task's tokens only (leading-space forms; tasks where a
# token isn't a single token are reported and skipped). Which heads score highest per task, and is L5H9 general?
import sys, json, math, numpy as np
from safetensors.numpy import load_file
md = sys.argv[1]; W = {k: v.astype(np.float64) for k, v in load_file(md + "/model.safetensors").items() if v.dtype != np.uint8}
cfg = json.load(open(md + "/config.json")); D, nh, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]; hd = D // nh
voc = json.load(open(md + "/tokenizer.json"))["model"]["vocab"]
def ln(x, p):
    m = x.mean(-1, keepdims=True); s = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(s + 1e-5) * W[p + ".weight"] + W[p + ".bias"]
erf = np.vectorize(math.erf)
tasks = {"days": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
         "months": "January February March April May June July August September October November December".split(),
         "letters": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
         "ordinals": "first second third fourth fifth sixth seventh eighth ninth tenth".split(),
         "numbers 0-99": [str(n) for n in range(100)]}
p0 = "gpt_neox.layers.0."
for name, words in tasks.items():
    toks = ["Ġ" + w for w in words]; missing = [t for t in toks if t not in voc]
    if missing: print(f"{name}: not all single tokens (missing {len(missing)}: {missing[:3]}) → skipped"); continue
    ids = [voc[t] for t in toks]; E = W["gpt_neox.embed_in.weight"][ids]
    f = ln(E, p0 + "post_attention_layernorm") @ W[p0 + "mlp.dense_h_to_4h.weight"].T + W[p0 + "mlp.dense_h_to_4h.bias"]
    f = 0.5 * f * (1 + erf(f / np.sqrt(2))); e = E + f @ W[p0 + "mlp.dense_4h_to_h.weight"].T + W[p0 + "mlp.dense_4h_to_h.bias"]
    U = W["embed_out.weight"][ids]; n = len(ids); sc = np.zeros((L, nh))
    for l in range(L):
        p = f"gpt_neox.layers.{l}."; x = ln(e, p + "input_layernorm")
        Wqkv = W[p + "attention.query_key_value.weight"].reshape(nh, 3 * hd, D); Wo = W[p + "attention.dense.weight"]
        for h in range(nh):
            o = (x @ Wqkv[h, 2 * hd:].T) @ Wo[:, h * hd:(h + 1) * hd].T
            sc[l, h] = ((o @ U.T)[:n - 1].argmax(1) == np.arange(1, n)).mean()
    top = sorted(((sc[l, h], l, h) for l in range(L) for h in range(nh)), reverse=True)[:4]
    print(f"{name:13s} (n={n}): L5H9={sc[5, 9]:.2f}; top: " + ", ".join(f"L{l}H{h}={s:.2f}" for s, l, h in top) + f"; chance≈{1 / n:.2f}")

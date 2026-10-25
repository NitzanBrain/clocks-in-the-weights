# Succession score (Gould et al. 2023, arXiv 2312.09230) for all 144 heads of Pythia-160m, restricted to
# number tokens " 0".." 99": effective embedding e_t = E_t + MLP0(LN_post0(E_t)) (MLP0 on the token alone; Pythia's
# parallel residual means MLP0 reads LN(E_t)), then head OV: o = LN_in_l(e_t) @ W_V^h @ W_O^h, logits = o @ U^T over
# the 100 number tokens. Score = fraction of t in 0..98 whose top logit is t+1.
import sys, json, math, numpy as np
from safetensors.numpy import load_file
md = sys.argv[1]; W = {k: v.astype(np.float64) for k, v in load_file(md + "/model.safetensors").items() if v.dtype != np.uint8}
cfg = json.load(open(md + "/config.json")); D, nh, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]; hd = D // nh
v = json.load(open(md + "/tokenizer.json"))["model"]["vocab"]; ids = [v["Ġ" + str(n)] for n in range(100)]
def ln(x, p):
    m = x.mean(-1, keepdims=True); s = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(s + 1e-5) * W[p + ".weight"] + W[p + ".bias"]
erf = np.vectorize(math.erf)
E = W["gpt_neox.embed_in.weight"][ids]; p0 = "gpt_neox.layers.0."
f = ln(E, p0 + "post_attention_layernorm") @ W[p0 + "mlp.dense_h_to_4h.weight"].T + W[p0 + "mlp.dense_h_to_4h.bias"]
f = 0.5 * f * (1 + erf(f / np.sqrt(2))); e = E + f @ W[p0 + "mlp.dense_4h_to_h.weight"].T + W[p0 + "mlp.dense_4h_to_h.bias"]
U = W["embed_out.weight"][ids]; scores = np.zeros((L, nh))
for l in range(L):
    p = f"gpt_neox.layers.{l}."; x = ln(e, p + "input_layernorm")
    Wqkv = W[p + "attention.query_key_value.weight"].reshape(nh, 3 * hd, D); Wo = W[p + "attention.dense.weight"]
    for h in range(nh):
        Wv = Wqkv[h, 2 * hd:]; o = (x @ Wv.T) @ Wo[:, h * hd:(h + 1) * hd].T
        lg = o @ U.T; scores[l, h] = (lg[:99].argmax(1) == np.arange(1, 100)).mean()
np.set_printoptions(precision=2, suppress=True, linewidth=140); print("succession score [layer x head]:\n", scores)
top = sorted(((scores[l, h], l, h) for l in range(L) for h in range(nh)), reverse=True)[:6]
print("top heads:", [(f"L{l}H{h}", round(float(s), 3)) for s, l, h in top])

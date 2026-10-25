# Link §8 and §9: does the successor head L5H9 READ the periodic directions? Recompute its succession score after
# removing (a) the 7 periodic directions, (b) 7 high-variance non-periodic directions (orthogonal), (c) 7 random
# directions from the number embeddings " 0".." 99" (relative to their mean, as in count_controls*.py; directions
# estimated on 0..499).
import sys, numpy as np, pathlib
src = open(pathlib.Path(__file__).parent / "succession.py").read()
exec(src[:src.index("E = W[\"gpt_neox.embed_in.weight\"][ids]")])  # loads W, ln, erf, ids (0..99), cfg
v500 = [v["Ġ" + str(n)] for n in range(500)]; E5 = W["gpt_neox.embed_in.weight"][v500]; mu = E5.mean(0); C = E5 - mu
V = []
for T in (2, 5, 10, 100):
    ang = 2 * np.pi * np.arange(500) / T; V.append(np.cos(ang) @ C)
    if T != 2: V.append(np.sin(ang) @ C)
Qp, _ = np.linalg.qr(np.array(V).T); R = C - (C @ Qp) @ Qp.T; Qn = np.linalg.svd(R, full_matrices=False)[2][:7].T
Qr, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(C.shape[1], 7)))
p0 = "gpt_neox.layers.0."; p5 = "gpt_neox.layers.5."; U = W["embed_out.weight"][ids]
Wqkv = W[p5 + "attention.query_key_value.weight"].reshape(nh, 3 * hd, D); Wo = W[p5 + "attention.dense.weight"]
def score(Q, h=9):
    E = W["gpt_neox.embed_in.weight"][ids].copy()
    if Q is not None: E = E - ((E - mu) @ Q) @ Q.T
    f = ln(E, p0 + "post_attention_layernorm") @ W[p0 + "mlp.dense_h_to_4h.weight"].T + W[p0 + "mlp.dense_h_to_4h.bias"]
    f = 0.5 * f * (1 + erf(f / np.sqrt(2))); e = E + f @ W[p0 + "mlp.dense_4h_to_h.weight"].T + W[p0 + "mlp.dense_4h_to_h.bias"]
    o = (ln(e, p5 + "input_layernorm") @ Wqkv[h, 2 * hd:].T) @ Wo[:, h * hd:(h + 1) * hd].T
    return ((o @ U.T)[:99].argmax(1) == np.arange(1, 100)).mean()
for name, Q in [("none", None), ("periodic 7", Qp), ("non-periodic high-variance 7", Qn), ("random 7", Qr)]:
    print(f"L5H9 succession score, removed {name:30s}: {score(Q):.3f}")

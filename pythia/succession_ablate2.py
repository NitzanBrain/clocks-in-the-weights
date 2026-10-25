# Control distribution: L5H9 succession score after removing
# (a) 20 random draws of 7 directions, (b) 20 random 7-dim subspaces INSIDE the non-periodic high-variance space
#     (random rotations of the top-30 PCs of the embeddings with the periodic subspace projected out), which carry
#     variance comparable to the periodic set. Reports mean, min, max, and each set's variance share.
import sys, numpy as np, pathlib
src = open(pathlib.Path(__file__).parent / "succession_ablate.py").read()
exec(src[:src.index("for name, Q in")])
var = lambda Q: np.linalg.norm(C @ Q) ** 2 / np.linalg.norm(C) ** 2
P30 = np.linalg.svd(R, full_matrices=False)[2][:30].T  # top-30 non-periodic PCs (orthogonal to periodic)
print(f"periodic 7: score {score(Qp):.3f}, variance {var(Qp):.3f}")
for name, gen in [("random 7 (whole space)", lambda g: np.linalg.qr(g.normal(size=(C.shape[1], 7)))[0]),
                  ("random 7 inside non-periodic top-30", lambda g: P30 @ np.linalg.qr(g.normal(size=(30, 7)))[0])]:
    res = [(score(Q), var(Q)) for Q in (gen(np.random.default_rng(s)) for s in range(20))]
    sc = np.array([r[0] for r in res]); vv = np.array([r[1] for r in res])
    print(f"{name}: score mean {sc.mean():.3f} [min {sc.min():.3f}, max {sc.max():.3f}]; variance mean {vv.mean():.3f} [{vv.min():.3f}, {vv.max():.3f}]")

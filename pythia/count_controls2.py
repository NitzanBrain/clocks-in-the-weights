# Cleaner control (run11b showed the top PCs overlap the periodic subspace): remove the top-7 principal
# components of the number embeddings AFTER projecting out the periodic subspace, i.e. 7 high-variance directions
# orthogonal to the periodic ones. Also: periodic 7 + those 7 together.
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent)); src = open(pathlib.Path(__file__).parent / "count_controls.py").read()
exec(src[:src.index("V = []")])  # model, ids, E0, mu, C, run()
V = []
for T in (2, 5, 10, 100):
    ang = 2 * np.pi * np.arange(500) / T; V.append(np.cos(ang) @ C)
    if T != 2: V.append(np.sin(ang) @ C)
Qp, _ = np.linalg.qr(np.array(V).T)
R = C - (C @ Qp) @ Qp.T; U, S, Vt = np.linalg.svd(R, full_matrices=False); Qn = Vt[:7].T
var = lambda Q: np.linalg.norm(C @ Q) ** 2 / np.linalg.norm(C) ** 2
print(f"non-periodic top-7 (orthogonal to periodic): variance share {var(Qn):.3f} (periodic: {var(Qp):.3f}); max cos with periodic {np.abs(Qp.T @ Qn).max():.3f}")
print(f"  counting exact: {run(Qn):.3f}", flush=True)
Qb, _ = np.linalg.qr(np.hstack([Qp, Qn])); print(f"periodic + non-periodic (14 dirs): {run(Qb):.3f}")

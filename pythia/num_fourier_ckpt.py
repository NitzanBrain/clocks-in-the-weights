# Same test across Pythia-160m training checkpoints, fetching ONLY the two embedding tensors via HTTP range requests
# on the safetensors file (no full 650MB download). Prints the ×null ratio for periods 2, 5, 10, 100.
# Run: <venv>/bin/python -I num_fourier_ckpt.py <tokenizer.json> step1 step2 ...
import sys, json, struct, urllib.request, numpy as np
from concurrent.futures import ThreadPoolExecutor
v = json.load(open(sys.argv[1]))["model"]["vocab"]; ids = np.array([v[str(n)] for n in range(500)]); Nn = 500
def rng_get(url, a, b, tries=8):
    import time, urllib.error
    for t in range(tries):  # back off politely on HTTP 429 (rate limit) instead of failing
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"}); return urllib.request.urlopen(r).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or t == tries - 1: raise
            time.sleep(60 * (t + 1))
        except Exception:  # e.g. http.client.IncompleteRead on a 150MB body through the proxy
            if t == tries - 1: raise
            time.sleep(10)
def ratio(E):
    E = E - E.mean(0); P = (np.abs(np.fft.rfft(E, axis=0)) ** 2).sum(1); P /= P[1:].sum(); g = np.random.default_rng(0); nl = []
    for _ in range(100):
        q = (np.abs(np.fft.rfft(E[g.permutation(Nn)], axis=0)) ** 2).sum(1); nl.append(q / q[1:].sum())
    nl = np.array(nl).mean(0); return {T: P[Nn // T] / nl[Nn // T] for T in (2, 5, 10, 100)}
for st in sys.argv[2:]:
    url = f"https://huggingface.co/EleutherAI/pythia-160m/resolve/{'50f5173d932e8e61f858120bcb800b97af589f46' if st == 'main' else 'step' + st}/model.safetensors"
    n = struct.unpack("<Q", rng_get(url, 0, 8))[0]; hdr = json.loads(rng_get(url, 8, 8 + n)); out = []
    for name in ("gpt_neox.embed_in.weight", "embed_out.weight"):
        h = hdr[name]; a, b = h["data_offsets"]; dt = {"F16": np.float16, "F32": np.float32}[h["dtype"]]
        rows = h["shape"][1]; isz = np.dtype(dt).itemsize
        # ONE range request for the whole tensor (~150MB): row-by-row requests got HTTP 429 from the Hub
        full = np.frombuffer(rng_get(url, 8 + n + a, 8 + n + b), dt).reshape(h["shape"])
        E = full[ids].astype(np.float64); del full
        out.append(ratio(E))
    print(f"step {st:>6}: in " + " ".join(f"T{T}=x{out[0][T]:.1f}" for T in (2, 5, 10, 100)) + " | out " + " ".join(f"T{T}=x{out[1][T]:.1f}" for T in (2, 5, 10, 100)), flush=True)

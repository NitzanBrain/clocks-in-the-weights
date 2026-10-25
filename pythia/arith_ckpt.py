# Does Pythia-160m's ARITHMETIC ability appear together with its periodic number code (run6)?
# Few-shot 2-digit addition "a + b = c" (a, b in 10..49, so c < 100 is a single token), greedy next token, exact match.
# Also the parity of the predicted answer (is the units digit's parity right even when the answer is wrong?).
# One checkpoint at a time (to keep memory low), full HF model in float32.
# Run: python arith_ckpt.py step1 ... (xet cache needs a writable HF_HOME)
import sys, os, shutil, numpy as np, torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision="50f5173d932e8e61f858120bcb800b97af589f46")
rng = np.random.default_rng(0); shots = [(23, 14), (31, 25), (12, 40), (44, 18)]
tests = [(int(a), int(b)) for a, b in rng.integers(10, 50, (200, 2))]
prefix = "".join(f"{a} + {b} = {a + b}\n" for a, b in shots)
for st in sys.argv[1:]:
    rev = "50f5173d932e8e61f858120bcb800b97af589f46" if st == "main" else f"step{st}"  # main pinned to its commit
    m = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", revision=rev, torch_dtype=torch.float32,
                                           cache_dir=os.environ.get("WORK", "work") + "/ck-arith").eval()
    ok = par = 0
    for a, b in tests:
        ids = tok(prefix + f"{a} + {b} =", return_tensors="pt")["input_ids"]
        with torch.no_grad(): nxt = m(ids).logits[0, -1].argmax().item()
        s = tok.decode([nxt]).strip(); ok += s == str(a + b)
        par += s.isdigit() and int(s) % 2 == (a + b) % 2
    print(f"step {st:>6}: exact {ok / len(tests):.3f}  parity-correct {par / len(tests):.3f}", flush=True)
    del m; shutil.rmtree(os.environ.get("WORK", "work") + "/ck-arith", ignore_errors=True)

# When does the successor head appear? Max succession score over all 144 heads (and L5H9's own score) across
# Pythia-160m checkpoints, using succession.py's number-only variant. Downloads each checkpoint's safetensors to
# $WORK/ck-succ (default ./work) (one at a time), deletes after.
import sys, os, json, shutil, subprocess, numpy as np, pathlib
here = pathlib.Path(__file__).parent; py = sys.executable; base = os.environ.get("WORK", "work")
for st in sys.argv[1:]:
    d = pathlib.Path(base) / "ck-succ"; d.mkdir(exist_ok=True)
    for f in ("config.json", "tokenizer.json"): shutil.copy(pathlib.Path(base) / "p160" / f, d / f)
    rev = "50f5173d932e8e61f858120bcb800b97af589f46" if st == "main" else f"step{st}"  # main pinned to its commit
    subprocess.run(["curl", "-sL", "--retry", "5", "-o", str(d / "model.safetensors"),
                    f"https://huggingface.co/EleutherAI/pythia-160m/resolve/{rev}/model.safetensors"], check=True)
    out = subprocess.run([py, "-I", str(here / "succession.py"), str(d)], capture_output=True, text=True).stdout
    top = [l for l in out.splitlines() if l.startswith("top heads")][0]
    rows = [l for l in out.splitlines() if l.strip().startswith("[") or l.strip().startswith("[[")]
    print(f"step {st:>6}: {top}", flush=True)
    os.remove(d / "model.safetensors")

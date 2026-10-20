"""Training dashboard: launch runs, watch them live, look at past runs, try a model, see the networks.
Run from the repo root: python3 viewer/app.py   then open http://localhost:8000 (it opens by itself).

It reads checkpoints/runs/*.jsonl (the loss, written by scripts/train.py) and
checkpoints/runs/*.pt (the weights, same name). torch is only loaded when you look at a model.
"Start run" simply runs `python -m scripts.train ...` as a separate process.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RUNS_DIR = os.path.join(ROOT, "checkpoints", "runs")
LAUNCH_LOG = os.path.join(ROOT, "checkpoints", "last_launch.log")  # what the last launched run printed
STALE_AFTER = 30  # seconds without a new line: the run is not live any more
MAX_NEW = 40      # the model writes at most this many letters
MAX_TOTAL = 96    # prompt + answer, same as T in training

sys.path.insert(0, ROOT)
from bdhgraph.config import TINY  # only a dataclass: no torch needed for this

try:
    from plotly.offline import get_plotlyjs  # plotly.js is bundled in the python package

    PLOTLY_JS = get_plotlyjs()
except ImportError:
    PLOTLY_JS = None  # the page then loads plotly.js from the internet


# ---------- runs (loss curves) ----------

def run_path(run_id, ext):
    return os.path.join(RUNS_DIR, run_id + ext)


def read_run(run_id):
    """All records of one run file. A half-written last line is skipped."""
    records = []
    with open(run_path(run_id, ".jsonl")) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def list_runs():
    """One summary per run, newest first."""
    runs = []
    if not os.path.isdir(RUNS_DIR):
        return runs
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        if not name.endswith(".jsonl"):
            continue
        run_id = name[: -len(".jsonl")]
        records = read_run(run_id)
        start = next((r for r in records if r["type"] == "start"), {})
        steps = [r for r in records if r["type"] == "step"]
        vals = [r["loss"] for r in records if r["type"] == "val"]
        age = time.time() - os.path.getmtime(run_path(run_id, ".jsonl"))
        done = any(r["type"] == "done" for r in records)
        recent = steps[-10:]
        runs.append({
            "id": run_id,
            "lr": start.get("lr"),
            "n": start.get("n"),
            "L": start.get("L"),
            "steps_total": start.get("steps"),
            "steps_done": len(steps),
            "train": sum(r["loss"] for r in recent) / len(recent) if recent else None,
            "best_val": min(vals) if vals else None,
            "status": "finished" if done else ("live" if age < STALE_AFTER else "stopped"),
            "has_model": os.path.exists(run_path(run_id, ".pt")),
        })
    return runs


# ---------- launching a run ----------

PROC = None          # the training process started from this page
PROC_PARAMS = None   # its parameters, for the command shown on the page

# name -> (type, smallest, largest). The values are checked before anything is started.
LIMITS = {
    "steps": (int, 1, 20000), "lr": (float, 1e-5, 1.0), "n": (int, 8, 256), "L": (int, 1, 8),
    "u": (float, 0.5, 1.0), "batch": (int, 1, 64), "seed": (int, 0, 1_000_000),
    "gs_density": (float, 0.05, 1.0),
}


def available_levels():
    return [lvl for lvl in (1, 2, 3) if os.path.exists(os.path.join(ROOT, "data", f"toyworld_l{lvl}_train.txt"))]


def launch_config():
    return {
        "defaults": {"steps": 200, "lr": 0.1, "level": TINY.level, "n": TINY.n, "L": TINY.L, "u": TINY.u,
                     "batch": TINY.batch, "seed": TINY.seed, "gs": TINY.gs, "gs_density": TINY.gs_density,
                     "norm_a": TINY.norm_a},
        "limits": {k: [lo, hi] for k, (_, lo, hi) in LIMITS.items()},
        "levels": available_levels(),
    }


def build_command(params):
    """Check the parameters and turn them into the command line. Returns (command, error)."""
    cmd = [sys.executable, "-m", "scripts.train"]
    for name, (kind, lo, hi) in LIMITS.items():
        try:
            value = kind(params[name])
        except (KeyError, TypeError, ValueError):
            return None, f"{name}: a number is needed"
        if not lo <= value <= hi:
            return None, f"{name}: must be between {lo} and {hi}"
        cmd += [f"--{name}", str(value)]
    if params.get("level") not in available_levels():
        return None, f"level {params.get('level')}: no data yet (python -m scripts.make_toyworld --level {params.get('level')})"
    if params.get("gs") not in ("dense", "random"):
        return None, "gs: dense or random"
    cmd += ["--level", str(params["level"]), "--gs", params["gs"]]
    if not params.get("norm_a", True):
        cmd.append("--no_norm_a")
    return cmd, None


def running():
    return PROC is not None and PROC.poll() is None


def launch(params):
    """Start scripts.train as a separate process. Returns a dict for the page."""
    global PROC, PROC_PARAMS
    if running():
        return {"error": "a run is already going: stop it first"}, 409
    cmd, error = build_command(params)
    if error:
        return {"error": error}, 400
    log = open(LAUNCH_LOG, "w")
    # a new session: closing the dashboard (Ctrl+C) does not kill the training
    PROC = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PROC_PARAMS = params
    time.sleep(2.5)  # a bad setting (for example a story longer than T) fails right away
    if PROC.poll() not in (None, 0):
        with open(LAUNCH_LOG) as f:
            tail = f.read().strip().splitlines()[-4:]
        return {"error": "the run stopped at once: " + " | ".join(tail)}, 400
    return {"ok": True, "command": " ".join(["python", "-m", "scripts.train"] + cmd[3:])}, 200


def stop():
    if running():
        PROC.terminate()
        return {"ok": True}
    return {"ok": False, "error": "nothing to stop (this page did not start a run, or it is over)"}


# ---------- looking at a model ----------

LOCK = threading.Lock()  # one model computation at a time (generate briefly switches Gs off)
MODELS = {}              # run id -> (model, step it was saved at)


def safe_id(raw):
    run_id = os.path.basename(raw)  # no ../ tricks
    return run_id if os.path.exists(run_path(run_id, ".jsonl")) else None


def run_level(run_id):
    start = next((r for r in read_run(run_id) if r["type"] == "start"), {})
    return start.get("level", 1)


def read_lines(level, split):
    with open(os.path.join(ROOT, "data", f"toyworld_l{level}_{split}.txt")) as f:
        return f.read().splitlines()


def info(run_id):
    """What the page needs to help typing: allowed characters, words, example prompts."""
    level = run_level(run_id)
    train, val = read_lines(level, "train"), read_lines(level, "val")
    allowed = "".join(sorted(set("".join(train + val))))
    words = sorted(set(re.findall(r"[a-z]+", " ".join(train))) - {"s"})  # "s" comes from "tom's"
    rng = random.Random(0)
    examples = []
    for story in rng.sample(val, 5):
        prompt, answer = story.rsplit(" ", 1)  # the story up to its last word
        examples.append({"prompt": prompt + " ", "answer": answer.rstrip(".")})
    return {"has_model": os.path.exists(run_path(run_id, ".pt")), "level": level,
            "allowed": allowed, "words": words, "examples": examples, "max_prompt": MAX_TOTAL - 1}


def show(byte):
    """A readable name for one byte, for the probability bars."""
    return {32: "space", 10: "end of story"}.get(byte, chr(byte) if 32 < byte < 127 else f"byte {byte}")


def get_model(run_id):
    """Load the saved model once. Call it while holding LOCK."""
    import torch  # only now: the dashboard works without torch until you look at a model

    from bdhgraph.config import Config
    from bdhgraph.model import BDHGraph

    if run_id not in MODELS:
        ckpt = torch.load(run_path(run_id, ".pt"), weights_only=True)
        model = BDHGraph(Config(**ckpt["cfg"]))
        model.load_state_dict(ckpt["state"])
        model.eval()
        MODELS[run_id] = (model, ckpt["step"])
    return MODELS[run_id]


def check_prompt(run_id, prompt):
    """An error message, or None if the model can read this text."""
    if not run_id or not os.path.exists(run_path(run_id, ".pt")):
        return "this run has no saved model (train a new run)"
    bad = sorted({c for c in prompt if c not in info(run_id)["allowed"]})
    if bad:
        return f"not allowed: {' '.join(repr(c) for c in bad)}"
    if not prompt:
        return "type something first"
    if len(prompt) >= MAX_TOTAL:
        return f"too long: at most {MAX_TOTAL - 1} letters"
    return None


def generate(run_id, prompt, memory_on):
    """Greedy completion: always take the most likely next letter, until the story ends."""
    import torch

    with LOCK:
        model, step = get_model(run_id)
        ids = list(prompt.encode())
        out, top = [], []
        saved_gs = model.Gs.clone()
        if not memory_on:
            model.Gs.zero_()  # no memory: the model only sees the current letter
        try:
            with torch.no_grad():
                for i in range(MAX_NEW):
                    logits, _ = model(torch.tensor([ids]))
                    probs = logits[0, -1].softmax(-1)
                    if i == 0:  # the 5 most likely first letters of the answer
                        p, b = probs.topk(5)
                        top = [{"char": show(int(k)), "p": float(v)} for v, k in zip(p, b)]
                    nxt = int(probs.argmax())
                    ids.append(nxt)
                    out.append(nxt)
                    if nxt == 10 or len(ids) >= MAX_TOTAL:
                        break
        finally:
            model.Gs.copy_(saved_gs)
    return {"completion": bytes(out).decode("utf-8", errors="replace"), "top": top, "step": step}


def weights(run_id):
    """The 4 trained G's and the Gs mask, as lists of numbers (for the heatmaps and the 3D graph)."""
    with LOCK:
        model, step = get_model(run_id)
        mats = {name: getattr(model, name).detach().numpy().round(4).tolist()
                for name in ("Gx_e", "Gx_i", "Gy_e", "Gy_i")}
        mats["Gs"] = model.Gs.numpy().round(4).tolist()
    return {"n": model.cfg.n, "L": model.cfg.L, "step": step, "mats": mats}


def trace(run_id, prompt, memory_on):
    """x, y, a of every neuron at every letter and layer, and sigma at the end of the text."""
    import torch

    from bdhgraph.trace import Tracer

    with LOCK:
        model, step = get_model(run_id)
        saved_gs = model.Gs.clone()
        if not memory_on:
            model.Gs.zero_()
        tracer = Tracer()
        try:
            with torch.no_grad():
                model(torch.tensor([list(prompt.encode())]), tracer=tracer)
        finally:
            model.Gs.copy_(saved_gs)
    L, T = model.cfg.L, len(prompt)
    # what the output head would predict from x at every (letter, layer); after the last layer it is the real prediction
    with torch.no_grad():
        xs = torch.from_numpy(np.stack([st["x"] for st in tracer.steps]))  # row t * L + l
        probs = model.head(xs).softmax(-1)
    top_p, top_b = probs.topk(5, dim=-1)
    ids = list(prompt.encode())
    pred = [[[[int(b), round(float(p), 4)] for b, p in zip(top_b[t * L + l], top_p[t * L + l])] for t in range(T)] for l in range(L)]
    p_next = [[round(float(probs[t * L + l, ids[t + 1]]), 4) if t + 1 < T else None for t in range(T)] for l in range(L)]
    pick = lambda key: [[tracer.steps[t * L + l][key].round(4).tolist() for t in range(T)] for l in range(L)]
    sigma = [tracer.steps[(T - 1) * L + l]["sigma"].round(4).tolist() for l in range(L)]
    return {"letters": list(prompt), "L": L, "n": model.cfg.n, "x": pick("x"), "y": pick("y"),
            "a": pick("a"), "y_in": pick("y_in"), "pred": pred, "p_next": p_next, "sigma": sigma, "step": step}


# ---------- the server ----------

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        run_id = safe_id(query.get("run", ""))
        if url.path == "/runs":
            self.send_json(list_runs())
        elif url.path == "/data":
            self.send_json(read_run(run_id) if run_id else [], 200 if run_id else 404)
        elif url.path == "/info":
            self.send_json(info(run_id) if run_id else {"error": "unknown run"}, 200 if run_id else 404)
        elif url.path == "/config":
            self.send_json(launch_config())
        elif url.path == "/status":
            self.send_json({"running": running(), "params": PROC_PARAMS if running() else None})
        elif url.path in ("/generate", "/trace", "/weights"):
            self.model_request(url.path, run_id, query)
        elif url.path == "/plotly.min.js" and PLOTLY_JS:
            self.send(PLOTLY_JS, "application/javascript")
        elif url.path == "/":
            with open(os.path.join(HERE, "index.html")) as f:
                html = f.read()
            if PLOTLY_JS is None:
                html = html.replace("/plotly.min.js", "https://cdn.plot.ly/plotly-2.35.2.min.js")
            self.send(html, "text/html")
        else:
            self.send("not found", "text/plain", 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/launch":
            try:
                length = int(self.headers.get("Content-Length", 0))
                params = json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError):
                return self.send_json({"error": "bad request"}, 400)
            body, code = launch(params)
            self.send_json(body, code)
        elif path == "/stop":
            self.send_json(stop())
        else:
            self.send("not found", "text/plain", 404)

    def model_request(self, path, run_id, query):
        prompt, memory_on = query.get("prompt", ""), query.get("memory", "1") == "1"
        if not run_id or not os.path.exists(run_path(run_id, ".pt")):
            return self.send_json({"error": "this run has no saved model (train a new run)"}, 400)
        if path != "/weights":
            error = check_prompt(run_id, prompt)
            if error:
                return self.send_json({"error": error}, 400)
        try:
            if path == "/generate":
                self.send_json(generate(run_id, prompt, memory_on))
            elif path == "/trace":
                self.send_json(trace(run_id, prompt, memory_on))
            else:
                self.send_json(weights(run_id))
        except ImportError:
            self.send_json({"error": "torch not found: start the dashboard from the venv (source .venv/bin/activate)"}, 500)
        except Exception as e:  # a broken checkpoint, for example
            self.send_json({"error": f"could not run the model: {e}"}, 500)

    def send_json(self, obj, code=200):
        self.send(json.dumps(obj), "application/json", code)

    def send(self, body, content_type, code=200):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass  # keep the terminal quiet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"
    print(f"dashboard on {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    ThreadingHTTPServer(("localhost", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

"""Loopback-only CPU chat playground for a verified Prompt-3 100M checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from src.config import load_config
from src.model import LanguageModel

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "dense": ("dense75_ref", "configs/prompt3/dense75_ref.yaml"),
    "sparse": ("sparse75", "configs/prompt3/sparse75.yaml"),
    "sparse10m": ("sparse75_ngram10m", "configs/prompt3/sparse75_ngram10m.yaml"),
    "sparse25m": ("sparse75_ngram25m", "configs/prompt3/sparse75_ngram25m.yaml"),
}
parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=VARIANTS)
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
tag, config_path = VARIANTS[args.variant]
directory = ROOT / "experiments/prompt3_v1" / tag
checkpoint = directory / "checkpoints/last.pt"
summary = json.loads((directory / "summary_100M.json").read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


assert sha(checkpoint) == summary["checkpoint_sha256"], "checkpoint hash mismatch"
assert summary["actual_tokens"] == 100_007_936, "checkpoint is not the 100M gate"
cfg = load_config(ROOT / config_path)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model = LanguageModel(cfg).cpu().eval()
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert state["metadata"]["tokens_seen"] == 100_007_936
model.load_state_dict(state["model"], strict=True)
del state
tokenizer = Tokenizer.from_file(str(ROOT / cfg["data"]["tokenizer"]))
page = (ROOT / "web/local_dense_chat.html").read_text(encoding="utf-8")
label = args.variant.replace("sparse", "Sparse").replace("dense", "Dense")
page = page.replace("LOCAL · CPU · 50M TOKEN CHECKPOINT", f"LOCAL · CPU · PROMPT-3 {label.upper()} · 100M TOKENS")
page = page.replace("Dense model playground", f"Prompt-3 {label} playground")
page = page.replace("Your actual DenseCompute checkpoint. This is a small base model, with no chat instruction tuning. Try a short code prefix or a simple conversation.", f"Prompt-3 {label} research checkpoint at 100M training tokens. Base model, with no chat instruction tuning; try a short code prefix or a simple conversation.")
page = page.replace("message('DenseCompute','')", f"message('Prompt-3 {label}','')")
page = page.replace("Training continues on GPU.", "Training continues independently on GPU.")
PAGE = page.encode("utf-8")
LOCK = threading.Lock()
info = {
    "model": f"Prompt-3 {label} · 100M tokens",
    "tokens": 100_007_936,
    "parameters": summary["stored_params"],
    "device": "CPU",
    "threads": 1,
    "context": 1024,
    "checkpoint_sha256": summary["checkpoint_sha256"],
    "type": "Base language model; no chat/instruction fine-tuning",
}


def next_logits(ids: torch.Tensor) -> torch.Tensor:
    hidden = model.embedding(ids)
    for block in model.blocks:
        hidden, _ = block(hidden)
    return F.linear(model.norm(hidden[:, -1]), model.embedding.weight)[0]


class Handler(BaseHTTPRequestHandler):
    def reply(self, status: int, body: dict | bytes, kind: str = "application/json; charset=utf-8") -> None:
        if isinstance(body, dict):
            body = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.reply(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/info":
            self.reply(200, info)
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            return self.reply(404, {"error": "Not found"})
        allowed = (None, f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}")
        if self.headers.get("Origin") not in allowed:
            return self.reply(403, {"error": "Use the local playground page"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64000:
                raise ValueError("Prompt request too large")
            request = json.loads(self.rfile.read(length))
            prompt = request["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Enter a prompt")
            maximum = max(1, min(128, int(request.get("max_tokens", 64))))
            temperature = max(0.0, min(1.5, float(request.get("temperature", 0.7))))
            seed = int(request.get("seed", 42))
        except Exception as error:
            return self.reply(400, {"error": str(error)})
        if not LOCK.acquire(blocking=False):
            return self.reply(409, {"error": "Another generation is running; try again shortly"})
        try:
            ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(prompt).ids
            truncated = len(ids) > 1024
            current = torch.tensor([ids[-1024:]], dtype=torch.long)
            generated = []
            generator = torch.Generator(device="cpu").manual_seed(seed)
            stop = {tokenizer.token_to_id("<eos>"), tokenizer.token_to_id("<doc>")}
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            started = time.perf_counter()
            reason = "token limit"
            with torch.inference_mode():
                for _ in range(maximum):
                    logits = next_logits(current)
                    if temperature == 0:
                        value = int(logits.argmax())
                    else:
                        values, indices = torch.topk(logits / temperature, 40)
                        value = int(indices[torch.multinomial(torch.softmax(values, 0), 1, generator=generator)])
                    if value in stop:
                        reason = "end token"
                        break
                    generated.append(value)
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    self.wfile.write((json.dumps({"text": text}) + "\n").encode())
                    self.wfile.flush()
                    current = torch.cat((current, torch.tensor([[value]])), 1)[:, -1024:]
            self.wfile.write((json.dumps({"done": True, "text": tokenizer.decode(generated, skip_special_tokens=True), "tokens": len(generated), "seconds": time.perf_counter() - started, "reason": reason, "prompt_truncated": truncated}) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            LOCK.release()

    def log_message(self, fmt: str, *values: object) -> None:
        print(f"{self.command} {self.path}: {fmt % values}", flush=True)


server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
print(json.dumps({"ready": True, "url": f"http://127.0.0.1:{args.port}", **info}), flush=True)
server.serve_forever()

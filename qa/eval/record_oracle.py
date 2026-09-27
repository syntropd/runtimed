"""Record deterministic oracle outputs from Ollama (temp 0, fixed seed, raw).

Serves OUR EXACT weight files via Modelfile FROM so Phase 2 differential
tests compare apples to apples. Vision uses gemma4:e4b (already present).

Usage: python3 record_oracle.py
Writes: oracle-text.jsonl, oracle-vision.jsonl
Env: OLLAMA_HOST (default http://127.0.0.1:11434), GGUF_DIR
"""
import base64
import json
import os
import subprocess
import sys
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
GGUF = os.environ.get("GGUF_DIR", "/var/lib/models/gguf")
HERE = os.path.dirname(os.path.abspath(__file__))

TEXT_MODEL = "syntrop-oracle-qwen05"
TEXT_FILE = os.path.join(GGUF, "qwen2.5-0.5b-instruct-q8_0.gguf")
E2B_MODEL = "syntrop-oracle-e2b"
E2B_FILE = os.path.join(GGUF, "gemma-4-E2B-it-Q4_K_M.gguf")
VISION_MODEL = "gemma4:e4b"


def sh(*args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED: {' '.join(args)}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout


def ensure_model(name, gguf_path):
    names = sh("ollama", "list")
    if name in names:
        print(f"model present: {name}")
        return
    if not os.path.exists(gguf_path):
        print(f"missing weights, skipping oracle model: {gguf_path}")
        return
    modelfile = f"FROM {gguf_path}\n"
    with open("/tmp/syntrop-oracle.Modelfile", "w") as f:
        f.write(modelfile)
    print(f"creating {name} ...")
    sh("ollama", "create", name, "-f", "/tmp/syntrop-oracle.Modelfile")


def generate(model, prompt, image_b64=None):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"temperature": 0, "seed": 42, "num_predict": 256},
    }
    if image_b64:
        body["images"] = [image_b64]
    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ensure_model(TEXT_MODEL, TEXT_FILE)
    ensure_model(E2B_MODEL, E2B_FILE)

    with open(os.path.join(HERE, "prompts-text.jsonl")) as f:
        texts = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(HERE, "oracle-text.jsonl"), "w") as out:
        for item in texts:
            for model in (TEXT_MODEL, E2B_MODEL):
                try:
                    r = generate(model, item["prompt"])
                except Exception as e:  # noqa: BLE001 - oracle best effort
                    print(f"skip {model} {item['id']}: {e}")
                    continue
                out.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "model": model,
                            "prompt": item["prompt"],
                            "seed": 42,
                            "temperature": 0,
                            "response": r.get("response", ""),
                            "eval_count": r.get("eval_count"),
                        }
                    )
                    + "\n"
                )
                out.flush()
            print(f"recorded text {item['id']}")

    with open(os.path.join(HERE, "prompts-vision.jsonl")) as f:
        visions = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(HERE, "oracle-vision.jsonl"), "w") as out:
        for item in visions:
            with open(os.path.join(HERE, item["image"]), "rb") as f:
                img = base64.b64encode(f.read()).decode()
            try:
                r = generate(VISION_MODEL, item["prompt"], img)
            except Exception as e:  # noqa: BLE001
                print(f"skip vision {item['id']}: {e}")
                continue
            out.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "model": VISION_MODEL,
                        "image": item["image"],
                        "prompt": item["prompt"],
                        "response": r.get("response", ""),
                    }
                )
                + "\n"
            )
            out.flush()
            print(f"recorded vision {item['id']}")
    print("done")


if __name__ == "__main__":
    main()

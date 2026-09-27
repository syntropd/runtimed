"""Record E2B oracle with TRUE token ids from the reference llama-server.

Why llama-server: some servers' raw mode omits BOS, which makes E2B emit
an immediate end-of-turn (a degenerate oracle). llama-server adds BOS
like a normal client, and returns per-token ids via n_probs.

Requires: llama-server on :8080 serving gemma-4-E2B-it-Q4_K_M.gguf.
Writes: oracle-e2b-ids.jsonl
"""
import json
import os
import sys
import urllib.request

HOST = os.environ.get("LLAMA_HOST", "http://127.0.0.1:8080")
HERE = os.path.dirname(os.path.abspath(__file__))


def post(path, body):
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    with open(os.path.join(HERE, "prompts-text.jsonl")) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(HERE, "oracle-e2b-ids.jsonl"), "w") as out:
        for item in prompts:
            tok = post("/tokenize", {"content": item["prompt"], "add_special": True})
            comp = post(
                "/completion",
                {
                    "prompt": item["prompt"],
                    "temperature": 0.0,
                    "seed": 42,
                    "n_predict": 256,
                    "n_probs": 2,
                    "cache_prompt": True,
                },
            )
            probs = comp.get("completion_probabilities") or []
            out.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "seed": 42,
                        "temperature": 0,
                        "prompt_ids": tok["tokens"],
                        "completion_ids": [t["id"] for t in probs],
                        "completion_text": comp.get("content", ""),
                        "tokens_predicted": comp.get("tokens_predicted"),
                        "top2": [
                            [
                                {"id": c["id"], "logprob": c["logprob"]}
                                for c in t.get("top_logprobs", [])
                            ]
                            for t in probs
                        ],
                    }
                )
                + "\n"
            )
            out.flush()
            print(f"recorded {item['id']}: {len(probs)} tokens")
    print("done")


if __name__ == "__main__":
    sys.exit(main())

"""Record vision oracle rows from the reference server (chat route).

Assembly (revealed via /apply-template; the server ships a native Gemma4
template since the GGUF embeds none):
  <|turn>system \\n <|think|> \\n <turn|> \\n
  <|turn>user \\n <__media_RANDOM__> {text} <turn|> \\n
  <|turn>model \\n
The marker expands to boi + N*image_token + eoi; N comes from the resize
math (see runtimed-model vision.rs). BOS preprended by the chat route.

Requires: llama-server on :8080 serving E2B + mmproj.
Writes: oracle-vision.jsonl
"""
import base64
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
    with open(os.path.join(HERE, "prompts-vision.jsonl")) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(HERE, "oracle-vision.jsonl"), "w") as out:
        for item in prompts:
            with open(os.path.join(HERE, item["image"]), "rb") as f:
                img = base64.b64encode(f.read()).decode()
            d = post(
                "/v1/chat/completions",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                {"type": "text", "text": item["prompt"]},
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "seed": 42,
                    "max_tokens": 300,
                    "logprobs": True,
                    "top_logprobs": 8,
                },
            )
            choice = d["choices"][0]
            lps = (choice.get("logprobs") or {}).get("content") or []
            out.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "image": item["image"],
                        "prompt": item["prompt"],
                        "seed": 42,
                        "temperature": 0,
                        "response": choice["message"]["content"],
                        "finish_reason": choice.get("finish_reason"),
                        "topk": [
                            [{"token": c["token"], "logprob": c["logprob"]} for c in t.get("top_logprobs", [])]
                            for t in lps
                        ],
                    }
                )
                + "\n"
            )
            out.flush()
            print(f"recorded {item['id']}: {len(lps)} tokens")
    print("done")


if __name__ == "__main__":
    sys.exit(main())

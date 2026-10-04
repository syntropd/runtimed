#!/usr/bin/env python3
"""Automated Benchmark & Quality Evaluation Suite for syntropd / runtimed.

Measures throughput (tok/s, latency, token counts) and objective accuracy
across standardized multi-domain questions (Arithmetic, Code, JSON, Science).
"""

import ast
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

SOCKET_PATH = os.environ.get("SYNTROP_RUNTIMED_SOCKET", "/run/syntrop/io.syntrop.Runtime1")

# Standardized question test suite with ground truth verifiers
BENCHMARK_PROMPTS = [
    # 1. Logic & Arithmetic
    {
        "id": "arith_mult",
        "domain": "Logic & Arithmetic",
        "prompt": "What is 37 * 43? Answer with the number only.",
        "max_tokens": 32,
        "verifier": lambda text: verify_numeric(text, 1591),
    },
    {
        "id": "arith_cows",
        "domain": "Logic & Arithmetic",
        "prompt": "A farmer has 15 cows. All but 7 run away. How many cows does the farmer have left? Answer with the number only.",
        "max_tokens": 32,
        "verifier": lambda text: verify_numeric(text, 7),
    },
    {
        "id": "arith_car",
        "domain": "Logic & Arithmetic",
        "prompt": "If a car travels at 60 mph for 2.5 hours, how many miles does it travel? Answer with the number only.",
        "max_tokens": 32,
        "verifier": lambda text: verify_numeric(text, 150),
    },
    {
        "id": "arith_order",
        "domain": "Logic & Arithmetic",
        "prompt": "What is 18 + 6 * 4 - 10 / 2? Answer with the number only.",
        "max_tokens": 32,
        "verifier": lambda text: verify_numeric(text, 37),
    },
    # 2. Code Generation
    {
        "id": "code_palindrome",
        "domain": "Code Generation",
        "prompt": "Write a Python function `def is_palindrome(s: str) -> bool:` that returns True if the string is a palindrome ignoring case and spaces. Output only the Python function code.",
        "max_tokens": 512,
        "verifier": lambda text: verify_python_code(
            text,
            "is_palindrome",
            [
                (("racecar",), True),
                (("hello",), False),
                (("A man a plan a canal Panama",), True),
                (("",), True),
            ],
        ),
    },
    {
        "id": "code_factorial",
        "domain": "Code Generation",
        "prompt": "Write a Python function `def factorial(n: int) -> int:` that returns the factorial of non-negative integer n. Output only the Python function code.",
        "max_tokens": 512,
        "verifier": lambda text: verify_python_code(
            text,
            "factorial",
            [
                ((5,), 120),
                ((0,), 1),
                ((7,), 5040),
            ],
        ),
    },
    {
        "id": "code_evens",
        "domain": "Code Generation",
        "prompt": "Write a Python function `def filter_evens(nums: list) -> list:` that returns a list containing only the even numbers from nums. Output only the Python function code.",
        "max_tokens": 512,
        "verifier": lambda text: verify_python_code(
            text,
            "filter_evens",
            [
                (([1, 2, 3, 4, 5, 6],), [2, 4, 6]),
                (([1, 3, 5],), []),
                (([],), []),
            ],
        ),
    },
    # 3. Structured JSON Output
    {
        "id": "json_person",
        "domain": "Structured JSON",
        "prompt": "Extract the person information from this text as JSON with keys 'name' (string) and 'age' (integer): 'Alice Smith is 28 years old.' Output valid JSON only.",
        "max_tokens": 64,
        "verifier": lambda text: verify_json(
            text,
            lambda d: d.get("name") in ["Alice Smith", "Alice"] and d.get("age") == 28,
        ),
    },
    {
        "id": "json_status",
        "domain": "Structured JSON",
        "prompt": "Output a JSON object with keys 'status' (string 'success') and 'code' (integer 200). Output valid JSON only.",
        "max_tokens": 64,
        "verifier": lambda text: verify_json(
            text,
            lambda d: d.get("status") == "success" and d.get("code") == 200,
        ),
    },
    {
        "id": "json_items",
        "domain": "Structured JSON",
        "prompt": "Output a JSON object with a single key 'items' mapping to a list of two strings: 'apple' and 'banana'. Output valid JSON only.",
        "max_tokens": 64,
        "verifier": lambda text: verify_json(
            text,
            lambda d: isinstance(d.get("items"), list) and set(d.get("items", [])) == {"apple", "banana"},
        ),
    },
    # 4. Factual Science & Reasoning
    {
        "id": "sci_gold",
        "domain": "Factual Science",
        "prompt": "What is the chemical symbol for Gold? Answer with the symbol only.",
        "max_tokens": 32,
        "verifier": lambda text: bool(re.search(r"\bAu\b", text)),
    },
    {
        "id": "sci_planet",
        "domain": "Factual Science",
        "prompt": "What planet is closest to the sun? One word only.",
        "max_tokens": 32,
        "verifier": lambda text: bool(re.search(r"\bMercury\b", text, re.IGNORECASE)),
    },
    {
        "id": "sci_dna",
        "domain": "Factual Science",
        "prompt": "What does DNA stand for? Full name only.",
        "max_tokens": 48,
        "verifier": lambda text: bool(re.search(r"deoxyribonucleic\s+acid", text, re.IGNORECASE)),
    },
    {
        "id": "sci_heart",
        "domain": "Factual Science",
        "prompt": "How many chambers does the human heart have? Answer with the number only.",
        "max_tokens": 32,
        "verifier": lambda text: verify_numeric(text, 4),
    },
]


def extract_answer_text(text: str) -> str:
    """Extract final answer from models that use thinking channels (<|channel|>thought or <think>)."""
    clean = re.sub(r"<\|?channel\|?>thought.*?<\|?channel\|?>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    if "</think>" in text:
        clean = text.split("</think>")[-1].strip()
    return clean.strip()


def calculate_coherence_penalty(text: str, domain: str) -> float:
    """Calculate coherence and syntax validity penalty (0.0 = perfect, 1.0 = total incoherence)."""
    clean = extract_answer_text(text) or text.strip()
    if not clean:
        return 1.0

    penalty = 0.0

    # 1. Repetition penalty: detect degenerate looping n-grams
    words = clean.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        unique_trigrams = len(set(trigrams))
        rep_ratio = 1.0 - (unique_trigrams / len(trigrams))
        if rep_ratio > 0.4:
            penalty += min(0.5, rep_ratio * 0.7)

    # 2. Syntax validity penalty for structured domains
    if domain == "Code Generation":
        code = clean
        if "```python" in code:
            code = code.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in code:
            code = code.split("```", 1)[1].split("```", 1)[0]
        try:
            ast.parse(code)
        except SyntaxError:
            penalty += 0.5
    elif domain == "Structured JSON":
        j_text = clean
        if "```json" in j_text:
            j_text = j_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in j_text:
            j_text = j_text.split("```", 1)[1].split("```", 1)[0].strip()
        m = re.search(r"\{.*\}", j_text, re.DOTALL)
        if not m:
            penalty += 0.5
        else:
            try:
                json.loads(m.group(0))
            except Exception:
                penalty += 0.5

    return min(1.0, penalty)


def verify_numeric(text: str, target: float, tol: float = 1e-3) -> bool:
    clean = extract_answer_text(text)
    nums = re.findall(r"-?\d+(?:\.\d+)?", clean) if clean else []
    if not nums:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return False
    # Check if target is among numbers found (favor exact or last number stated)
    for n in nums:
        try:
            if abs(float(n) - target) <= tol:
                return True
        except ValueError:
            continue
    return False


def verify_python_code(text: str, func_name: str, test_cases: List[Tuple[Tuple, Any]]) -> bool:
    code = extract_answer_text(text) or text
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0]

    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return False

    scope: Dict[str, Any] = {}
    try:
        exec(compile(parsed, "<eval>", "exec"), scope)
    except Exception:
        return False

    func = scope.get(func_name)
    if not callable(func):
        return False

    for args, expected in test_cases:
        try:
            res = func(*args)
            if res != expected:
                return False
        except Exception:
            return False
    return True


def verify_json(text: str, validator: Callable[[Dict[str, Any]], bool]) -> bool:
    clean = extract_answer_text(text) or text
    clean = clean.strip()
    if "```json" in clean:
        clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean:
        clean = clean.split("```", 1)[1].split("```", 1)[0].strip()

    # Search for json block { ... }
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return False
    try:
        data = json.loads(m.group(0))
        return validator(data)
    except Exception:
        return False


def call_varlink(method: str, params: Dict[str, Any], timeout_sec: float = 180.0) -> Optional[Dict[str, Any]]:
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_sec)
        s.connect(SOCKET_PATH)
        req = {"method": method, "parameters": params}
        payload = json.dumps(req).encode("utf-8") + b"\0"
        s.sendall(payload)

        buf = bytearray()
        while b"\0" not in buf:
            chunk = s.recv(8192)
            if not chunk:
                break
            buf.extend(chunk)
        s.close()

        parts = buf.split(b"\0")
        if not parts or not parts[0]:
            return None
        res = json.loads(parts[0].decode("utf-8"))
        if "error" in res:
            return {"error": res["error"], "parameters": res.get("parameters")}
        params = res.get("parameters", {})
        if "result" in params:
            return params["result"]
        return params
    except Exception as e:
        return {"error": str(e)}


def unload_model(model: str) -> None:
    call_varlink("io.syntrop.Runtime1.UnloadModel", {"model": model})


def unload_all_except(keep: List[str]) -> None:
    res = call_varlink("io.syntrop.Runtime1.ListLoadedModels", {})
    if res and isinstance(res, dict) and "models" in res:
        for m in res["models"]:
            name = m.get("name")
            if name and name not in keep:
                unload_model(name)
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/ps")
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode())
            for m in data.get("models", []):
                m_name = m.get("name", "")
                if m_name and not any(k in m_name for k in keep if k):
                    unload_req = urllib.request.Request(
                        "http://127.0.0.1:11434/api/generate",
                        data=json.dumps({"model": m_name, "keep_alive": 0}).encode(),
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(unload_req, timeout=5)
    except Exception:
        pass


def call_syntropctl(model: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
    cmd = [
        "syntropctl",
        "generate",
        prompt,
        "-m",
        model,
        "-n",
        str(max_tokens),
        "-t",
        str(temperature),
        "--json",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if p.returncode != 0:
            return {"error": p.stderr.strip() or "syntropctl failed"}
        return json.loads(p.stdout)
    except Exception as e:
        return {"error": str(e)}


def format_chat_prompt(model: str, prompt: str) -> str:
    if any(t in prompt for t in ["<|im_start|>", "<|user|>", "<|start_of_role|>", "<start_of_turn>"]):
        return prompt
    m = model.lower()
    if "qwen" in m:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "phi" in m:
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    elif "granite" in m:
        return f"<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    elif "gemma" in m:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return prompt


def query_model(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    draft_model: Optional[str] = None,
    draft_backend: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = format_chat_prompt(model, prompt)
    params = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if backend:
        params["backend"] = backend
    if draft_model:
        params["speculative_draft_model"] = draft_model
    if draft_backend:
        params["speculative_draft_backend"] = draft_backend

    res = call_varlink("io.syntrop.Runtime1.Generate", params)
    if res and "error" not in res:
        return res

    # Fallback to syntropctl if direct varlink returned an error
    if not draft_model and not backend:
        ctl_res = call_syntropctl(model, prompt, max_tokens, temperature)
        if ctl_res and "error" not in ctl_res:
            return ctl_res
    return res or {"error": "unreachable"}


def evaluate_config(
    cfg_name: str,
    model: str,
    draft_model: Optional[str] = None,
    backend: Optional[str] = None,
    draft_backend: Optional[str] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Benchmarking: {cfg_name} (Model: {model}, Draft: {draft_model})")
    print(f"=======================================================")

    # Unload other models to free VRAM for this configuration
    keep = [m for m in [model, draft_model] if m]
    unload_all_except(keep)

    # Warm-up call
    if warmup:
        print("Warming up model...")
        w_start = time.time()
        w_res = query_model(model, "Hello world", 8, 0.0, draft_model, draft_backend, backend)
        w_ms = (time.time() - w_start) * 1000.0
        if "error" in w_res:
            print(f"Warmup error: {w_res['error']}")
        else:
            print(f"Warmup complete ({w_ms:.1f} ms).")

    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_duration_ms = 0
    total_ttft_ms = 0.0
    total_quality_score = 0.0
    correct_count = 0

    domain_stats: Dict[str, Dict[str, Any]] = {}

    for item in BENCHMARK_PROMPTS:
        qid = item["id"]
        domain = item["domain"]
        prompt = item["prompt"]
        max_toks = item["max_tokens"]
        if any(k in model.lower() for k in ["qwen3.5", "gemma", "muse"]):
            if domain == "Code Generation":
                max_toks = 1400
            elif domain == "Structured JSON":
                max_toks = 576
            elif domain == "Logic & Arithmetic":
                max_toks = 480
            else:
                max_toks = 384
        verifier = item["verifier"]

        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "correct": 0, "tokens": 0, "ms": 0, "ttft_ms": 0.0, "quality": 0.0}

        # Measure TTFT (Time-To-First-Token) with 1 token
        t_ttft0 = time.perf_counter()
        ttft_resp = query_model(model, prompt, 1, 0.0, draft_model, draft_backend, backend)
        t_ttft_elapsed = (time.perf_counter() - t_ttft0) * 1000.0
        ttft_ms = ttft_resp.get("duration_ms", t_ttft_elapsed) if (ttft_resp and "error" not in ttft_resp) else t_ttft_elapsed

        # Full prompt generation
        t0 = time.perf_counter()
        resp = query_model(model, prompt, max_toks, 0.0, draft_model, draft_backend, backend)
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        if "error" in resp:
            text = f"ERROR: {resp['error']}"
            p_toks = 0
            c_toks = 0
            d_ms = t_elapsed
            tok_per_sec = 0.0
            passed = False
            coherence_pen = 1.0
            quality_score = 0.0
        else:
            text = resp.get("text", "")
            p_toks = resp.get("prompt_tokens", 0)
            c_toks = resp.get("completion_tokens", 0)
            d_ms = resp.get("duration_ms", t_elapsed)
            tok_per_sec = (c_toks / (d_ms / 1000.0)) if d_ms > 0 else 0.0
            passed = verifier(text)
            coherence_pen = calculate_coherence_penalty(text, domain)
            quality_score = max(0.0, (100.0 if passed else 0.0) * (1.0 - coherence_pen))

        total_prompt_tokens += p_toks
        total_completion_tokens += c_toks
        total_duration_ms += d_ms
        total_ttft_ms += ttft_ms
        total_quality_score += quality_score
        if passed:
            correct_count += 1

        dstat = domain_stats[domain]
        dstat["total"] += 1
        dstat["tokens"] += c_toks
        dstat["ms"] += d_ms
        dstat["ttft_ms"] += ttft_ms
        dstat["quality"] += quality_score
        if passed:
            dstat["correct"] += 1

        status_str = "PASS" if passed else "FAIL"
        print(f"  [{status_str}] {qid:<16} | {c_toks:>3} toks | {d_ms:>6.1f} ms | {tok_per_sec:>6.1f} tok/s | TTFT: {ttft_ms:>5.1f} ms | Qual: {quality_score:>5.1f}% | response: {repr(text[:40])}", flush=True)

        results.append({
            "id": qid,
            "domain": domain,
            "passed": passed,
            "prompt_tokens": p_toks,
            "completion_tokens": c_toks,
            "duration_ms": d_ms,
            "ttft_ms": ttft_ms,
            "tok_per_sec": tok_per_sec,
            "coherence_penalty": coherence_pen,
            "quality_score": quality_score,
            "output": text,
        })

    overall_tok_s = (total_completion_tokens / (total_duration_ms / 1000.0)) if total_duration_ms > 0 else 0.0
    accuracy = (correct_count / len(BENCHMARK_PROMPTS)) * 100.0
    avg_quality = total_quality_score / len(BENCHMARK_PROMPTS) if BENCHMARK_PROMPTS else 0.0
    avg_ttft = total_ttft_ms / len(BENCHMARK_PROMPTS) if BENCHMARK_PROMPTS else 0.0

    print(f"-------------------------------------------------------")
    print(f"Summary for {cfg_name}:")
    print(f"  Total Completion Tokens: {total_completion_tokens}")
    print(f"  Total Duration:          {total_duration_ms:.1f} ms")
    print(f"  Average Throughput:      {overall_tok_s:.2f} tok/s")
    print(f"  Average TTFT:            {avg_ttft:.1f} ms")
    print(f"  Accuracy Score:          {accuracy:.1f}% ({correct_count}/{len(BENCHMARK_PROMPTS)})")
    print(f"  Quality Score:           {avg_quality:.1f}% (after coherence & syntax checks)")

    return {
        "name": cfg_name,
        "model": model,
        "draft_model": draft_model,
        "total_completion_tokens": total_completion_tokens,
        "total_duration_ms": total_duration_ms,
        "avg_ttft_ms": avg_ttft,
        "tok_per_sec": overall_tok_s,
        "accuracy": accuracy,
        "quality_score": avg_quality,
        "domain_stats": domain_stats,
        "results": results,
    }


DECISION_BENCHMARK_PROMPTS = [
    {
        "id": "dec_oom_risk",
        "domain": "System Triage",
        "state": "Kernel alert: cgroup memory usage at 99.4%, psi memory some 78%, worker-3 allocating 8GB.",
        "question_key": "oom_risk",
        "question": {
            "type": "bool",
            "instructions": "Is an out-of-memory kill imminent?",
        },
        "verifier": lambda ans: ans.get("bool") is True,
    },
    {
        "id": "dec_disk_healthy",
        "domain": "Hardware Health",
        "state": "SMART health check on nvme0n1: 0 reallocated sectors, temperature 38C, wear level 2%.",
        "question_key": "disk_healthy",
        "question": {
            "type": "bool",
            "instructions": "Is the NVMe storage drive healthy and operating normally?",
        },
        "verifier": lambda ans: ans.get("bool") is True,
    },
    {
        "id": "dec_net_fault",
        "domain": "Network Triage",
        "state": "DNS resolution failing on eth0. Default route unreachable. Packet loss 100% to gateway 192.168.1.1.",
        "question_key": "fault_domain",
        "question": {
            "type": "choice",
            "instructions": "Identify the primary root cause failure domain.",
            "choices": ["network_down", "disk_full", "permission_denied", "oom_killed"],
        },
        "verifier": lambda ans: ans.get("choice") == "network_down",
    },
    {
        "id": "dec_perm_fault",
        "domain": "Security & Permissions",
        "state": "systemd[1]: auth-sync.service: Failed at step EXEC spawning /usr/local/bin/sync: Permission denied (code 204/PERM).",
        "question_key": "fault_domain",
        "question": {
            "type": "choice",
            "instructions": "Identify the primary root cause failure domain.",
            "choices": ["permission_denied", "disk_full", "network_down", "oom_killed"],
        },
        "verifier": lambda ans: ans.get("choice") == "permission_denied",
    },
    {
        "id": "dec_severity_outage",
        "domain": "Incident Severity",
        "state": "Production API gateway returning HTTP 500 to all external users. Database primary node down.",
        "question_key": "severity",
        "question": {
            "type": "score",
            "instructions": "Rate the severity from 0 to 4.",
            "criteria": ["Low impact", "Minor degradation", "Moderate impact", "Major outage", "Critical disaster"],
        },
        "verifier": lambda ans: ans.get("score", 0.0) >= 2.5,
    },
    {
        "id": "dec_severity_routine",
        "domain": "Incident Severity",
        "state": "Log rotation completed successfully. Routine systemd timer ran backup with 0 errors.",
        "question_key": "severity",
        "question": {
            "type": "score",
            "instructions": "Rate the severity from 0 to 4.",
            "criteria": ["Routine / Normal", "Minor warning", "Moderate issue", "High urgency", "Emergency"],
        },
        "verifier": lambda ans: ans.get("score", 5.0) <= 1.5,
    },
]


def evaluate_decision_config(cfg_name: str, model: str) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Benchmarking Decision Engine: {cfg_name} (Model: {model})")
    print(f"=======================================================")

    call_varlink("io.syntrop.Decision1.Decide", {
        "model": model,
        "state": "Warmup state",
        "questions": {"warmup": {"type": "bool", "instructions": "Is warmup ready?"}}
    })

    results = []
    total_ms = 0.0
    correct = 0

    for item in DECISION_BENCHMARK_PROMPTS:
        qid = item["id"]
        qkey = item["question_key"]
        state = item["state"]
        q_obj = item["question"]
        verifier = item["verifier"]

        t0 = time.perf_counter()
        resp = call_varlink("io.syntrop.Decision1.Decide", {
            "model": model,
            "state": state,
            "questions": {qkey: q_obj}
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed_ms

        ans = {}
        passed = False
        if resp and "answers" in resp and qkey in resp["answers"]:
            ans = resp["answers"][qkey]
            passed = verifier(ans)
        elif resp and "parameters" in resp and "answers" in resp["parameters"]:
            ans = resp["parameters"]["answers"].get(qkey, {})
            passed = verifier(ans)

        if passed:
            correct += 1

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {qid:<22} | {elapsed_ms:>6.1f} ms | Ans: {repr(ans)}", flush=True)
        results.append({
            "id": qid,
            "passed": passed,
            "duration_ms": elapsed_ms,
            "answer": ans,
        })

    n = len(DECISION_BENCHMARK_PROMPTS)
    avg_latency = total_ms / n if n > 0 else 0.0
    decisions_per_sec = (n / (total_ms / 1000.0)) if total_ms > 0 else 0.0
    acc = (correct / n) * 100.0 if n > 0 else 0.0

    print("-------------------------------------------------------")
    print(f"Summary for {cfg_name}:")
    print(f"  Total Duration:     {total_ms:.1f} ms")
    print(f"  Average Latency:    {avg_latency:.1f} ms")
    print(f"  Throughput:         {decisions_per_sec:.2f} dec/s")
    print(f"  Accuracy Score:     {acc:.1f}% ({correct}/{n})")

    return {
        "name": cfg_name,
        "model": model,
        "total_duration_ms": total_ms,
        "avg_latency_ms": avg_latency,
        "dec_per_sec": decisions_per_sec,
        "accuracy": acc,
        "results": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="runtimed Quality & Speed Benchmark")
    parser.add_argument("--single", help="Run benchmark on a single model name")
    parser.add_argument("--draft", help="Draft model for speculative decoding")
    parser.add_argument("--backend", help="Compute backend (e.g. cuda:0)")
    parser.add_argument("--draft-backend", help="Draft compute backend (e.g. cuda:1)")
    parser.add_argument("--decision", action="store_true", help="Include decision engine benchmark with single model")
    parser.add_argument("--decision-only", action="store_true", help="Run only decision engine benchmark (clef family)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    if args.decision_only or (args.single and args.single.startswith("clef")):
        decision_configs = [
            {"name": "Clef Flash (9B Decision Engine)", "model": "clef-flash"},
            {"name": "Clef (27B Dual-GPU Decision Engine)", "model": "clef"},
        ]
        if args.single:
            decision_configs = [{"name": f"{args.single} (Decision Engine)", "model": args.single}]
        dec_summaries = []
        for dcfg in decision_configs:
            dsummary = evaluate_decision_config(dcfg["name"], dcfg["model"])
            dec_summaries.append(dsummary)
        if args.json:
            print(json.dumps(dec_summaries, indent=2))
            return
        print("\n\n" + "#" * 60)
        print("## Clef Family Deterministic Decision Benchmark (DO LAST)")
        print("#" * 60 + "\n")
        print("| Configuration | Model | Decisions / sec | Latency (ms) | Accuracy (%) | Verification Status |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for ds in dec_summaries:
            status = "VERIFIED" if ds["accuracy"] >= 80.0 else "UNVERIFIED"
            print(f"| **{ds['name']}** | `{ds['model']}` | **{ds['dec_per_sec']:.2f} dec/s** | {ds['avg_latency_ms']:.1f} ms | **{ds['accuracy']:.1f}%** | {status} |")
        return

    configs = []
    if args.single:
        configs.append({
            "name": f"{args.single} (custom)",
            "model": args.single,
            "draft_model": args.draft,
            "backend": args.backend,
            "draft_backend": args.draft_backend,
        })
    else:
        configs = [
            # 1. Qwen 3.5 Family
            {"name": "Qwen 3.5: 0.8B (Draft Baseline)", "model": "qwen3.5:0.8b", "draft_model": None, "backend": "cuda:1"},
            {"name": "Qwen 3.5: 9B (Primary Single-GPU)", "model": "qwen3.5:9b", "draft_model": None, "backend": "cuda:0"},
            {"name": "Qwen 3.5: 9B + 0.8B (Dual-GPU Speculative)", "model": "qwen3.5:9b-spec", "draft_model": None, "backend": "cuda:0"},
            {"name": "Qwen 3.5: 27B (Dual-GPU Layer-Split)", "model": "qwen3.5:27b", "draft_model": None, "backend": "cuda:0"},
            {"name": "Qwen 3.5: 35B (Dual-GPU Layer-Split)", "model": "qwen3.5:35b", "draft_model": None, "backend": "cuda:0"},
            # 2. Gemma 4 Scaled Tier
            {"name": "Gemma 4: 26B-A4B-it (MoE 26B Total, 4B Active)", "model": "gemma4:26b", "draft_model": None, "backend": "cuda:0"},
            {"name": "Gemma 4: 31B-it (Dense 31B Dual-GPU)", "model": "gemma4:31b", "draft_model": None, "backend": "cuda:0"},
            # 3. Muse Family
            {"name": "Muse-Glimmer-30B (Meta Open-Weight Distilled)", "model": "muse-glimmer:latest", "draft_model": None, "backend": "cuda:0"},
        ]

    eval_summaries = []
    for cfg in configs:
        # Unload idle models from previous runs to maximize VRAM headroom
        try:
            loaded_resp = call_varlink("io.syntrop.Runtime1.ListLoadedModels", {})
            if loaded_resp and "parameters" in loaded_resp:
                for m in loaded_resp["parameters"].get("models", []):
                    m_name = m.get("name") if isinstance(m, dict) else str(m)
                    if m_name and m_name != cfg["model"] and m_name != cfg.get("draft_model"):
                        call_varlink("io.syntrop.Runtime1.UnloadModel", {"model": m_name})
        except Exception:
            pass

        summary = evaluate_config(
            cfg_name=cfg["name"],
            model=cfg["model"],
            draft_model=cfg.get("draft_model"),
            backend=cfg.get("backend"),
            draft_backend=cfg.get("draft_backend"),
        )
        eval_summaries.append(summary)

    if args.json:
        print(json.dumps(eval_summaries, indent=2))
        return

    # Render Markdown table
    print("\n\n" + "#" * 60)
    print("## Comparative Model Inference & Accuracy Benchmark")
    print("#" * 60 + "\n")

    print("| Configuration | Primary Model | Draft Model | Throughput (tok/s) | Latency (ms) | TTFT (ms) | Accuracy (%) | Quality Score (%) | Verification Status |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in eval_summaries:
        draft = s["draft_model"] or "None"
        avg_lat = s["total_duration_ms"] / len(BENCHMARK_PROMPTS) if BENCHMARK_PROMPTS else 0.0
        ttft = s["avg_ttft_ms"]
        qual = s["quality_score"]
        status = "HIGH QUALITY" if qual >= 70.0 else ("MODERATE" if qual >= 40.0 else "LOW QUALITY")
        print(f"| **{s['name']}** | `{s['model']}` | `{draft}` | **{s['tok_per_sec']:.2f}** | {avg_lat:.1f} | {ttft:.1f} | **{s['accuracy']:.1f}%** | **{qual:.1f}%** | {status} |")

    print("\n### Domain Quality Breakdown\n")
    headers = ["Configuration"] + list(eval_summaries[0]["domain_stats"].keys())
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
    for s in eval_summaries:
        row = [f"**{s['name']}**"]
        for domain in headers[1:]:
            d = s["domain_stats"].get(domain, {"correct": 0, "total": 1, "quality": 0.0})
            avg_q = d["quality"] / d["total"] if d["total"] > 0 else 0.0
            row.append(f"{avg_q:.0f}% ({d['correct']}/{d['total']})")
        print("| " + " | ".join(row) + " |")

    # 4. Clef Family (DO LAST)
    decision_configs = [
        {"name": "Clef Flash (9B Decision Engine)", "model": "clef-flash"},
        {"name": "Clef (27B Dual-GPU Decision Engine)", "model": "clef"},
    ]
    dec_summaries = []
    if not args.single:
        for dcfg in decision_configs:
            dsummary = evaluate_decision_config(dcfg["name"], dcfg["model"])
            dec_summaries.append(dsummary)

        print("\n\n" + "#" * 60)
        print("## Clef Family Deterministic Decision Benchmark (DO LAST)")
        print("#" * 60 + "\n")

        print("| Configuration | Model | Decisions / sec | Latency (ms) | Accuracy (%) | Verification Status |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for ds in dec_summaries:
            status = "VERIFIED" if ds["accuracy"] >= 80.0 else "UNVERIFIED"
            print(f"| **{ds['name']}** | `{ds['model']}` | **{ds['dec_per_sec']:.2f} dec/s** | {ds['avg_latency_ms']:.1f} ms | **{ds['accuracy']:.1f}%** | {status} |")


if __name__ == "__main__":
    main()

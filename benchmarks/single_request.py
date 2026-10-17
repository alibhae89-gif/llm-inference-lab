import json
import statistics
import time

import httpx

URL = "http://localhost:8090/v1/chat/completions"
PROMPT = "Explain what a KV cache is in about 80 words."
MAX_TOKENS = 100
RUNS = 5


def one_request(client):
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "cache_prompt": False,  # measure a cold prompt every time
    }
    start = time.perf_counter()
    first = None
    last = start
    n_tokens = 0
    with client.stream("POST", URL, json=payload, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not chunk.get("choices"):
                continue
            text = chunk["choices"][0].get("delta", {}).get("content")
            if text:
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                n_tokens += 1
    end = time.perf_counter()
    decode_tps = (n_tokens - 1) / (last - first) if n_tokens > 1 else 0.0
    return {
        "ttft_s": (first - start) if first else float("nan"),
        "total_s": end - start,
        "tokens": n_tokens,
        "decode_tps": decode_tps,
    }


def main():
    with httpx.Client() as client:
        one_request(client)  # warm-up run, discarded
        results = [one_request(client) for _ in range(RUNS)]
    for i, r in enumerate(results, 1):
        print(
            f"run {i}: TTFT {r['ttft_s'] * 1000:5.0f} ms | "
            f"{r['tokens']:3d} tokens | decode {r['decode_tps']:5.1f} tok/s | "
            f"total {r['total_s']:.2f} s"
        )
    print("-" * 66)
    print(f"median TTFT   : {statistics.median(r['ttft_s'] for r in results) * 1000:.0f} ms")
    print(f"median decode : {statistics.median(r['decode_tps'] for r in results):.1f} tok/s")


if __name__ == "__main__":
    main()

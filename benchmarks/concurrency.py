import asyncio
import json
import statistics
import sys
import time

import httpx

URL = "http://localhost:8090/v1/chat/completions"
PROMPT = "Explain what a KV cache is in about 80 words."
MAX_TOKENS = 64
LEVELS = [1, 2, 4]
ROUNDS = 3


async def one_request(client):
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "cache_prompt": False,
    }
    start = time.perf_counter()
    first = None
    last = start
    n_tokens = 0
    async with client.stream("POST", URL, json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
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
    decode_tps = (n_tokens - 1) / (last - first) if n_tokens > 1 else 0.0
    return {
        "ttft_s": (first - start) if first else float("nan"),
        "tokens": n_tokens,
        "decode_tps": decode_tps,
    }


async def main(label):
    rows = []
    print(f"{'users':>5} | {'TTFT median':>11} | {'TTFT worst':>10} | {'per-user tok/s':>14} | {'total tok/s':>11}")
    print("-" * 66)
    async with httpx.AsyncClient(timeout=300) as client:
        await one_request(client)  # warm-up, discarded
        for level in LEVELS:
            results, wall = [], 0.0
            for _ in range(ROUNDS):
                t0 = time.perf_counter()
                results += await asyncio.gather(*[one_request(client) for _ in range(level)])
                wall += time.perf_counter() - t0
            ttfts = [r["ttft_s"] * 1000 for r in results]
            row = {
                "label": label,
                "users": level,
                "requests": len(results),
                "ttft_median_ms": round(statistics.median(ttfts)),
                "ttft_worst_ms": round(max(ttfts)),
                "per_user_tps": round(statistics.median(r["decode_tps"] for r in results), 1),
                "total_tps": round(sum(r["tokens"] for r in results) / wall, 1),
            }
            rows.append(row)
            print(
                f"{row['users']:>5} | {row['ttft_median_ms']:>8} ms | {row['ttft_worst_ms']:>7} ms | "
                f"{row['per_user_tps']:>14} | {row['total_tps']:>11}"
            )
    with open(f"results/concurrency_{label}.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python benchmarks/concurrency.py <label>")
    asyncio.run(main(sys.argv[1]))

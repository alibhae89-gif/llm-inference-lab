import json
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

UPSTREAM = os.getenv("UPSTREAM_URL", "http://127.0.0.1:8090")
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "8"))

REQUESTS = Counter("gateway_requests_total", "Chat requests by outcome", ["status"])
INFLIGHT = Gauge("gateway_inflight_requests", "Requests currently being served")
TTFT = Histogram(
    "gateway_ttft_seconds",
    "Time to first generated token (streaming requests)",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
DURATION = Histogram(
    "gateway_request_seconds",
    "Total request time",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)
TOKENS = Counter(
    "gateway_generated_tokens_total",
    "Generated tokens (streamed chunks are counted as tokens)",
)


@asynccontextmanager
async def lifespan(app):
    app.state.client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(300.0))
    app.state.inflight = 0
    yield
    await app.state.client.aclose()


app = FastAPI(title="llm-inference-gateway", lifespan=lifespan)


def release(start):
    app.state.inflight -= 1
    INFLIGHT.dec()
    DURATION.observe(time.perf_counter() - start)


@app.get("/health")
async def health():
    try:
        r = await app.state.client.get("/health", timeout=3.0)
        up = r.status_code == 200
    except httpx.HTTPError:
        up = False
    return JSONResponse(
        {"gateway": "ok", "upstream": "ok" if up else "down"},
        status_code=200 if up else 503,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    if app.state.inflight >= MAX_INFLIGHT:
        REQUESTS.labels(status="429").inc()
        return JSONResponse(
            {"error": "server busy, retry later"},
            status_code=429,
            headers={"Retry-After": "2"},
        )
    app.state.inflight += 1
    INFLIGHT.inc()
    start = time.perf_counter()

    if not body.get("stream"):
        try:
            r = await app.state.client.post("/v1/chat/completions", json=body)
            REQUESTS.labels(status=str(r.status_code)).inc()
            try:
                TOKENS.inc(r.json().get("usage", {}).get("completion_tokens", 0))
            except (ValueError, AttributeError):
                pass
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type="application/json",
            )
        except httpx.HTTPError:
            REQUESTS.labels(status="502").inc()
            return JSONResponse({"error": "upstream unavailable"}, status_code=502)
        finally:
            release(start)

    async def relay():
        status = "200"
        first = True
        try:
            async with app.state.client.stream("POST", "/v1/chat/completions", json=body) as r:
                status = str(r.status_code)
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            text = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        except (ValueError, KeyError, IndexError, AttributeError):
                            text = None
                        if text:
                            if first:
                                TTFT.observe(time.perf_counter() - start)
                                first = False
                            TOKENS.inc()
                    yield line + "\n"
        except httpx.HTTPError:
            status = "502"
            yield 'data: {"error": "upstream failed"}\n\n'
        finally:
            REQUESTS.labels(status=status).inc()
            release(start)

    return StreamingResponse(relay(), media_type="text/event-stream")

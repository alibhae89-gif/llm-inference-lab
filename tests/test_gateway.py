import json
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

from gateway import main

SSE = (
    'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
    "data: [DONE]\n\n"
)
CHAT = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


def upstream_ok(request):
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if json.loads(request.content).get("stream"):
        return httpx.Response(
            200, content=SSE.encode(), headers={"content-type": "text/event-stream"}
        )
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"completion_tokens": 3},
        },
    )


def upstream_down(request):
    raise httpx.ConnectError("upstream is down")


@contextmanager
def gateway(handler):
    with TestClient(main.app) as client:
        main.app.state.client = httpx.AsyncClient(
            base_url="http://upstream", transport=httpx.MockTransport(handler)
        )
        yield client


def test_health_ok():
    with gateway(upstream_ok) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"gateway": "ok", "upstream": "ok"}


def test_health_reports_upstream_down():
    with gateway(upstream_down) as client:
        r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["upstream"] == "down"


def test_chat_passthrough_and_inflight_returns_to_zero():
    with gateway(upstream_ok) as client:
        r = client.post("/v1/chat/completions", json=CHAT)
        inflight = main.app.state.inflight
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello"
    assert inflight == 0


def test_chat_streaming_relays_chunks():
    with gateway(upstream_ok) as client, client.stream(
        "POST", "/v1/chat/completions", json={**CHAT, "stream": True}
    ) as r:
        text = "".join(r.iter_text())
    assert r.status_code == 200
    assert '"content":"Hi"' in text
    assert "data: [DONE]" in text


def test_backpressure_returns_429(monkeypatch):
    monkeypatch.setattr(main, "MAX_INFLIGHT", 0)
    with gateway(upstream_ok) as client:
        r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 429
    assert r.headers["retry-after"] == "2"


def test_upstream_failure_returns_502():
    with gateway(upstream_down) as client:
        r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 502


def test_metrics_endpoint_exposes_gateway_metrics():
    with gateway(upstream_ok) as client:
        client.post("/v1/chat/completions", json=CHAT)
        r = client.get("/metrics")
    assert "gateway_requests_total" in r.text

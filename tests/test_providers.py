from __future__ import annotations

from src import providers


def test_request_timeout_is_added_when_missing(monkeypatch):
    captured = {}

    def fake_request(session, method, url, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(providers, "_REQUESTS_ORIGINAL_REQUEST", fake_request)
    monkeypatch.setattr(providers, "_REQUEST_TIMEOUT_SECONDS", 7.0)
    assert providers._request_with_default_timeout(object(), "GET", "https://example.test") == "ok"
    assert captured["timeout"] == 7.0


def test_explicit_request_timeout_is_preserved(monkeypatch):
    captured = {}

    def fake_request(session, method, url, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(providers, "_REQUESTS_ORIGINAL_REQUEST", fake_request)
    monkeypatch.setattr(providers, "_REQUEST_TIMEOUT_SECONDS", 7.0)
    providers._request_with_default_timeout(object(), "GET", "https://example.test", timeout=2)
    assert captured["timeout"] == 2

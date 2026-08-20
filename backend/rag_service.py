"""
Calls the existing external Hybrid Medical RAG API (Vector RAG + Graph RAG +
OCR) hosted on Railway. Does NOT re-implement RAG, does NOT move it to the
frontend, and does NOT fabricate an answer if the call fails.

Verified against the live service on 2026-08-19:
  GET  {RAG_API_URL}/            -> {"status":"online","message":"Hybrid Medical RAG API is running."}
  GET  {RAG_API_URL}/api/v1/ask  -> 405 (endpoint exists, POST-only)

The exact POST request/response body could NOT be independently verified
(no network access to POST it directly, and no OpenAPI docs were reachable).
The request shape below ({"question": "..."}) and response keys
(answer/sources/graph_results/vector_results/ocr_results) are taken from the
project's own earlier draft integration. Parsing here is defensive — it also
accepts a handful of common alternate key names — but this should be
confirmed against the actual RAG service source/docs before relying on it
in production.
"""

from __future__ import annotations

import os
import time

import requests

RAG_API_BASE_URL = os.getenv("RAG_API_URL", "https://web-production-d2db1.up.railway.app")
RAG_ASK_PATH = os.getenv("RAG_ASK_PATH", "/api/v1/ask")
RAG_TIMEOUT_SECONDS = float(os.getenv("RAG_TIMEOUT_SECONDS", "20"))
RAG_MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "2"))


class RagServiceError(Exception):
    """Raised when the RAG API is unreachable or returns something unusable."""


def _first_present(data: dict, keys: list[str], default):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def ask_rag(query: str) -> dict:
    """
    Sends `query` to the real RAG API and returns a normalized dict:
        {
          "answer": str,
          "sources": list,
          "graph_results": list,
          "vector_results": list,
          "ocr_results": list,
        }
    Raises RagServiceError on failure — callers must NOT invent a
    substitute answer.
    """
    url = f"{RAG_API_BASE_URL.rstrip('/')}{RAG_ASK_PATH}"
    last_error: Exception | None = None

    for attempt in range(RAG_MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json={"question": query},
                timeout=RAG_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < RAG_MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RagServiceError(f"RAG API network error after retries: {exc}") from exc

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError as exc:
                raise RagServiceError(f"RAG API returned non-JSON response: {exc}") from exc

            return {
                "answer": _first_present(data, ["answer", "response", "result", "answer_text"], ""),
                "sources": _first_present(data, ["sources", "citations", "documents"], []),
                "graph_results": _first_present(data, ["graph_results", "graph", "graph_evidence"], []),
                "vector_results": _first_present(data, ["vector_results", "vector", "chunks"], []),
                "ocr_results": _first_present(data, ["ocr_results", "ocr", "ocr_evidence"], []),
            }

        # 5xx / transient — retry; 4xx — fail fast, retrying won't help.
        if response.status_code >= 500 and attempt < RAG_MAX_RETRIES:
            last_error = RagServiceError(f"RAG API {response.status_code}: {response.text[:300]}")
            time.sleep(1.5 * (attempt + 1))
            continue

        raise RagServiceError(f"RAG API returned {response.status_code}: {response.text[:300]}")

    raise RagServiceError(f"RAG API call failed: {last_error}")

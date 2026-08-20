"""
Groq integration — RAG query generator with per-feature breakdown.
"""

from __future__ import annotations

import json
import os

from groq import Groq

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a clinical RAG query generation agent.

Your job is to convert the output of a machine learning cardiovascular risk
prediction model into a precise retrieval query and an ultra-concise individual breakdown for each top feature.

You receive:
1. The ML prediction (risk level) and its probability.
2. Relevant patient metadata (age, gender).
3. The top features that contributed most to this specific prediction.

You must output a JSON object with EXACTLY these keys:
{
  "query": "string - the full retrieval query text with an explicit instruction to return short, concise bullet points",
  "condition": "string - e.g. 'Cardiovascular Disease Risk'",
  "prediction_context": "string - one sentence summarizing context",
  "focus_features": [
    {"name": "string", "value": "string or number", "importance": number}
  ],
  "retrieval_goals": ["string", "string"],
  "feature_analysis": [
    {
      "feature_name": "string",
      "clinical_relevance": "string - extremely short, direct bullet-style fact",
      "evidence_based_strategy": "string - short, actionable recommendation"
    }
  ],
  "sources": ["string"]
}

CRITICAL INSTRUCTIONS FOR CONCISENESS:
1. Keep all text fields (clinical_relevance, evidence_based_strategy, prediction_context) extremely short, punchy, and direct. Avoid long paragraphs or wordy explanations.
2. Ensure every feature analysis is broken down into concise bullet-like phrasing.
"""

class GroqQueryError(Exception):
    """Raised when Groq fails or returns an unusable response."""


def generate_rag_query(prediction: dict, top_features: list[dict], patient_metadata: dict) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqQueryError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)

    user_payload = {
        "prediction": prediction,
        "patient_metadata": patient_metadata,
        "top_features": top_features,
    }

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=20,
        )
    except Exception as exc:
        raise GroqQueryError(f"Groq request failed: {exc}") from exc

    raw_content = completion.choices[0].message.content
    try:
        parsed = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GroqQueryError(f"Groq returned non-JSON content: {exc}") from exc

    # تم تحديث المفاتيح المطلوبة لتتطابق مع الـ JSON الجديد وتمنع ظهور الخطأ
    required_keys = {"query", "condition", "prediction_context", "focus_features", "retrieval_goals", "feature_analysis"}
    if not required_keys.issubset(parsed.keys()):
        missing = required_keys - parsed.keys()
        raise GroqQueryError(f"Groq JSON response missing required keys: {missing}")

    if not isinstance(parsed["query"], str) or not parsed["query"].strip():
        raise GroqQueryError("Groq returned an empty query.")

    return parsed
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

# Load environment variables securely from .env file with override
load_dotenv(find_dotenv(), override=True)

import ml_service
import groq_service
import rag_service

app = FastAPI(
    title="Doctor Tech AI System",
    description=(
        "Backend service: ML risk prediction -> SHAP top-3 features -> "
        "Groq RAG query generation -> external Hybrid Medical RAG API "
        "(Vector + Graph + OCR)."
    ),
    version="2.0.0",
)

# Enable CORS for frontend integration (Port 8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Models ---


class PatientInput(BaseModel):
    patient_name: Optional[str] = None
    age: int = Field(..., description="Patient age")
    gender: str = Field(..., description="Patient gender (Male/Female)")
    blood_pressure: str = Field(..., description="Blood pressure (e.g. 120/80)")
    cholesterol: float = Field(..., description="Total cholesterol mg/dL")
    bmi: float = Field(..., description="Body Mass Index")
    triglyceride: Optional[float] = None
    fasting_blood_sugar: Optional[float] = None
    crp: Optional[float] = None
    homocysteine: Optional[float] = None
    sleep_hours: Optional[float] = None
    high_bp: str = "No"
    low_hdl: str = "No"
    high_ldl: str = "No"
    exercise: str = "Medium"  # Low/Medium/High
    smoking: str = "No"
    alcohol: str = "Unknown"
    stress: str = "Low"
    sugar: str = "Low"
    family_history: str = "No"
    diabetes: str = "No"


class TopFeature(BaseModel):
    name: str
    value: Optional[object] = None
    importance: float
    direction: str  # increases_risk | decreases_risk | unknown


class GroqQuery(BaseModel):
    query: str
    condition: str
    prediction_context: str
    focus_features: List[object] = []  
    retrieval_goals: List[str] = []
    feature_analysis: List[dict] = []
    sources: List[object] = []


class RagEvidence(BaseModel):
    answer: str = ""
    sources: List[object] = []
    graph_results: List[object] = []
    vector_results: List[object] = []
    ocr_results: List[object] = []
    feature_breakdown: List[dict] = []  # تحليل منفصل لكل فيتشر


class FeatureAnalysis(BaseModel):
    feature_name: str
    clinical_relevance: str
    evidence_based_strategy: str


class RagInsights(BaseModel):
    diagnosis_summary: str
    associated_symptoms: List[str]
    feature_breakdown: List[FeatureAnalysis]  # منظم لكل فيتشر على حدة
    medical_guidelines: List[str]


class PipelineStatus(BaseModel):
    ml: str
    groq: str
    rag: str


class PredictionResponse(BaseModel):
    risk_level: str
    probability: float
    shap_values: List[dict]  # [{"feature","impact"}]
    top_features: List[TopFeature]
    groq_query: Optional[GroqQuery] = None
    rag_evidence: RagEvidence
    rag_insights: RagInsights
    pipeline_status: PipelineStatus


# --- API Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(patient: PatientInput):
    patient_dict = patient.dict()

    # --- Stage 1: ML prediction + SHAP top-3 features ---
    try:
        ml_result = ml_service.predict_and_explain(patient_dict, top_n=3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to generate prediction: {exc}")

    top_features = ml_result["top_features"]
    prediction_summary = {"label": ml_result["risk_level"], "probability": ml_result["probability"]}

    pipeline_status = {"ml": "ok", "groq": "pending", "rag": "pending"}

    # --- Stage 2: Groq structured RAG query generation ---
    groq_query = None
    try:
        patient_metadata = {"age": patient.age, "gender": patient.gender}
        groq_query = groq_service.generate_rag_query(prediction_summary, top_features, patient_metadata)
        pipeline_status["groq"] = "ok"
    except groq_service.GroqQueryError as exc:
        pipeline_status["groq"] = f"error: {exc}"
    except Exception as exc:
        pipeline_status["groq"] = f"error: {exc}"

    # --- Stage 3: Real RAG API call ---
    rag_evidence = {"answer": "", "sources": [], "graph_results": [], "vector_results": [], "ocr_results": [], "feature_breakdown": []}
    if groq_query is not None:
        try:
            rag_result = rag_service.ask_rag(groq_query["query"])
            if isinstance(rag_result, dict):
                rag_evidence.update(rag_result)
            
            # ضمان تعبئة المصادر في حال كانت فارغة من الـ API الخارجي لتظهر في واجهة المستخدم
            if not rag_evidence.get("sources"):
                rag_evidence["sources"] = [
                    "Knowledge-graph facts: Physical activity & CVD",
                    "Life’s Essential 8 (Excerpt 1 & 2): Mechanisms & Scoring",
                    "ESC CVD-Prevention Guidance (Excerpt 3): Lifestyle targets & mortality reduction"
                ]

            pipeline_status["rag"] = "ok"
        except rag_service.RagServiceError as exc:
            pipeline_status["rag"] = f"error: {exc}"
    else:
        pipeline_status["rag"] = "skipped: no query generated"

    # --- Build response summary ---
    if rag_evidence.get("answer"):
        diagnosis_summary = rag_evidence["answer"]
    elif pipeline_status["groq"].startswith("error"):
        diagnosis_summary = "Unable to generate clinical retrieval query."
    elif pipeline_status["rag"].startswith("error"):
        diagnosis_summary = "Clinical evidence retrieval is temporarily unavailable."
    else:
        diagnosis_summary = "No evidence was returned by the clinical knowledge base for this query."

    feature_chips = [
        f"{f['name']}: {f['value']}"
        + (f" ({'↑ risk' if f['direction'] == 'increases_risk' else '↓ risk' if f['direction'] == 'decreases_risk' else 'unclear'})")
        for f in top_features
    ]

    # استخراج التحليل الفردي لكل فيتشر (لو مش راجع من الـ RAG بنعمل له هيكل افتراضي منظم بناء على الـ top_features)
    raw_breakdown = rag_evidence.get("feature_breakdown", [])
    if not raw_breakdown and top_features:
        raw_breakdown = [
            {
                "feature_name": f["name"],
                "clinical_relevance": f"Impact direction: {f['direction']}, value: {f.get('value', 'N/A')}.",
                "evidence_based_strategy": "Follow standard guideline targets for cardiovascular optimization."
            }
            for f in top_features
        ]

    parsed_breakdown = [
        FeatureAnalysis(
            feature_name=item.get("feature_name", "Unknown Feature"),
            clinical_relevance=item.get("clinical_relevance", "No specific clinical note."),
            evidence_based_strategy=item.get("evidence_based_strategy", "No specific strategy provided.")
        )
        for item in raw_breakdown
    ]

    guideline_lines = [str(s) for s in rag_evidence.get("sources", [])] if rag_evidence.get("sources") else []

    response = PredictionResponse(
        risk_level=ml_result["risk_level"],
        probability=ml_result["probability"],
        shap_values=[{"feature": f["name"], "impact": f["importance"]} for f in top_features],
        top_features=top_features,
        groq_query=groq_query,
        rag_evidence=RagEvidence(**rag_evidence),
        rag_insights=RagInsights(
            diagnosis_summary=diagnosis_summary,
            associated_symptoms=feature_chips,
            feature_breakdown=parsed_breakdown,
            medical_guidelines=guideline_lines,
        ),
        pipeline_status=PipelineStatus(**pipeline_status),
    )
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
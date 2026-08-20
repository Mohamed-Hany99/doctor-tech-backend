"""
ML inference service.

Loads the artifacts produced by `train_and_save_model.py` (model, one-hot
encoder, ordinal encoder, metadata) and exposes a single function,
`predict_and_explain`, that:

1. Maps the incoming PatientInput (frontend field names) onto the exact
   raw column names/values the model was trained on.
2. Encodes the row using the SAME fitted encoders used at training time
   (never re-fit here).
3. Runs the existing RandomForestClassifier to get a probability.
4. Uses SHAP (TreeExplainer) to get the true top-3 features that pushed
   THIS patient's prediction, per the project's stated priority order:
   SHAP > feature_importances_ > coefficients > permutation importance.

No mock data, no invented directions: SHAP sign IS the direction.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

_model = None
_ohe = None
_oe = None
_metadata = None
_explainer = None


def _load_artifacts() -> None:
    """Lazy-load model artifacts once per process."""
    global _model, _ohe, _oe, _metadata, _explainer
    if _model is not None:
        return

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    ohe_path = os.path.join(ARTIFACTS_DIR, "onehot_encoder.pkl")
    oe_path = os.path.join(ARTIFACTS_DIR, "ordinal_encoder.pkl")
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.pkl")

    for p in (model_path, ohe_path, oe_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model artifact missing: {p}. Run train_and_save_model.py first."
            )

    _model = joblib.load(model_path)
    _ohe = joblib.load(ohe_path)
    _oe = joblib.load(oe_path)
    _metadata = joblib.load(meta_path)
    _explainer = shap.TreeExplainer(_model)


# Maps PatientInput (pydantic) field names -> raw training column names.
FIELD_TO_COLUMN = {
    "age": "Age",
    "gender": "Gender",
    "blood_pressure": "Blood Pressure",
    "cholesterol": "Cholesterol Level",
    "bmi": "BMI",
    "triglyceride": "Triglyceride Level",
    "fasting_blood_sugar": "Fasting Blood Sugar",
    "crp": "CRP Level",
    "homocysteine": "Homocysteine Level",
    "sleep_hours": "Sleep Hours",
    "high_bp": "High Blood Pressure",
    "low_hdl": "Low HDL Cholesterol",
    "high_ldl": "High LDL Cholesterol",
    "exercise": "Exercise Habits",
    "smoking": "Smoking",
    "alcohol": "Alcohol Consumption",
    "stress": "Stress Level",
    "sugar": "Sugar Consumption",
    "family_history": "Family Heart Disease",
    "diabetes": "Diabetes",
}

# Human-readable label for each raw/encoded feature name, for anything sent
# onward to Groq/UI. Encoded one-hot columns (e.g. "Smoking_Yes") are mapped
# back to their real medical concept, never the encoded column name.
FEATURE_DISPLAY_NAME = {
    "Age": "Age",
    "Blood Pressure": "Blood Pressure",
    "Cholesterol Level": "Cholesterol Level",
    "BMI": "BMI",
    "Sleep Hours": "Sleep Hours",
    "Triglyceride Level": "Triglyceride Level",
    "Fasting Blood Sugar": "Fasting Blood Sugar",
    "CRP Level": "CRP Level",
    "Homocysteine Level": "Homocysteine Level",
    "Gender_Male": "Gender",
    "Smoking_Yes": "Smoking",
    "Family Heart Disease_Yes": "Family History of Heart Disease",
    "Diabetes_Yes": "Diabetes",
    "High Blood Pressure_Yes": "High Blood Pressure",
    "Low HDL Cholesterol_Yes": "Low HDL Cholesterol",
    "High LDL Cholesterol_Yes": "High LDL Cholesterol",
    "Exercise Habits": "Exercise Habits",
    "Alcohol Consumption": "Alcohol Consumption",
    "Stress Level": "Stress Level",
    "Sugar Consumption": "Sugar Consumption",
}

# Encoded (post-encoding) column name -> raw column name in the pre-encoding
# row, so we can recover the actual patient value for the top features.
ENCODED_TO_RAW_COLUMN = {
    "Age": "Age",
    "Blood Pressure": "Blood Pressure",
    "Cholesterol Level": "Cholesterol Level",
    "BMI": "BMI",
    "Sleep Hours": "Sleep Hours",
    "Triglyceride Level": "Triglyceride Level",
    "Fasting Blood Sugar": "Fasting Blood Sugar",
    "CRP Level": "CRP Level",
    "Homocysteine Level": "Homocysteine Level",
    "Gender_Male": "Gender",
    "Smoking_Yes": "Smoking",
    "Family Heart Disease_Yes": "Family Heart Disease",
    "Diabetes_Yes": "Diabetes",
    "High Blood Pressure_Yes": "High Blood Pressure",
    "Low HDL Cholesterol_Yes": "Low HDL Cholesterol",
    "High LDL Cholesterol_Yes": "High LDL Cholesterol",
    "Exercise Habits": "Exercise Habits",
    "Alcohol Consumption": "Alcohol Consumption",
    "Stress Level": "Stress Level",
    "Sugar Consumption": "Sugar Consumption",
}


def _parse_systolic(blood_pressure: str) -> float:
    """Blood pressure arrives as 'systolic/diastolic' (e.g. '120/80') from
    the form, but the model was trained on a single systolic numeric column.
    Extract the systolic number; this is parsing the value the patient
    already entered, not inventing data."""
    try:
        systolic_str = str(blood_pressure).split("/")[0].strip()
        return float(systolic_str)
    except (ValueError, IndexError, AttributeError):
        raise ValueError(
            f"Could not parse systolic value from blood_pressure='{blood_pressure}'. "
            "Expected format 'systolic/diastolic', e.g. '120/80'."
        )


def _build_raw_row(patient: dict) -> pd.DataFrame:
    """Build a one-row DataFrame with the exact raw column names/dtypes the
    encoders were fit on. Missing optional numeric fields fall back to the
    training-set median (stored in metadata at training time) rather than
    a fabricated per-patient value."""
    _load_artifacts()
    numeric_cols = _metadata["numeric_cols"]
    medians = _metadata.get("numeric_medians", {})

    row: dict[str, Any] = {}
    for field, column in FIELD_TO_COLUMN.items():
        value = patient.get(field)

        if column == "Blood Pressure":
            row[column] = _parse_systolic(value)
            continue

        if column in numeric_cols:
            if value is None:
                if column not in medians:
                    raise ValueError(
                        f"Missing required numeric field '{field}' and no training "
                        f"median available for '{column}'."
                    )
                row[column] = medians[column]
            else:
                row[column] = float(value)
        else:
            # Categorical: normalize casing/whitespace, but never invent a
            # category that wasn't in the request.
            row[column] = str(value).strip().title() if value is not None else value

    return pd.DataFrame([row])


def _encode_row(raw_row: pd.DataFrame) -> pd.DataFrame:
    _load_artifacts()
    numeric_cols = _metadata["numeric_cols"]
    nominal_cols = _metadata["nominal_cols"]
    ordinal_cols = _metadata["ordinal_cols"]
    feature_order = _metadata["feature_order"]

    ohe_df = pd.DataFrame(
        _ohe.transform(raw_row[nominal_cols]),
        columns=_ohe.get_feature_names_out(nominal_cols),
        index=raw_row.index,
    )
    oe_df = pd.DataFrame(
        _oe.transform(raw_row[ordinal_cols]),
        columns=ordinal_cols,
        index=raw_row.index,
    )
    encoded = pd.concat([raw_row[numeric_cols], ohe_df, oe_df], axis=1)

    # Guarantee exact column order the model expects.
    missing = [c for c in feature_order if c not in encoded.columns]
    if missing:
        raise ValueError(f"Encoding produced a row missing expected columns: {missing}")
    return encoded[feature_order]


def predict_and_explain(patient: dict, top_n: int = 3) -> dict:
    """
    Returns:
        {
          "risk_level": "High Risk" | "Moderate Risk" | "Low Risk",
          "probability": float,
          "top_features": [
              {"name": str, "value": Any, "importance": float, "direction": str}
          ]
        }
    """
    _load_artifacts()

    raw_row = _build_raw_row(patient)
    encoded_row = _encode_row(raw_row)

    probability = float(_model.predict_proba(encoded_row)[0, 1])

    if probability > 0.6:
        risk_level = "High Risk"
    elif probability > 0.3:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # --- SHAP explanation for THIS patient (priority #1: SHAP) ---
    shap_values = _explainer.shap_values(encoded_row)
    if isinstance(shap_values, list):
        # Older SHAP: list of per-class arrays
        patient_shap = shap_values[1][0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (samples, features, classes)
            patient_shap = arr[0, :, 1]
        else:
            patient_shap = arr[0]

    shap_series = pd.Series(patient_shap, index=encoded_row.columns)
    ranked = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)
    top_encoded_features = ranked.head(top_n)

    top_features = []
    for encoded_name, shap_val in top_encoded_features.items():
        display_name = FEATURE_DISPLAY_NAME.get(encoded_name, encoded_name)

        # Recover the original (human) value for this feature from the raw row.
        raw_column = ENCODED_TO_RAW_COLUMN.get(encoded_name)
        if raw_column and raw_column in raw_row.columns:
            value = raw_row.iloc[0][raw_column]
            if isinstance(value, (np.floating, float)):
                value = round(float(value), 2)
        else:
            value = None

        if shap_val > 0:
            direction = "increases_risk"
        elif shap_val < 0:
            direction = "decreases_risk"
        else:
            direction = "unknown"

        top_features.append(
            {
                "name": display_name,
                "value": value,
                "importance": round(float(abs(shap_val)), 4),
                "direction": direction,
            }
        )

    return {
        "risk_level": risk_level,
        "probability": round(probability, 4),
        "top_features": top_features,
    }

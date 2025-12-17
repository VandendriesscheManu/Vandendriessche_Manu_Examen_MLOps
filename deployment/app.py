from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os

app = FastAPI(title="GOT House Predictor")

MODEL_PATH = "model.joblib"
FEATURES_PATH = "feature_columns.json"

model = joblib.load(MODEL_PATH)

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        f"Missing {FEATURES_PATH}. "
        "Copy it from AzureML train_dt output (model_output) or save it during training."
    )

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    TRAIN_COLUMNS = json.load(f)


class CharacterInput(BaseModel):
    region: str
    primary_role: str
    alignment: str
    status: str
    species: str
    honour_1to5: int
    ruthlessness_1to5: int
    intelligence_1to5: int
    combat_skill_1to5: int
    diplomacy_1to5: int
    leadership_1to5: int
    trait_loyal: bool
    trait_scheming: bool


def to_model_frame(payload: CharacterInput) -> pd.DataFrame:
    # 1 row dataframe from input
    df = pd.DataFrame([payload.model_dump()])

    # bool -> int (optioneel maar netjes)
    for c in ["trait_loyal", "trait_scheming"]:
        df[c] = df[c].astype(int)

    # One-hot encode categoricals (moet matchen met training)
    df = pd.get_dummies(
        df,
        columns=["region", "primary_role", "alignment", "status", "species"],
        dtype=int
    )

    # Align columns with training feature set (missing -> 0, extra -> drop)
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    return df


@app.post("/predict")
def predict_house(data: CharacterInput):
    X = to_model_frame(data)
    pred = model.predict(X)[0]

    # (optioneel) ook probabilities indien beschikbaar
    probs = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        probs = {str(cls): float(val) for cls, val in zip(model.classes_, p)}

    return {
        "predicted_house": pred,
        "probabilities": probs
    }

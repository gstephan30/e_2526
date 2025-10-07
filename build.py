# build.py
# Predict match goal differences with a robust statsmodels pipeline
# - Fits OLS on historical results
# - Predicts upcoming fixtures
# - FIX: aligns prediction exog to model.exog_names (handles intercept & dummies)

import os
import json
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------ Config -------------
HIST_PATH = os.getenv("HIST_PATH", "data/results.csv")     # needs columns: heim, auswärts, tore_heim, tore_auswärts, (optional) date
FIXTURES_PATH = os.getenv("FIXTURES_PATH", "data/fixtures.csv")  # needs columns: heim, auswärts
OUT_DIR = os.getenv("OUT_DIR", "build")
PREDICTIONS_CSV = os.path.join(OUT_DIR, "predictions.csv")
MODEL_META_JSON = os.path.join(OUT_DIR, "model_meta.json")

os.makedirs(OUT_DIR, exist_ok=True)

# ------------ Helpers -------------
def _validate_columns(df: pd.DataFrame, required: list, name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}. "
                         f"Available: {list(df.columns)}")

def _make_design_matrix(df: pd.DataFrame, home_col: str, away_col: str,
                        all_teams: list, drop_first: bool = True) -> pd.DataFrame:
    """
    Create a stable design matrix for home/away team fixed effects.
    Uses a fixed category list to ensure identical dummy columns at train & predict.
    """
    # Lock categories so get_dummies yields stable columns
    home = pd.Categorical(df[home_col], categories=all_teams, ordered=True)
    away = pd.Categorical(df[away_col], categories=all_teams, ordered=True)
    H = pd.get_dummies(home, prefix="home", drop_first=drop_first)
    A = pd.get_dummies(away, prefix="away", drop_first=drop_first)
    X = pd.concat([H, A], axis=1)
    # Add constant last; we’ll reorder later against model.exog_names anyway
    X = sm.add_constant(X, has_constant="add")
    return X

def _align_to_exog_names(X: pd.DataFrame, exog_names: list) -> pd.DataFrame:
    # Intercept naming normalization
    cols = list(X.columns)
    if "Intercept" in exog_names and "const" in cols:
        X = X.rename(columns={"const": "Intercept"})
    if "const" in exog_names and "Intercept" in cols:
        X = X.rename(columns={"Intercept": "const"})
    # Reindex to training columns; fill missing with 0 (unseen categories)
    X = X.reindex(columns=exog_names, fill_value=0)
    return X

# ------------ Load data -------------
hist = pd.read_csv(HIST_PATH)
_validate_columns(hist, ["heim", "auswärts", "tore_heim", "tore_auswärts"], "results")

fixtures = pd.read_csv(FIXTURES_PATH)
_validate_columns(fixtures, ["heim", "auswärts"], "fixtures")

# ------------ Train -------------
hist = hist.copy()
hist["goal_diff"] = hist["tore_heim"] - hist["tore_auswärts"]

# Known teams universe (stable order)
teams = sorted(pd.unique(pd.concat([hist["heim"], hist["auswärts"]], axis=0)))

# Design matrix for training
X_train = _make_design_matrix(hist, "heim", "auswärts", all_teams=teams, drop_first=True)
y_train = hist["goal_diff"].astype(float)

# Fit OLS
res = sm.OLS(y_train, X_train).fit()

# Persist minimal model meta for debugging / reuse
model_meta = {
    "exog_names": res.model.exog_names,
    "teams": teams,
    "drop_first": True,
    "n_params": int(len(res.params)),
}
with open(MODEL_META_JSON, "w", encoding="utf-8") as f:
    json.dump(model_meta, f, ensure_ascii=False, indent=2)

# ------------ Predict -------------
def predict_match(heim: str, auswaerts: str) -> float:
    row = pd.DataFrame([{"heim": heim, "auswärts": auswaerts}])
    X = _make_design_matrix(row, "heim", "auswärts", all_teams=teams, drop_first=True)
    X = _align_to_exog_names(X, res.model.exog_names)
    # Safety check (prevents shape mismatch crash)
    if X.shape[1] != len(res.params):
        raise RuntimeError(f"Design matrix mismatch at predict: X has {X.shape[1]} cols, "
                           f"model expects {len(res.params)}")
    pred = res.predict(X)[0]
    return float(pred)

pred_rows = []
for idx, row in fixtures.iterrows():
    h = row["heim"]
    a = row["auswärts"]
    try:
        gd = predict_match(h, a)
        # Optional: convert goal diff to simple win/draw/loss probabilities (logit-ish heuristic)
        # Here we just keep the raw expected difference; you can map to probabilities externally.
        pred_rows.append({
            "heim": h,
            "auswärts": a,
            "exp_goal_diff": round(gd, 3),
        })
    except Exception as e:
        pred_rows.append({
            "heim": h,
            "auswärts": a,
            "exp_goal_diff": np.nan,
            "error": str(e),
        })

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(PREDICTIONS_CSV, index=False)

# ------------ Minimal console output -------------
print(f"Trained OLS with {len(res.params)} params on {len(hist)} matches.")
print(f"Saved predictions to {PREDICTIONS_CSV}")
if pred_df.get("error", pd.Series()).notna().any():
    n_err = pred_df["error"].notna().sum()
    print(f"NOTE: {n_err} prediction rows had alignment issues (see CSV 'error' column).")

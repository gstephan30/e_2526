import os
import shutil
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from jinja2 import Environment, FileSystemLoader

# === Pfade ===
DATA_DIR = "data"
DATA_CSV = os.path.join(DATA_DIR, "results.csv")  # Passe an, falls dein CSV anders heißt
TEMPLATE_DIR = "templates"
TEMPLATE_INDEX = "index.html"
OUTPUT_DIR = "docs"
STATIC_DIR = "static"

# === Laden der Datendatei ===
def load_data():
    df = pd.read_csv(DATA_CSV)
    df["tore_heim"] = pd.to_numeric(df.get("tore_heim"), errors="coerce")
    df["tore_auswärts"] = pd.to_numeric(df.get("tore_auswärts"), errors="coerce")
    return df

# === Tabelle erzeugen ===
def compute_tabelle(df_played):
    teams = sorted(set(df_played["heim"].dropna().tolist() + df_played["auswärts"].dropna().tolist()))
    tbl = pd.DataFrame(0, index=teams, columns=["spiele","siege","unentschieden","niederlagen","tore","geg"])
    for _, row in df_played.iterrows():
        hm = row["heim"]
        aw = row["auswärts"]
        th = int(row["tore_heim"])
        ta = int(row["tore_auswärts"])
        tbl.at[hm, "spiele"] += 1
        tbl.at[aw, "spiele"] += 1
        tbl.at[hm, "tore"] += th
        tbl.at[hm, "geg"] += ta
        tbl.at[aw, "tore"] += ta
        tbl.at[aw, "geg"] += th
        if th > ta:
            tbl.at[hm, "siege"] += 1
            tbl.at[aw, "niederlagen"] += 1
        elif th == ta:
            tbl.at[hm, "unentschieden"] += 1
            tbl.at[aw, "unentschieden"] += 1
        else:
            tbl.at[aw, "siege"] += 1
            tbl.at[hm, "niederlagen"] += 1
    tbl["punkte"] = 3 * tbl["siege"] + tbl["unentschieden"]
    tabelle_list = []
    for team in tbl.index:
        tabelle_list.append({
            "team": team,
            "spiele": int(tbl.at[team, "spiele"]),
            "punkte": int(tbl.at[team, "punkte"]),
            "tore": int(tbl.at[team, "tore"]),
            "geg": int(tbl.at[team, "geg"])
        })
    tabelle_list.sort(key=lambda x: (x["punkte"], x["tore"] - x["geg"]), reverse=True)
    return tabelle_list

# === Modelltraining ===
def train_model(df_played):
    teams = sorted(set(df_played["heim"].dropna().tolist() + df_played["auswärts"].dropna().tolist()))
    idx = {team: i for i, team in enumerate(teams)}
    n = len(df_played)
    # Dummy-Matrizen
    Xh = np.zeros((n, len(teams)))
    Xa = np.zeros((n, len(teams)))
    for i, row in enumerate(df_played.itertuples()):
        Xh[i, idx[row.heim]] = 1
        Xa[i, idx[row.auswärts]] = 1
    Xdiff = Xh - Xa
    X = sm.add_constant(Xdiff)  # fügt konstante Spalte hinzu → Spaltenanzahl = 1 + len(teams)
    y_diff = (df_played["tore_heim"] - df_played["tore_auswärts"]).values
    model = sm.OLS(y_diff, X)
    res = model.fit()
    return res, teams

# === Prognose für ein Spiel ===
def predict_match(res, teams_model, home, away):
    if home not in teams_model or away not in teams_model:
        return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3, "lam_h": None, "lam_a": None}
    # Ensure the exogenous vector has the same length as the fitted params
    # res.params corresponds to [const, team_0, team_1, ..., team_{m-1}]
    p_len = len(res.params)
    expected_teams = p_len - 1
    vec = np.zeros(expected_teams)
    # Safe mapping: only set indices that exist within the expected size
    idx_home = teams_model.index(home)
    idx_away = teams_model.index(away)
    if idx_home < expected_teams:
        vec[idx_home] = 1
    if idx_away < expected_teams:
        vec[idx_away] = -1
    # Build Xrow as [1, vec...]
    Xrow = np.concatenate(([1.0], vec)).reshape(1, -1)
    # Compute predicted difference via dot product to avoid statsmodels shape issues
    est_diff = float(np.dot(Xrow, res.params))
    total_lambda = 5.0
    lam_h = max(0.1, (total_lambda + est_diff) / 2)
    lam_a = max(0.1, total_lambda - lam_h)
    max_goals = 8
    p_home = p_draw = p_away = 0.0
    for k in range(max_goals + 1):
        for m in range(max_goals + 1):
            pk = poisson.pmf(k, lam_h) * poisson.pmf(m, lam_a)
            if k > m:
                p_home += pk
            elif k == m:
                p_draw += pk
            else:
                p_away += pk
    return {"home_win": p_home, "draw": p_draw, "away_win": p_away, "lam_h": lam_h, "lam_a": lam_a}

# === Hauptfunktion, Bau der Seite ===
def build_site():
    df = load_data()
    df_played = df.dropna(subset=["tore_heim", "tore_auswärts"]).copy()
    tabelle_list = compute_tabelle(df_played)
    res, teams_model = train_model(df_played)
    df_future = df[df["tore_heim"].isna() & df["heim"].notna() & df["auswärts"].notna()].copy()
    predictions = []
    for _, row in df_future.iterrows():
        p = predict_match(res, teams_model, row["heim"], row["auswärts"])
        predictions.append({
            "spielnr": int(row["spielnr"]),
            "heim": row["heim"],
            "auswärts": row["auswärts"],
            "home_win": p["home_win"],
            "draw": p["draw"],
            "away_win": p["away_win"]
        })
    spiele_history = df_played.to_dict(orient="records")
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_INDEX)
    html = template.render(
        title="Prognose & Spielplan",
        predictions=predictions,
        tabelle=tabelle_list,
        spiele_history=spiele_history
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Erstellt {output_path}")
    
    # Kopiere static-Dateien nach docs/static
    output_static = os.path.join(OUTPUT_DIR, "static")
    if os.path.exists(STATIC_DIR):
        if os.path.exists(output_static):
            shutil.rmtree(output_static)
        shutil.copytree(STATIC_DIR, output_static)
        print(f"Kopiert {STATIC_DIR} → {output_static}")

if __name__ == "__main__":
    build_site()

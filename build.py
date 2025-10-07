import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from jinja2 import Environment, FileSystemLoader

# --- Konfiguration ---
DATA_DIR = "data"
SPIELE_CSV = os.path.join(DATA_DIR, "spiele.csv")
HIST_CSV = os.path.join(DATA_DIR, "results.csv")  # optional, für historische Daten

TEMPLATE_DIR = "templates"
OUTPUT_DIR = "docs"
TEMPLATE_INDEX = "index.html"

# --- Funktionen ---

def load_spiele():
    df = pd.read_csv(SPIELE_CSV)
    # Konvertiere Tore zu numerisch, falls möglich
    df["tore_heim"] = pd.to_numeric(df.get("tore_heim"), errors="coerce")
    df["tore_auswärts"] = pd.to_numeric(df.get("tore_auswärts"), errors="coerce")
    return df

def compute_tabelle(df_played):
    # teams sammeln
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
    tbl["punkte"] = 3 * tbl["siege"] + 1 * tbl["unentschieden"]
    # Umwandlung in Liste von Dicts für Template
    tabelle_list = []
    for team in tbl.index:
        tabelle_list.append({
            "team": team,
            "spiele": int(tbl.at[team, "spiele"]),
            "punkte": int(tbl.at[team, "punkte"]),
            "tore": int(tbl.at[team, "tore"]),
            "geg": int(tbl.at[team, "geg"]),
        })
    # sortieren: Punkte absteigend, bei Gleichstand Tordifferenz
    tabelle_list.sort(key=lambda x: (x["punkte"], x["tore"] - x["geg"]), reverse=True)
    return tabelle_list

def train_model(df_played):
    # Modell auf Tordifferenz (Heim minus Auswärts) schätzen
    # Designmatrix: Dummy für Heim vs Auswärts
    teams = sorted(set(df_played["heim"].dropna().tolist() + df_played["auswärts"].dropna().tolist()))
    idx = {team: i for i, team in enumerate(teams)}
    n = df_played.shape[0]
    Xh = np.zeros((n, len(teams)))
    Xa = np.zeros((n, len(teams)))
    for i, row in enumerate(df_played.itertuples()):
        Xh[i, idx[row.heim]] = 1
        Xa[i, idx[row.auswärts]] = 1
    y_diff = (df_played["tore_heim"] - df_played["tore_auswärts"]).values
    Xdiff = Xh - Xa
    model = sm.OLS(y_diff, sm.add_constant(Xdiff))
    res = model.fit()
    return res, teams

def predict_match(res, teams_model, home, away):
    # Prognose der Wahrscheinlichkeiten für ein Match
    if home not in teams_model or away not in teams_model:
        # Fallback, wenn unbekannte Mannschaft
        return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3, "lam_h": None, "lam_a": None}
    # Baue Dummy-Vektor
    vec = np.zeros(len(teams_model))
    vec[teams_model.index(home)] = 1
    vec[teams_model.index(away)] = -1
    Xrow = sm.add_constant(pd.DataFrame([vec])).values
    est_diff = res.predict(Xrow)[0]
    total_lambda = 5.0
    lam_h = max(0.1, (total_lambda + est_diff) / 2)
    lam_a = max(0.1, total_lambda - lam_h)
    # Berechne P mit Poisson-Ansatz
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
    return {
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
        "lam_h": lam_h,
        "lam_a": lam_a
    }

def build_site():
    # Lade alle Spiele
    df = load_spiele()
    # Spiele mit Ergebnis
    df_played = df.dropna(subset=["tore_heim", "tore_auswärts"]).copy()

    # Tabelle berechnen
    tabelle_list = compute_tabelle(df_played)

    # Modell trainieren
    res, teams_model = train_model(df_played)

    # Prognosen für kommende Spiele
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

    # Historical Spiele zur Anzeige
    spiele_history = df_played.to_dict(orient="records")

    # Template rendern
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_INDEX)
    html = template.render(
        title = "Prognose & Spielplan",
        predictions = predictions,
        tabelle = tabelle_list,
        spiele_history = spiele_history
    )

    # Output schreiben
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Erstellt {output_path}")

if __name__ == "__main__":
    build_site()

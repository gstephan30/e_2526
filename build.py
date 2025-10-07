import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from jinja2 import Environment, FileSystemLoader

# --- 1. Lade Daten ---
df = pd.read_csv("data/spiele.csv")

# Typkonvertierung für vorhandene Tore
df["tore_heim"] = pd.to_numeric(df["tore_heim"], errors="coerce")
df["tore_auswärts"] = pd.to_numeric(df["tore_auswärts"], errors="coerce")

# Gespielte Spiele (mit Ergebnis)
df_played = df.dropna(subset=["tore_heim", "tore_auswärts"]).copy()

# --- 2. Tabelle / Punktestand berechnen ---
teams = sorted(set(df["heim"].dropna().tolist() + df["auswärts"].dropna().tolist()))
# DataFrame für Tabelle initialisieren
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
# Für Template: Liste von Dikt mit team, spiele, tore, geg, punkte
tabelle_list = []
for team in tbl.index:
    tabelle_list.append({
        "team": team,
        "spiele": int(tbl.at[team, "spiele"]),
        "punkte": int(tbl.at[team, "punkte"]),
        "tore": int(tbl.at[team, "tore"]),
        "geg": int(tbl.at[team, "geg"]),
    })
# Sortieren nach Punkte (absteigend), dann Torverhältnis
tabelle_list.sort(key=lambda x: (x["punkte"], x["tore"] - x["geg"]), reverse=True)

# --- 3. Prognosen für nächsten Spieltag ---
# Wir definieren eine Funktion wie vorher
# (Ich benutzte das Differenzmodell in deinem früheren Python-Beispiel.)

# Bau Design-Matrizen
def make_design(df_in):
    teams_in = sorted(set(df_in["heim"].dropna().tolist() + df_in["auswärts"].dropna().tolist()))
    idx = {team: i for i, team in enumerate(teams_in)}
    Xh = np.zeros((len(df_in), len(teams_in)))
    Xa = np.zeros((len(df_in), len(teams_in)))
    for i, row in enumerate(df_in.itertuples()):
        Xh[i, idx[getattr(row, "heim")]] = 1
        Xa[i, idx[getattr(row, "auswärts")]] = 1
    return Xh, Xa, teams_in

Xh, Xa, teams_model = make_design(df_played)
y_diff = (df_played["tore_heim"] - df_played["tore_auswärts"]).values

# Differenzmodell (OLS)
Xdiff = Xh - Xa
model = sm.OLS(y_diff, sm.add_constant(Xdiff))
res = model.fit()

def predict_match(home, away):
    if home not in teams_model or away not in teams_model:
        # unbekannte Mannschaft — fallback
        return {"home_win": 0.33, "draw": 0.33, "away_win": 0.33, "lam_h": None, "lam_a": None}
    Xrow = np.zeros(len(teams_model))
    Xrow[teams_model.index(home)] = 1
    Xrow[teams_model.index(away)] = -1
    Xmat = sm.add_constant(pd.DataFrame([Xrow])).values
    pred_diff = res.predict(Xmat)[0]
    total_lambda = 5.0
    lam_h = max(0.1, (total_lambda + pred_diff) / 2)
    lam_a = max(0.1, total_lambda - lam_h)
    # Berechnung Wahrscheinlichkeiten
    max_goals = 8
    p_home = p_draw = p_away = 0.0
    for k in range(max_goals + 1):
        for m in range(max_goals + 1):
            p = poisson.pmf(k, lam_h) * poisson.pmf(m, lam_a)
            if k > m:
                p_home += p
            elif k == m:
                p_draw += p
            else:
                p_away += p
    return {"home_win": p_home, "draw": p_draw, "away_win": p_away, "lam_h": lam_h, "lam_a": lam_a}

# Alle zukünftigen Spiele (ohne Tore)
df_future = df[df["tore_heim"].isna() & df["heim"].notna() & df["auswärts"].notna()].copy()
predictions = []
for _, row in df_future.iterrows():
    p = predict_match(row["heim"], row["auswärts"])
    rec = {
        "spielnr": int(row["spielnr"]),
        "heim": row["heim"],
        "auswärts": row["auswärts"],
        "home_win": p["home_win"],
        "draw": p["draw"],
        "away_win": p["away_win"]
    }
    predictions.append(rec)

# --- 4. Template rendern ---
env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("index.html")

html = template.render(
    title = "Fußball Prognose & Spielplan",
    predictions = predictions,
    tabelle = tabelle_list,
    spiele_history = df_played.to_dict(orient="records")
)

# Stelle sicher, dass docs/ existiert
os.makedirs("docs", exist_ok=True)
with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Erzeugt docs/index.html")

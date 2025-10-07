# E-Junioren Saison 25/26 - Prognose & Spielplan

Willkommen zum automatisierten Prognose- und Spielplan-System für die U11 (E-Junioren) Ruperti Gruppe 03 der Saison 25/26.

## Über das Projekt

Dieses Projekt generiert eine statische Webseite mit:
- **Ergebnissen** bisheriger Spiele
- **Aktueller Tabelle** der Staffel
- **Prognosen** für kommende Spiele basierend auf statistischen Modellen

## Funktionen

### 📜 Ergebnisse
Zeigt alle bisher gespielten Spiele mit Datum, Teams und Endergebnis.

### 📊 Tabelle
Aktuelle Tabellenübersicht mit Punkten, Tordifferenz und weiteren Statistiken.

### ⚽ Prognose
Vorhersage von Spielausgängen für den nächsten Spieltag mit Wahrscheinlichkeiten für:
- Heimsieg
- Unentschieden
- Auswärtssieg

## Technologie

Das Projekt verwendet:
- **Python 3** für Datenverarbeitung und Modellierung
- **Pandas** und **NumPy** für Datenanalyse
- **Statsmodels** und **SciPy** für statistische Modelle
- **Jinja2** für Template-Rendering
- **HTML/CSS/JavaScript** für die Weboberfläche

## Installation

1. Repository klonen:
```bash
git clone https://github.com/gstephan30/e_2526.git
cd e_2526
```

2. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

3. Webseite generieren:
```bash
python build.py
```

Die generierte Webseite befindet sich im `docs/` Verzeichnis.

## Datenstruktur

Die Spieldaten werden in `data/results.csv` gespeichert mit folgenden Spalten:
- `spielnr`: Spielnummer
- `spieltag`: Spieltag
- `zeit`: Datum und Uhrzeit
- `heim`: Heimmannschaft
- `auswärts`: Auswärtsmannschaft
- `tore_heim`: Tore der Heimmannschaft
- `tore_auswärts`: Tore der Auswärtsmannschaft
- `kommentar`: Optionale Kommentare

## Modell-Dokumentation

Detaillierte Informationen zum verwendeten statistischen Modell finden Sie auf der [Dokumentationsseite](docs/dokumentation.html).

## Lizenz

Dieses Projekt dient ausschließlich informativen Zwecken für die U11 E-Junioren Saison 25/26.

## Kontakt

Für weitere Informationen siehe [Impressum](docs/impressum.html).

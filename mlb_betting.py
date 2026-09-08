"""
MLB Betting ML System - Pro Edition
===================================
Sistema de Machine Learning para proyección de mercados de apuestas MLB.

Novedades:
  - Integración con Odds-API.io para cuotas reales en vivo y cálculo de Expected Value (EV%).
  - Hidratación de abridores confirmados (Probable Pitchers) desde MLB Stats API.
  - Ajuste por Factores de Parque (Park Factor Adjustment).
  - Generación de Picks EV+ con recomendación de Stake.

Uso:
    python mlb_betting.py [AAAA-MM-DD]
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# ---------------------------------------------------------------
# Configuración y Constantes
# ---------------------------------------------------------------
SEASON = 2026
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "output"
WINDOWS = [3, 5, 10]

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://odds-api.io/api/v1/odds"

HITTING = ["runs", "hits", "doubles", "triples", "homeRuns", "totalBases",
           "strikeOuts", "baseOnBalls", "walk", "atBats"]

PITCHING = ["runs", "hits", "strikeOuts", "baseOnBalls", "homeRuns",
            "inningsPitched", "era", "whip"]

TARGETS = [
    "hom_runs", "away_runs", "total_runs",
    "hom_hits", "away_hits",
    "hom_total_bases", "away_total_bases",
    "hom_k", "away_k",
    "hom_p_k", "away_p_k",
    "total_k",
    "hom_win",
]

API_BASE = "https://statsapi.mlb.com/api/v1"
CACHE_TTL = 6 * 3600  # 6 horas

# Factores de Parque MLB (2025/2026 Baseline Index — 1.00 es Neutral)
# Impacta la producción de Runs, Hits y HRs
PARK_FACTORS = {
    "Colorado Rockies": 1.35,       # Coors Field
    "Boston Red Sox": 1.08,         # Fenway Park
    "Cincinnati Reds": 1.12,        # Great American Ball Park
    "Philadelphia Phillies": 1.06,  # Citizens Bank Park
    "Kansas City Royals": 1.05,     # Kauffman Stadium
    "Chicago Cubs": 1.04,           # Wrigley Field
    "Baltimore Orioles": 1.02,      # Oriole Park
    "Texas Rangers": 1.02,          # Globe Life Field
    "Los Angeles Dodgers": 1.01,    # Dodger Stadium
    "Atlanta Braves": 1.01,         # Truist Park
    "Minnesota Twins": 1.00,        # Target Field
    "Chicago White Sox": 1.00,      # Guaranteed Rate Field
    "St. Louis Cardinals": 0.99,    # Busch Stadium
    "Milwaukee Brewers": 0.99,      # American Family Field
    "New York Yankees": 0.98,       # Yankee Stadium
    "Houston Astros": 0.98,         # Minute Maid Park
    "Toronto Blue Jays": 0.98,      # Rogers Centre
    "Arizona Diamondbacks": 0.97,   # Chase Field
    "Washington Nationals": 0.97,   # Nationals Park
    "Los Angeles Angels": 0.96,     # Angel Stadium
    "San Francisco Giants": 0.95,   # Oracle Park
    "Detroit Tigers": 0.95,         # Comerica Park
    "Pittsburgh Pirates": 0.95,     # PNC Park
    "New York Mets": 0.94,          # Citi Field
    "Cleveland Guardians": 0.94,    # Progressive Field
    "Tampa Bay Rays": 0.93,         # Tropicana Field / Temp
    "Miami Marlins": 0.93,          # loanDepot park
    "San Diego Padres": 0.92,       # Petco Park
    "Seattle Mariners": 0.91,       # T-Mobile Park
    "Athletics": 0.96,              # Sutter Health Park / Temp
}


# ---------------------------------------------------------------
# Cliente MLB Stats API y Odds API
# ---------------------------------------------------------------
def _get(url: str, params: dict[str, Any] | None = None) -> dict:
    for intento in range(4):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(10 * (intento + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if intento == 3:
                return {}
            time.sleep(10 * (intento + 1))
    return {}


def _leer_cache(nombre: str) -> pd.DataFrame | None:
    ruta = os.path.join(DATA_DIR, nombre)
    if os.path.exists(ruta) and (time.time() - os.path.getmtime(ruta)) < CACHE_TTL:
        try:
            return pd.read_csv(ruta)
        except Exception:
            return None
    return None


def _guardar_cache(nombre: str, df: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, nombre), index=False)


def _parsear_calendario(payload: dict) -> pd.DataFrame:
    juegos = []
    for dia in payload.get("dates", []):
        for g in dia.get("games", []):
            local = g.get("teams", {}).get("home", {})
            visit = g.get("teams", {}).get("away", {})
            
            # Hidratación de abridores (Probable Pitchers)
            hom_p = local.get("probablePitcher", {}).get("fullName", "TBD")
            away_p = visit.get("probablePitcher", {}).get("fullName", "TBD")
            hom_p_id = local.get("probablePitcher", {}).get("id", None)
            away_p_id = visit.get("probablePitcher", {}).get("id", None)

            juegos.append({
                "game_pk": g.get("gamePk"),
                "game_date": (g.get("gameDate") or "")[:10],
                "status": g.get("status", {}).get("abstractGameState"),
                "hom_team_id": local.get("team", {}).get("id"),
                "away_team_id": visit.get("team", {}).get("id"),
                "hom_team_name": local.get("team", {}).get("name"),
                "away_team_name": visit.get("team", {}).get("name"),
                "hom_pitcher": hom_p,
                "away_pitcher": away_p,
                "hom_pitcher_id": hom_p_id,
                "away_pitcher_id": away_p_id,
                "hom_score": local.get("score"),
                "away_score": visit.get("score"),
            })
    return pd.DataFrame(juegos)


def calendario_temporada() -> pd.DataFrame:
    cache = f"calendario_{SEASON}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{API_BASE}/schedule",
        {"sportId": 1, "season": SEASON, "gameType": "R", "hydrate": "probablePitcher"},
    )
    df = _parsear_calendario(payload)
    if not df.empty:
        _guardar_cache(cache, df)
    return df


def calendario_dia(fecha: str) -> pd.DataFrame:
    cache = f"calendario_{fecha}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{API_BASE}/schedule", 
        {"sportId": 1, "date": fecha, "hydrate": "probablePitcher"}
    )
    df = _parsear_calendario(payload)
    if not df.empty:
        _guardar_cache(cache, df)
    return df


def game_log_equipo(team_id: int, grupo: str) -> pd.DataFrame:
    cache = f"equipo_{team_id}_{SEASON}_{grupo}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{API_BASE}/teams/{team_id}/stats",
        {"season": SEASON, "group": grupo, "stats": "gameLog"},
    )
    filas = []
    for bloque in payload.get("stats", []):
        for split in bloque.get("splits", []):
            fila = {
                "date": (split.get("date") or "")[:10],
                "team_id": split.get("team", {}).get("id"),
                "opponent_id": split.get("opponent", {}).get("id"),
            }
            fila.update(split.get("stat", {}))
            filas.append(fila)
    df = pd.DataFrame(filas)
    if not df.empty:
        _guardar_cache(cache, df)
    return df


def obtener_cuotas_odds_api(fecha: str) -> dict[str, dict[str, float]]:
    """Consulta la API de odds-api.io para cruzar cuotas en vivo."""
    if not ODDS_API_KEY:
        print("  ⚠️ ODDS_API_KEY no configurada. Se omitirá el cálculo de EV%.")
        return {}
    
    try:
        url = f"{ODDS_API_URL}?apiKey={ODDS_API_KEY}&sport=baseball_mlb&date={fecha}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json()
        cuotas = {}
        for game in data.get("data", []):
            home = game.get("home_team")
            away = game.get("away_team")
            key = f"{away} @ {home}"
            bookmakers = game.get("bookmakers", [])
            if bookmakers:
                # Tomamos las cuotas promedio o del primer bookmaker principal
                bm = bookmakers[0]
                markets = bm.get("markets", {})
                h_odds = markets.get("h2h", {}).get("home", 0.0)
                a_odds = markets.get("h2h", {}).get("away", 0.0)
                tot_line = markets.get("totals", {}).get("line", 0.0)
                cuotas[key] = {
                    "hom_ml_odds": float(h_odds),
                    "away_ml_odds": float(a_odds),
                    "total_line": float(tot_line),
                }
        return cuotas
    except Exception as e:
        print(f"  ⚠️ Error consultando Odds-API: {e}")
        return {}


# ---------------------------------------------------------------
# Utilidades Numéricas y Normalización
# ---------------------------------------------------------------
def a_float(x, por_defecto=0.0):
    if x is None:
        return por_defecto
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return por_defecto
    try:
        return float(x)
    except (TypeError, ValueError):
        return por_defecto


def parsear_innings(x):
    if isinstance(x, str) and "." in x:
        entero, _, resto = x.partition(".")
        try:
            return float(entero) + {"0": 0.0, "1": 1/3, "2": 2/3}.get(resto[:1], 0.0)
        except ValueError:
            pass
    return a_float(x)


def normalizar_log(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "team_id", "opponent_id"])
    df = df.sort_values("date").reset_index(drop=True)
    df["team_id"] = df["team_id"].astype(int)
    df["opponent_id"] = df["opponent_id"].astype(int)

    for c in HITTING + PITCHING:
        if c in df.columns:
            df[c] = df[c].apply(a_float)

    if "inningsPitched" in df.columns:
        df["inningsPitched"] = df["inningsPitched"].apply(parsear_innings)
        ip = df["inningsPitched"].replace(0, np.nan)
        df["k_per_9"] = df["strikeOuts"] / (ip / 9)
        df["whip_game"] = (df["hits"] + df["baseOnBalls"]) / ip
        df[["k_per_9", "whip_game"]] = df[["k_per_9", "whip_game"]].fillna(0.0)

    if "atBats" in df.columns:
        df["avg_game"] = df["hits"] / df["atBats"].replace(0, np.nan)
        df["avg_game"] = df["avg_game"].fillna(0.0)

    return df


def características_equipo(df: pd.DataFrame, prefijo: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    cols = [c for c in HITTING + PITCHING + ["k_per_9", "whip_game", "avg_game"]
            if c in df.columns]

    feat: dict[str, Any] = {
        "date": df["date"].values,
        "team_id": df["team_id"].values,
        "opponent_id": df["opponent_id"].values,
    }
    for c in cols:
        for w in WINDOWS:
            feat[f"{prefijo}{c}_avg{w}"] = df[c].shift(1).rolling(w, min_periods=1).mean().values
        feat[f"{prefijo}{c}_season"] = df[c].expanding().mean().shift(1).values
        ult5 = df[c].shift(1).rolling(5, min_periods=1).mean()
        prev10 = df[c].shift(6).rolling(10, min_periods=1).mean()
        feat[f"{prefijo}{c}_trend"] = (ult5 - prev10).values
    return pd.DataFrame(feat)


def ultimas_características(frame: pd.DataFrame, fecha: pd.Timestamp) -> dict:
    if frame is None or frame.empty:
        return {}
    previos = frame[frame["date"] < fecha]
    if previos.empty:
        return {}
    return previos.sort_values("date").iloc[-1].to_dict()


def dias_descanso(df: pd.DataFrame, fecha: pd.Timestamp) -> float:
    if df is None or df.empty:
        return 0.0
    previos = df[df["date"] < fecha]
    if previos.empty:
        return 0.0
    return max(0.0, (fecha - previos["date"].max()).days)


def cargar_equipo(team_id: int) -> dict:
    hitting_raw = game_log_equipo(team_id, "hitting")
    pitching_raw = game_log_equipo(team_id, "pitching")

    hitting = normalizar_log(hitting_raw)
    pitching = normalizar_log(pitching_raw)

    hf = características_equipo(hitting, prefijo="h_")
    pf = características_equipo(pitching, prefijo="p_")

    if hf.empty:
        combinado = pf
    elif pf.empty:
        combinado = hf
    else:
        combinado = pd.merge(hf, pf, on=["date", "team_id", "opponent_id"], how="outer")

    return {"hitting": hitting, "pitching": pitching, "features": combinado}


def cargar_equipos(calendario: pd.DataFrame) -> dict:
    ids = pd.unique(calendario[["hom_team_id", "away_team_id"]].values.ravel())
    equipos = {}
    for tid in ids:
        if pd.isna(tid):
            continue
        try:
            equipos[int(tid)] = cargar_equipo(int(tid))
        except Exception:
            pass
    return equipos


# ---------------------------------------------------------------
# Construcción de Dataset
# ---------------------------------------------------------------
def construir_datos() -> pd.DataFrame:
    programa = calendario_temporada()
    finales = programa[programa["status"] == "Final"].copy()
    finales = finales.dropna(subset=["hom_score", "away_score"])

    equipos = cargar_equipos(finales)

    filas = []
    for _, g in finales.iterrows():
        gdate = pd.Timestamp(g["game_date"])
        hid, aid = int(g["hom_team_id"]), int(g["away_team_id"])
        pf = PARK_FACTORS.get(g["hom_team_name"], 1.00)

        feat = {"game_date": gdate, "park_factor": pf}

        for lado, tid in [("hom", hid), ("away", aid)]:
            eq = equipos.get(tid)
            if not eq:
                continue
            for k, v in ultimas_características(eq["features"], gdate).items():
                if k in ("date", "team_id", "opponent_id"):
                    continue
                feat[f"{lado}_{k}"] = v
            feat[f"{lado}_descanso"] = dias_descanso(eq["hitting"], gdate)

        feat["hom_runs"] = a_float(g["hom_score"])
        feat["away_runs"] = a_float(g["away_score"])
        feat["total_runs"] = feat["hom_runs"] + feat["away_runs"]
        feat["hom_win"] = 1 if feat["hom_runs"] > feat["away_runs"] else 0

        filas.append(feat)

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    cols_feat = [c for c in df.columns if c not in TARGETS and c != "game_date"]
    return df.dropna(subset=cols_feat, how="all")


# ---------------------------------------------------------------
# Entrenamiento de Modelos
# ---------------------------------------------------------------
def entrenar_modelos():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = construir_datos()

    if len(df) < 30:
        print("  ⚠️ Insuficientes datos para entrenar. Se requiere más historial.")
        return

    df = df.sort_values("game_date").reset_index(drop=True)
    cols_features = [c for c in df.columns if c not in TARGETS and c != "game_date"]
    X = df[cols_features].fillna(0.0)
    corte = int(len(df) * 0.8)

    for objetivo in TARGETS:
        if objetivo not in df.columns:
            continue
        es_clasificacion = objetivo == "hom_win"
        y = df[objetivo].astype(int) if es_clasificacion else df[objetivo].astype(float)

        if es_clasificacion:
            modelo = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=-1)
        else:
            modelo = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)

        modelo.fit(X.iloc[:corte], y.iloc[:corte])
        joblib.dump({"modelo": modelo, "features": cols_features}, os.path.join(MODEL_DIR, f"{objetivo}.joblib"))


# ---------------------------------------------------------------
# Generación de Predicciones y Cálculo EV+
# ---------------------------------------------------------------
def predecir_dia(fecha: str | None = None) -> pd.DataFrame:
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")
    juegos = calendario_dia(fecha)

    if juegos.empty:
        return pd.DataFrame()

    proximos = juegos[juegos["status"].isin(["Scheduled", "Preview", "Pre-Game"])].copy()
    if proximos.empty:
        return pd.DataFrame()

    equipos = cargar_equipos(proximos)
    odds_data = obtener_cuotas_odds_api(fecha)

    modelos = {}
    for objetivo in TARGETS:
        ruta = os.path.join(MODEL_DIR, f"{objetivo}.joblib")
        if os.path.exists(ruta):
            modelos[objetivo] = joblib.load(ruta)

    filas = []
    for _, g in proximos.iterrows():
        gdate = pd.Timestamp(g["game_date"])
        hid, aid = int(g["hom_team_id"]), int(g["away_team_id"])
        pf = PARK_FACTORS.get(g["hom_team_name"], 1.00)

        feat = {"park_factor": pf}
        for lado, tid in [("hom", hid), ("away", aid)]:
            eq = equipos.get(tid)
            if not eq:
                continue
            for k, v in ultimas_características(eq["features"], gdate).items():
                if k in ("date", "team_id", "opponent_id"):
                    continue
                feat[f"{lado}_{k}"] = v
            feat[f"{lado}_descanso"] = dias_descanso(eq["hitting"], gdate)

        pred = {
            "fecha": fecha,
            "matchup": f"{g['away_team_name']} @ {g['hom_team_name']}",
            "hom_team": g["hom_team_name"],
            "away_team": g["away_team_name"],
            "hom_pitcher": g.get("hom_pitcher", "TBD"),
            "away_pitcher": g.get("away_pitcher", "TBD"),
            "park_factor": pf,
        }

        for objetivo, paquete in modelos.items():
            X_row = pd.DataFrame([feat]).reindex(columns=paquete["features"], fill_value=0.0)
            modelo = paquete["modelo"]
            if objetivo == "hom_win":
                proba = modelo.predict_proba(X_row)[0]
                prob = proba[list(modelo.classes_).index(1)] if 1 in modelo.classes_ else 0.5
                pred["ml_local_%"] = round(prob * 100, 1)
                pred["ml_visitante_%"] = round((1 - prob) * 100, 1)
            else:
                val = float(modelo.predict(X_row)[0])
                # Aplicación directa del Park Factor en métricas de anotación y bateo
                if "runs" in objetivo or "hits" in objetivo or "total_bases" in objetivo:
                    val = val * pf
                pred[objetivo] = round(val, 2)

        # Integración de Cuotas y Expected Value (EV%)
        match_key = pred["matchup"]
        cuota_info = odds_data.get(match_key, {})
        h_odds = cuota_info.get("hom_ml_odds", 0.0)
        a_odds = cuota_info.get("away_ml_odds", 0.0)

        pred["cuota_local"] = h_odds if h_odds > 0 else "—"
        pred["cuota_visitante"] = a_odds if a_odds > 0 else "—"

        # Cálculo de EV% = (Prob * Cuota) - 1
        prob_h = pred.get("ml_local_%", 50.0) / 100.0
        prob_a = pred.get("ml_visitante_%", 50.0) / 100.0

        ev_h = (prob_h * h_odds - 1) * 100 if h_odds > 1.0 else -999.0
        ev_a = (prob_a * a_odds - 1) * 100 if a_odds > 1.0 else -999.0

        if ev_h > 3.0:
            pred["pick_ev"] = f"{g['hom_team_name']} ML (EV +{ev_h:.1f}%)"
            pred["stake_rec"] = "1.5% Bankroll"
        elif ev_a > 3.0:
            pred["pick_ev"] = f"{g['away_team_name']} ML (EV +{ev_a:.1f}%)"
            pred["stake_rec"] = "1.5% Bankroll"
        else:
            pred["pick_ev"] = "Sin Valor Claro"
            pred["stake_rec"] = "0%"

        filas.append(pred)

    return pd.DataFrame(filas)


# ---------------------------------------------------------------
# Reportes Markdown
# ---------------------------------------------------------------
def guardar_reporte(pred: pd.DataFrame, fecha: str | None = None) -> tuple[str, str]:
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_ruta = os.path.join(OUTPUT_DIR, f"predicciones_{fecha}.csv")
    pred.to_csv(csv_ruta, index=False)

    lineas = [
        f"# ⚾ Predicciones MLB EV+ — {fecha}",
        "",
        "> Análisis probabilístico combinando Random Forest, Park Factors, Abridores y Cuotas en vivo.",
        "",
        "| Partido | Abridores (V/L) | Prob Local | Cuota L/V | Total Runs | Pick EV+ | Stake |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in pred.iterrows():
        lineas.append(
            f"| {r['matchup']} "
            f"| {r['away_pitcher']} vs {r['hom_pitcher']} "
            f"| {r.get('ml_local_%', '—')}% "
            f"| {r.get('cuota_local', '—')} / {r.get('cuota_visitante', '—')} "
            f"| {r.get('total_runs', '—')} "
            f"| **{r.get('pick_ev', '—')}** "
            f"| {r.get('stake_rec', '—')} |"
        )

    md_ruta = os.path.join(OUTPUT_DIR, f"predicciones_{fecha}.md")
    with open(md_ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return md_ruta, csv_ruta


def main():
    fecha = sys.argv[1] if len(sys.argv) > 1 else None
    entrenar_modelos()
    pred = predecir_dia(fecha)
    if not pred.empty:
        md, csv = guardar_reporte(pred, fecha)
        print(f"✅ Generado: {md} | {csv}")


if __name__ == "__main__":
    main()

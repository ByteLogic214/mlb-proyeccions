"""
MLB Betting ML System - Pro Edition
===================================
Sistema de Machine Learning para proyección de mercados de apuestas MLB.

Novedades:
  - Integración con The Odds API (the-odds-api.com) para cuotas reales en vivo y cálculo de Expected Value (EV%).
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
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

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
PARK_FACTORS = {
    "Colorado Rockies": 1.35,
    "Boston Red Sox": 1.08,
    "Cincinnati Reds": 1.12,
    "Philadelphia Phillies": 1.06,
    "Kansas City Royals": 1.05,
    "Chicago Cubs": 1.04,
    "Baltimore Orioles": 1.02,
    "Texas Rangers": 1.02,
    "Los Angeles Dodgers": 1.01,
    "Atlanta Braves": 1.01,
    "Minnesota Twins": 1.00,
    "Chicago White Sox": 1.00,
    "St. Louis Cardinals": 0.99,
    "Milwaukee Brewers": 0.99,
    "New York Yankees": 0.98,
    "Houston Astros": 0.98,
    "Toronto Blue Jays": 0.98,
    "Arizona Diamondbacks": 0.97,
    "Washington Nationals": 0.97,
    "Los Angeles Angels": 0.96,
    "San Francisco Giants": 0.95,
    "Detroit Tigers": 0.95,
    "Pittsburgh Pirates": 0.95,
    "New York Mets": 0.94,
    "Cleveland Guardians": 0.94,
    "Tampa Bay Rays": 0.93,
    "Miami Marlins": 0.93,
    "San Diego Padres": 0.92,
    "Seattle Mariners": 0.91,
    "Athletics": 0.96,
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
        {"sportId": 1, "date": fecha, "hydrate": "probablePitcher"},
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
    """
    Consulta The Odds API (the-odds-api.com) para cruzar cuotas en vivo.

    Endpoint:
        GET https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/
        ?apiKey=YOUR_KEY&regions=us&markets=h2h,totals&date=YYYY-MM-DD
    """
    if not ODDS_API_KEY:
        print("  ⚠️ ODDS_API_KEY no configurada. Se omitirá el cálculo de EV%.")
        return {}

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "date": fecha,
    }

    try:
        r = requests.get(ODDS_API_URL, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  ⚠️ The Odds API respondió HTTP {r.status_code}")
            return {}

        data = r.json()

        cuotas = {}

        if not isinstance(data, list):
            return cuotas

        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")

            if not home or not away:
                continue

            key = f"{away} @ {home}"
            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            # Tomar el primer bookmaker disponible
            bm = bookmakers[0]
            markets_list = bm.get("markets", [])

            # Indexar mercados por key
            markets = {m["key"]: m for m in markets_list}

            # --- H2H (Moneyline) ---
            h2h = markets.get("h2h", {})
            h2h_outcomes = {
                o["name"]: o.get("price", 0.0)
                for o in h2h.get("outcomes", [])
            }
            h_odds = h2h_outcomes.get(home, 0.0)
            a_odds = h2h_outcomes.get(away, 0.0)

            # --- Totals ---
            totals = markets.get("totals", {})
            total_line = 0.0
            for outcome in totals.get("outcomes", []):
                if "point" in outcome:
                    total_line = float(outcome["point"])
                    break

            cuotas[key] = {
                "hom_ml_odds": float(h_odds),
                "away_ml_odds": float(a_odds),
                "total_line": float(total_line),
            }

        print(f"  ✅ Cuotas obtenidas para {len(cuotas)} partidos (The Odds API)")
        return cuotas

    except Exception as e:
        print(f"  ⚠️ Error consultando The Odds API: {e}")
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
            return float(entero) + {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(resto[:1], 0.0)
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
def american_to_decimal(american_odds: float) -> float:
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    elif american_odds < 0:
        return (100.0 / abs(american_odds)) + 1.0
    return 1.0


def predecir_dia(fecha: str | None = None) -> pd.DataFrame:
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")
    juegos = calendario_dia(fecha)

    if juegos.empty:
        return pd.DataFrame()

    proximos = juegos[juegos["status"].isin(["Scheduled", "Pre-Game", "Warmup", "In-Progress", "Live"])]
    if proximos.empty:
        return pd.DataFrame()

    equipos = cargar_equipos(proximos)
    cuotas_api = obtener_cuotas_odds_api(fecha)

    predicciones = []
    gdate = pd.Timestamp(fecha)

    for _, g in proximos.iterrows():
        hid, aid = g["hom_team_id"], g["away_team_id"]
        if pd.isna(hid) or pd.isna(aid):
            continue

        hid, aid = int(hid), int(aid)
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

        row_pred = {
            "fecha": fecha,
            "matchup": f"{g['away_team_name']} @ {g['hom_team_name']}",
            "hom_team": g["hom_team_name"],
            "away_team": g["away_team_name"],
            "hom_pitcher": g["hom_pitcher"],
            "away_pitcher": g["away_pitcher"],
            "park_factor": pf,
        }

        # Inferencia con modelos entrenados
        for objetivo in TARGETS:
            ruta_m = os.path.join(MODEL_DIR, f"{objetivo}.joblib")
            if not os.path.exists(ruta_m):
                continue
            obj_loaded = joblib.load(ruta_m)
            modelo = obj_loaded["modelo"]
            cols = obj_loaded["features"]

            X_in = pd.DataFrame([feat]).reindex(columns=cols, fill_value=0.0)

            if objetivo == "hom_win":
                probs = modelo.predict_proba(X_in)[0]
                row_pred["ml_local_%"] = round(probs[1] * 100, 1)
                row_pred["ml_visitante_%"] = round(probs[0] * 100, 1)
            else:
                val = modelo.predict(X_in)[0]
                row_pred[objetivo] = round(float(val), 2)

        # Integración con Odds de mercado
        match_key = f"{g['away_team_name']} @ {g['hom_team_name']}"
        odds_info = cuotas_api.get(match_key, {})

        cuota_h_am = odds_info.get("hom_ml_odds", 0.0)
        cuota_a_am = odds_info.get("away_ml_odds", 0.0)

        cuota_h_dec = american_to_decimal(cuota_h_am) if cuota_h_am != 0.0 else 0.0
        cuota_a_dec = american_to_decimal(cuota_a_am) if cuota_a_am != 0.0 else 0.0

        row_pred["cuota_local"] = cuota_h_dec if cuota_h_dec > 0 else "—"
        row_pred["cuota_visitante"] = cuota_a_dec if cuota_a_dec > 0 else "—"

        # Lógica de cálculo EV+ y Stake Kelly
        p_local = row_pred.get("ml_local_%", 50.0) / 100.0
        p_away = row_pred.get("ml_visitante_%", 50.0) / 100.0

        ev_h = (p_local * cuota_h_dec) - 1.0 if cuota_h_dec > 1.0 else -1.0
        ev_a = (p_away * cuota_a_dec) - 1.0 if cuota_a_dec > 1.0 else -1.0

        if ev_h > 0.03 and ev_h > ev_a:
            b = cuota_h_dec - 1.0
            kelly = max(0.0, ((b * p_local) - (1 - p_local)) / b) * 0.25 # Fraccional Kelly (1/4)
            row_pred["pick_ev"] = f"Local (+EV {ev_h:.1%})"
            row_pred["stake_rec"] = f"{kelly:.1%}"
        elif ev_a > 0.03 and ev_a > ev_h:
            b = cuota_a_dec - 1.0
            kelly = max(0.0, ((b * p_away) - (1 - p_away)) / b) * 0.25
            row_pred["pick_ev"] = f"Visitante (+EV {ev_a:.1%})"
            row_pred["stake_rec"] = f"{kelly:.1%}"
        else:
            row_pred["pick_ev"] = "Sin Valor Claro"
            row_pred["stake_rec"] = "0%"

        predicciones.append(row_pred)

    return pd.DataFrame(predicciones)


# ---------------------------------------------------------------
# Punto de Entrada Principal
# ---------------------------------------------------------------
def main():
    fecha_target = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")

    print(f"🚀 Iniciando Pipeline MLB ML Pro (Fecha: {fecha_target})...")

    print("1️⃣ Entrenando / Actualizando modelos cuantitativos...")
    entrenar_modelos()

    print("2️⃣ Generando predicciones y buscando cuotas de valor (+EV)...")
    df_res = predecir_dia(fecha_target)

    if df_res.empty:
        print("⚠️ No se encontraron partidos agendados o activos para procesar.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_out = os.path.join(OUTPUT_DIR, f"predicciones_{fecha_target}.csv")
    df_res.to_csv(file_out, index=False)

    print(f"✅ Proceso finalizado con éxito. Resultado guardado en: {file_out}")


if __name__ == "__main__":
    main()

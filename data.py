"""
Capa de datos
=============
Cliente MLB Stats API con cache en disco, normalización de game logs e
ingeniería de features (medias móviles 3/5/10, temporada, tendencia,
descanso, factor de parque, targets desde logs reales).
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import config

HITTING = ["runs", "hits", "doubles", "triples", "homeRuns", "totalBases",
           "strikeOuts", "baseOnBalls", "walk", "atBats"]

PITCHING = ["runs", "hits", "strikeOuts", "baseOnBalls", "homeRuns",
            "inningsPitched", "era", "whip"]

# Factores de Parque MLB (baseline 1.00 = neutral)
PARK_FACTORS: dict[str, float] = {
    "Colorado Rockies": 1.35, "Boston Red Sox": 1.08, "Cincinnati Reds": 1.12,
    "Philadelphia Phillies": 1.06, "Kansas City Royals": 1.05,
    "Chicago Cubs": 1.04, "Baltimore Orioles": 1.02, "Texas Rangers": 1.02,
    "Los Angeles Dodgers": 1.01, "Atlanta Braves": 1.01,
    "Minnesota Twins": 1.00, "Chicago White Sox": 1.00,
    "St. Louis Cardinals": 0.99, "Milwaukee Brewers": 0.99,
    "New York Yankees": 0.98, "Houston Astros": 0.98,
    "Toronto Blue Jays": 0.98, "Arizona Diamondbacks": 0.97,
    "Washington Nationals": 0.97, "Los Angeles Angels": 0.96,
    "San Francisco Giants": 0.95, "Detroit Tigers": 0.95,
    "Pittsburgh Pirates": 0.95, "New York Mets": 0.94,
    "Cleveland Guardians": 0.94, "Tampa Bay Rays": 0.93,
    "Miami Marlins": 0.93, "San Diego Padres": 0.92,
    "Seattle Mariners": 0.91, "Athletics": 0.96,
}


def park_factor(team_name: str) -> float:
    return PARK_FACTORS.get(team_name, 1.00)


# ---------------------------------------------------------------
# HTTP con reintentos y cache en disco
# ---------------------------------------------------------------
def _get(url: str, params: dict[str, Any] | None = None) -> dict:
    for intento in range(config.api.max_retries):
        try:
            r = requests.get(url, params=params, timeout=config.api.timeout)
            if r.status_code == 429:
                time.sleep(config.api.retry_delay * (intento + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if intento == config.api.max_retries - 1:
                return {}
            time.sleep(config.api.retry_delay * (intento + 1))
    return {}


def _leer_cache(nombre: str) -> pd.DataFrame | None:
    ruta = os.path.join(config.data.data_dir, nombre)
    if os.path.exists(ruta) and (time.time() - os.path.getmtime(ruta)) < config.data.cache_ttl:
        try:
            return pd.read_csv(ruta)
        except Exception:
            return None
    return None


def _guardar_cache(nombre: str, df: pd.DataFrame) -> None:
    os.makedirs(config.data.data_dir, exist_ok=True)
    df.to_csv(os.path.join(config.data.data_dir, nombre), index=False)


# ---------------------------------------------------------------
# API MLB
# ---------------------------------------------------------------
def _parsear_calendario(payload: dict) -> pd.DataFrame:
    juegos = []
    for dia in payload.get("dates", []):
        for g in dia.get("games", []):
            local = g.get("teams", {}).get("home", {})
            visit = g.get("teams", {}).get("away", {})
            juegos.append({
                "game_pk": g.get("gamePk"),
                "game_date": (g.get("gameDate") or "")[:10],
                "status": g.get("status", {}).get("abstractGameState"),
                "hom_team_id": local.get("team", {}).get("id"),
                "away_team_id": visit.get("team", {}).get("id"),
                "hom_team_name": local.get("team", {}).get("name"),
                "away_team_name": visit.get("team", {}).get("name"),
                "hom_pitcher": local.get("probablePitcher", {}).get("fullName", "TBD"),
                "away_pitcher": visit.get("probablePitcher", {}).get("fullName", "TBD"),
                "hom_pitcher_id": local.get("probablePitcher", {}).get("id", None),
                "away_pitcher_id": visit.get("probablePitcher", {}).get("id", None),
                "hom_score": local.get("score"),
                "away_score": visit.get("score"),
            })
    return pd.DataFrame(juegos)


def calendario_temporada() -> pd.DataFrame:
    cache = f"calendario_{config.data.season}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{config.api.mlb_base_url}/schedule",
        {"sportId": 1, "season": config.data.season, "gameType": "R",
         "hydrate": "probablePitcher"},
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
        f"{config.api.mlb_base_url}/schedule",
        {"sportId": 1, "date": fecha, "hydrate": "probablePitcher"},
    )
    df = _parsear_calendario(payload)
    if not df.empty:
        _guardar_cache(cache, df)
    return df


def game_log_equipo(team_id: int, grupo: str) -> pd.DataFrame:
    cache = f"equipo_{team_id}_{config.data.season}_{grupo}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{config.api.mlb_base_url}/teams/{team_id}/stats",
        {"season": config.data.season, "group": grupo, "stats": "gameLog"},
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


# ---------------------------------------------------------------
# Normalización y features
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
        for w in config.model.windows:
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


def stat_del_juego(log: pd.DataFrame, fecha: pd.Timestamp, col: str) -> float:
    """Valor real de `col` en el juego de `fecha` según el game log (para targets)."""
    if log is None or log.empty or col not in log.columns:
        return np.nan
    fila = log[log["date"] == fecha]
    if fila.empty:
        return np.nan
    return a_float(fila.iloc[-1][col])


def dias_descanso(df: pd.DataFrame, fecha: pd.Timestamp) -> float:
    if df is None or df.empty:
        return 0.0
    previos = df[df["date"] < fecha]
    if previos.empty:
        return 0.0
    return max(0.0, (fecha - previos["date"].max()).days)


def cargar_equipo(team_id: int) -> dict:
    hitting = normalizar_log(game_log_equipo(team_id, "hitting"))
    pitching = normalizar_log(game_log_equipo(team_id, "pitching"))
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

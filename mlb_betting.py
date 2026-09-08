"""
MLB Betting ML System
=====================
Sistema de Machine Learning para proyección de mercados de apuestas de la MLB
usando la MLB Stats API gratuita (sin API key).

Predice:
  - Moneyline (probabilidad de victoria del equipo local)
  - Total de runs del juego y runs por equipo
  - Hits por equipo
  - Bases totales por equipo
  - Ponches de bateadores y lanzadores

Uso:
    python mlb_betting.py [AAAA-MM-DD]
    (Si no se indica fecha, usa la fecha actual en UTC.)
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
# Configuración
# ---------------------------------------------------------------
SEASON = 2026
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "output"
WINDOWS = [3, 5, 10]

# ⚠️ CORREGIDO: nombres exactos que usa la MLB Stats API
HITTING = ["runs", "hits", "doubles", "triples", "homeRuns", "totalBases",
           "strikeOuts", "baseOnBalls", "walk", "atBats"]

PITCHING = ["runs", "hits", "strikeOuts", "baseOnBalls", "homeRuns",
            "inningsPitched", "era", "whip"]

# Diferencia clave: la API usa "baseOnBalls" no "baseOnBals"
# (verificado en https://statsapi.mlb.com/api/v1/teams/108/stats?season=2025&group=hitting&stats=gameLog)

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

# ---------------------------------------------------------------
# Cliente MLB Stats API
# ---------------------------------------------------------------
def _get(url: str, params: dict[str, Any] | None = None) -> dict:
    for intento in range(4):
        try:
            r = requests.get(url, params=params, timeout=45)  # timeout 45s
            if r.status_code == 429:  # rate limit
                time.sleep(10 * (intento + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if intento == 3:
                raise RuntimeError(f"No se pudo consultar {url}")
            time.sleep(10 * (intento + 1))
    raise RuntimeError(f"No se pudo consultar {url}")


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
            juegos.append({
                "game_pk": g.get("gamePk"),
                "game_date": (g.get("gameDate") or "")[:10],  # fecha UTC
                "status": g.get("status", {}).get("abstractGameState"),
                "hom_team_id": local.get("team", {}).get("id"),
                "away_team_id": visit.get("team", {}).get("id"),
                "hom_team_name": local.get("team", {}).get("name"),
                "away_team_name": visit.get("team", {}).get("name"),
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
        {"sportId": 1, "season": SEASON, "gameType": "R"},
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
    payload = _get(f"{API_BASE}/schedule", {"sportId": 1, "date": fecha})
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


# ---------------------------------------------------------------
# Utilidades numéricas
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
    """Convierte '7.2' (7 2/3 entradas) a 7.666"""
    if isinstance(x, str) and "." in x:
        entero, _, resto = x.partition(".")
        try:
            return float(entero) + {"0": 0.0, "1": 1/3, "2": 2/3}.get(resto[:1], 0.0)
        except ValueError:
            pass
    return a_float(x)


# ---------------------------------------------------------------
# Ingeniería de características
# ---------------------------------------------------------------
def normalizar_log(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["date", "team_id", "opponent_id"])
    df = df.sort_values("date").reset_index(drop=True)
    df["team_id"] = df["team_id"].astype(int)
    df["opponent_id"] = df["opponent_id"].astype(int)

    # Convertir todos los campos numéricos conocidos
    for c in HITTING + PITCHING:
        if c in df.columns:
            df[c] = df[c].apply(a_float)

    # Métricas derivadas
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


def _media_movil(df: pd.DataFrame, col: str, ventana: int) -> pd.Series:
    return df[col].shift(1).rolling(ventana, min_periods=1).mean()


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
            feat[f"{prefijo}{c}_avg{w}"] = _media_movil(df, c, w).values
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


def objetivo_de_log(df: pd.DataFrame, fecha: pd.Timestamp, rival_id: int) -> dict:
    if df is None or df.empty:
        return {}
    partido = df[(df["date"] == fecha) & (df["opponent_id"] == rival_id)]
    if partido.empty:
        partido = df[df["date"] == fecha]
    if partido.empty:
        return {}
    return partido.iloc[0].to_dict()


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
        except Exception as e:
            print(f"  ⚠️ No se pudo cargar el equipo {int(tid)}: {e}")
    return equipos


# ---------------------------------------------------------------
# Dataset de entrenamiento
# ---------------------------------------------------------------
def construir_datos() -> pd.DataFrame:
    programa = calendario_temporada()
    finales = programa[programa["status"] == "Final"].copy()
    finales = finales.dropna(subset=["hom_score", "away_score"])
    print(f"  → Juegos finalizados: {len(finales)}")

    equipos = cargar_equipos(finales)
    print(f"  → Equipos cargados: {len(equipos)}")

    filas = []
    for _, g in finales.iterrows():
        gdate = pd.Timestamp(g["game_date"])
        hid, aid = int(g["hom_team_id"]), int(g["away_team_id"])

        feat = {"game_date": gdate}

        for lado, tid in [("hom", hid), ("away", aid)]:
            eq = equipos.get(tid)
            if not eq:
                continue
            for k, v in ultimas_características(eq["features"], gdate).items():
                if k in ("date", "team_id", "opponent_id"):
                    continue
                feat[f"{lado}_{k}"] = v
            feat[f"{lado}_descanso"] = dias_descanso(eq["hitting"], gdate)

        # --- Objetivos ---
        feat["hom_runs"] = a_float(g["hom_score"])
        feat["away_runs"] = a_float(g["away_score"])
        feat["total_runs"] = feat["hom_runs"] + feat["away_runs"]
        feat["hom_win"] = 1 if feat["hom_runs"] > feat["away_runs"] else 0

        ht = objetivo_de_log(equipos[hid]["hitting"], gdate, aid) if hid in equipos else {}
        at = objetivo_de_log(equipos[aid]["hitting"], gdate, hid) if aid in equipos else {}
        pt = objetivo_de_log(equipos[hid]["pitching"], gdate, aid) if hid in equipos else {}
        at2 = objetivo_de_log(equipos[aid]["pitching"], gdate, hid) if aid in equipos else {}

        feat["hom_hits"] = a_float(ht.get("hits"))
        feat["away_hits"] = a_float(at.get("hits"))
        feat["hom_total_bases"] = a_float(ht.get("totalBases"))
        feat["away_total_bases"] = a_float(at.get("totalBases"))
        feat["hom_k"] = a_float(ht.get("strikeOuts"))
        feat["away_k"] = a_float(at.get("strikeOuts"))
        feat["hom_p_k"] = a_float(pt.get("strikeOuts"))
        feat["away_p_k"] = a_float(at2.get("strikeOuts"))
        feat["total_k"] = feat["hom_p_k"] + feat["away_p_k"]

        filas.append(feat)

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    cols_feat = [c for c in df.columns if c not in TARGETS and c != "game_date"]
    df = df.dropna(subset=cols_feat, how="all")
    return df


# ---------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------
def entrenar_modelos():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("  → Construyendo dataset de entrenamiento...")
    t0 = time.time()
    df = construir_datos()
    print(f"  → {len(df)} juegos con características ({time.time() - t0:.1f}s)")

    if len(df) < 50:
        raise RuntimeError(
            f"Dataset muy pequeño ({len(df)} filas). "
            "La temporada no tiene suficientes datos."
        )

    df = df.sort_values("game_date").reset_index(drop=True)
    cols_features = [c for c in df.columns if c not in TARGETS and c != "game_date"]
    X = df[cols_features].fillna(0.0)
    print(f"  → {len(cols_features)} características por juego")

    corte = int(len(df) * 0.8)

    for objetivo in TARGETS:
        if objetivo not in df.columns:
            continue
        es_clasificación = objetivo == "hom_win"
        y = df[objetivo].astype(int) if es_clasificación else df[objetivo].astype(float)

        if es_clasificación:
            modelo = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
        else:
            modelo = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=4,
                random_state=42, n_jobs=-1,
            )

        modelo.fit(X.iloc[:corte], y.iloc[:corte])
        pred = modelo.predict(X.iloc[corte:])
        y_test = y.iloc[corte:]

        if es_clasificación:
            score = accuracy_score(y_test, pred)
            print(f"  ✅ {objetivo}: exactitud = {score:.3f}")
        else:
            r2 = r2_score(y_test, pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            print(f"  ✅ {objetivo}: R² = {r2:.3f} | RMSE = {rmse:.3f}")

        joblib.dump(
            {"modelo": modelo, "features": cols_features},
            os.path.join(MODEL_DIR, f"{objetivo}.joblib"),
        )


# ---------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------
def predecir_dia(fecha: str | None = None) -> pd.DataFrame:
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")
    print(f"  → Fecha: {fecha}")

    juegos = calendario_dia(fecha)
    if juegos.empty:
        print("  → La API no devuelve juegos para esta fecha.")
        return pd.DataFrame()

    proximos = juegos[juegos["status"].isin(["Scheduled", "Preview"])].copy()
    print(f"  → Juegos programados: {len(proximos)}")
    if proximos.empty:
        return pd.DataFrame()

    equipos = cargar_equipos(proximos)

    modelos = {}
    for objetivo in TARGETS:
        ruta = os.path.join(MODEL_DIR, f"{objetivo}.joblib")
        if os.path.exists(ruta):
            modelos[objetivo] = joblib.load(ruta)
    if not modelos:
        raise RuntimeError("No hay modelos entrenados. Ejecuta primero entrenar_modelos().")

    filas = []
    for _, g in proximos.iterrows():
        gdate = pd.Timestamp(g["game_date"])
        hid, aid = int(g["hom_team_id"]), int(g["away_team_id"])

        feat = {}
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
            "hom_team": g["hom_team_name"],
            "away_team": g["away_team_name"],
        }

        for objetivo, paquete in modelos.items():
            X_row = pd.DataFrame([feat]).reindex(
                columns=paquete["features"], fill_value=0.0
            )
            modelo = paquete["modelo"]
            if objetivo == "hom_win":
                proba = modelo.predict_proba(X_row)[0]
                prob = proba[list(modelo.classes_).index(1)] if 1 in modelo.classes_ else 0.0
                pred["ml_local_%"] = round(prob * 100, 1)
            else:
                pred[objetivo] = round(float(modelo.predict(X_row)[0]), 2)

        filas.append(pred)

    return pd.DataFrame(filas)


# ---------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------
def guardar_reporte(pred: pd.DataFrame, fecha: str | None = None) -> tuple[str, str]:
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_ruta = os.path.join(OUTPUT_DIR, f"predicciones_{fecha}.csv")
    pred.to_csv(csv_ruta, index=False)

    lineas = [
        f"# Predicciones MLB — {fecha}",
        "",
        "> Compara estas proyecciones con las líneas de tu casa de apuestas.",
        "",
    ]
    if pred.empty:
        lineas.append("No hay juegos programados para esta fecha.")
    else:
        lineas.append(
            "| Juego | ML Local | Total Runs | Runs L/V | Hits L/V | "
            "Bases L/V | K Bateador L/V | K Lanzador L/V |"
        )
        lineas.append("|---|---|---|---|---|---|---|---|")
        for _, r in pred.iterrows():
            lineas.append(
                f"| {r['away_team']} @ {r['hom_team']} "
                f"| {r.get('ml_local_%', '—')}% "
                f"| {r.get('total_runs', '—')} "
                f"| {r.get('hom_runs', '—')}/{r.get('away_runs', '—')} "
                f"| {r.get('hom_hits', '—')}/{r.get('away_hits', '—')} "
                f"| {r.get('hom_total_bases', '—')}/{r.get('away_total_bases', '—')} "
                f"| {r.get('hom_k', '—')}/{r.get('away_k', '—')} "
                f"| {r.get('hom_p_k', '—')}/{r.get('away_p_k', '—')} |"
            )

    md_ruta = os.path.join(OUTPUT_DIR, f"predicciones_{fecha}.md")
    with open(md_ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return md_ruta, csv_ruta


# ---------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print(" MLB Betting ML System — MLB Stats API (gratuita)")
    print(f" Temporada {SEASON}")
    print("=" * 60)

    fecha = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        print("\n[1/2] Entrenando modelos...")
        entrenar_modelos()

        print("\n[2/2] Generando predicciones...")
        pred = predecir_dia(fecha)
        if pred.empty:
            print("  → Sin juegos para predecir.")
            return

        md, csv = guardar_reporte(pred, fecha)
        print(f"\n  📄 Reporte: {md}")
        print(f"  📊 CSV: {csv}")

        print("\nResumen:")
        for _, r in pred.iterrows():
            ml = r.get("ml_local_%", "?")
            total = r.get("total_runs", "?")
            print(f"  • {r['away_team']} @ {r['hom_team']}: "
                  f"ML local {ml}% | total {total} runs")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

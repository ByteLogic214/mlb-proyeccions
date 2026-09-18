"""
Capa de modelos
===============
Dataset, entrenamiento con evaluación temporal (holdout del 20% más
reciente), calibración de probabilidades (CalibratedClassifierCV, adoptada
solo si no empeora el log-loss), re-fit final sobre el 100% sin fuga, y
predicción diaria con EV por mercado.
"""

from __future__ import annotations

import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error, r2_score,
)

try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRACION_DISPONIBLE = True
except ImportError:  # pragma: no cover
    CALIBRACION_DISPONIBLE = False

from . import data
from .config import config
from .odds import obtener_cuotas

TARGETS = [
    "hom_runs", "away_runs", "total_runs",
    "hom_hits", "away_hits",
    "hom_total_bases", "away_total_bases",
    "hom_k", "away_k",
    "hom_p_k", "away_p_k",
    "total_k",
    "hom_win",
]

TARGETS_FROM_LOGS = [
    "hom_hits", "away_hits", "hom_total_bases", "away_total_bases",
    "hom_k", "away_k", "hom_p_k", "away_p_k", "total_k",
]

ESTADOS_PENDIENTES = ["Preview", "Scheduled", "Pre-Game", "Warmup",
                      "In-Progress", "Live"]


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
def construir_datos() -> pd.DataFrame:
    programa = data.calendario_temporada()
    finales = programa[programa["status"] == "Final"].copy()
    finales = finales.dropna(subset=["hom_score", "away_score"])
    equipos = data.cargar_equipos(finales)

    filas = []
    for _, g in finales.iterrows():
        gdate = pd.Timestamp(g["game_date"])
        hid, aid = int(g["hom_team_id"]), int(g["away_team_id"])
        feat = {"game_date": gdate, "park_factor": data.park_factor(g["hom_team_name"])}
        eq_h, eq_a = equipos.get(hid), equipos.get(aid)

        for lado, eq in [("hom", eq_h), ("away", eq_a)]:
            if not eq:
                continue
            for k, v in data.ultimas_características(eq["features"], gdate).items():
                if k in ("date", "team_id", "opponent_id"):
                    continue
                feat[f"{lado}_{k}"] = v
            feat[f"{lado}_descanso"] = data.dias_descanso(eq["hitting"], gdate)

        feat["hom_runs"] = data.a_float(g["hom_score"])
        feat["away_runs"] = data.a_float(g["away_score"])
        feat["total_runs"] = feat["hom_runs"] + feat["away_runs"]
        feat["hom_win"] = 1 if feat["hom_runs"] > feat["away_runs"] else 0

        if eq_h:
            feat["hom_hits"] = data.stat_del_juego(eq_h["hitting"], gdate, "hits")
            feat["hom_total_bases"] = data.stat_del_juego(eq_h["hitting"], gdate, "totalBases")
            feat["hom_k"] = data.stat_del_juego(eq_h["hitting"], gdate, "strikeOuts")
            feat["hom_p_k"] = data.stat_del_juego(eq_h["pitching"], gdate, "strikeOuts")
        if eq_a:
            feat["away_hits"] = data.stat_del_juego(eq_a["hitting"], gdate, "hits")
            feat["away_total_bases"] = data.stat_del_juego(eq_a["hitting"], gdate, "totalBases")
            feat["away_k"] = data.stat_del_juego(eq_a["hitting"], gdate, "strikeOuts")
            feat["away_p_k"] = data.stat_del_juego(eq_a["pitching"], gdate, "strikeOuts")
        if not pd.isna(feat.get("hom_p_k", np.nan)) and not pd.isna(feat.get("away_p_k", np.nan)):
            feat["total_k"] = feat["hom_p_k"] + feat["away_p_k"]

        filas.append(feat)

    df = pd.DataFrame(filas)
    if df.empty:
        return df
    cols_feat = [c for c in df.columns if c not in TARGETS and c != "game_date"]
    df = df.dropna(subset=cols_feat, how="all")
    presentes = [t for t in TARGETS_FROM_LOGS if t in df.columns]
    if presentes:
        antes = len(df)
        df = df.dropna(subset=presentes)
        if antes - len(df):
            print(f"  ℹ️ {antes - len(df)} juegos sin game log cruzable descartados.")
    return df


# ---------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------
def _crear_modelo_base(es_clasificacion: bool):
    mc = config.model
    if es_clasificacion:
        return RandomForestClassifier(
            n_estimators=mc.rf_n_estimators, max_depth=mc.rf_max_depth_classifier,
            min_samples_leaf=mc.rf_min_samples_leaf_classifier,
            random_state=mc.random_state, n_jobs=-1)
    return RandomForestRegressor(
        n_estimators=mc.rf_n_estimators, max_depth=mc.rf_max_depth_regressor,
        min_samples_leaf=mc.rf_min_samples_leaf_regressor,
        random_state=mc.random_state, n_jobs=-1)


def _evaluar_holdout(modelo, X_test, y_test, es_clasificacion: bool) -> dict:
    metricas: dict = {"n_test": int(len(X_test))}
    if es_clasificacion:
        metricas["accuracy"] = round(accuracy_score(y_test, modelo.predict(X_test)), 4)
        try:
            probas = modelo.predict_proba(X_test)
            if len(modelo.classes_) == 2:
                metricas["log_loss"] = round(log_loss(y_test, probas), 4)
                metricas["brier"] = round(brier_score_loss(y_test, probas[:, 1]), 4)
        except Exception:
            pass
    else:
        y_pred = modelo.predict(X_test)
        metricas["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metricas["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
        metricas["r2"] = round(r2_score(y_test, y_pred), 4)
    return metricas


def _top_features(modelo, cols, n=5) -> list:
    try:
        pares = sorted(zip(cols, modelo.feature_importances_), key=lambda p: p[1], reverse=True)
        return [f"{nom}={val:.3f}" for nom, val in pares[:n]]
    except Exception:
        return []


def entrenar_modelos() -> dict:
    """Entrena los 13 modelos; devuelve {target: metricas_test}."""
    os.makedirs(config.data.model_dir, exist_ok=True)
    df = construir_datos()
    if len(df) < config.model.min_training_samples:
        print("  ⚠️ Insuficientes datos para entrenar.")
        return {}

    df = df.sort_values("game_date").reset_index(drop=True)
    cols_features = [c for c in df.columns if c not in TARGETS and c != "game_date"]
    X = df[cols_features].fillna(0.0)
    corte = int(len(df) * (1 - config.model.test_size))
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]

    print(f"  📐 Dataset: {len(df)} juegos | train={len(X_train)} | holdout={len(X_test)} (temporal)")

    resumen: dict = {}
    for objetivo in TARGETS:
        if objetivo not in df.columns:
            print(f"  ⏭️  '{objetivo}' sin target — se omite.")
            continue
        es_clas = objetivo == "hom_win"
        y = df[objetivo].astype(int) if es_clas else df[objetivo].astype(float)
        y_train, y_test = y.iloc[:corte], y.iloc[corte:]
        if y_train.nunique() < 2 or (es_clas and y_test.nunique() < 2):
            print(f"  ⏭️  '{objetivo}' con una sola clase — se omite.")
            continue

        base = _crear_modelo_base(es_clas)
        try:
            base.fit(X_train, y_train)
            metricas = _evaluar_holdout(base, X_test, y_test, es_clas)
        except Exception as e:
            print(f"  ⚠️ Error evaluando '{objetivo}': {e}")
            metricas = {}

        calibrado = False
        modelo_final = base
        if es_clas and CALIBRACION_DISPONIBLE:
            try:
                modelo_cal = CalibratedClassifierCV(_crear_modelo_base(True), method="sigmoid", cv=3)
                modelo_cal.fit(X_train, y_train)
                m_cal = _evaluar_holdout(modelo_cal, X_test, y_test, True)
                if "log_loss" not in metricas or m_cal.get("log_loss", 9e9) <= metricas.get("log_loss", 9e9):
                    modelo_final, calibrado, metricas = modelo_cal, True, m_cal
            except Exception as e:
                print(f"  ⚠️ Calibración falló para '{objetivo}': {e}")

        modelo_final.fit(X, y)  # re-fit 100% tras evaluar (sin fuga)
        joblib.dump(
            {"modelo": modelo_final, "features": cols_features,
             "calibrado": calibrado, "metricas_test": metricas},
            os.path.join(config.data.model_dir, f"{objetivo}.joblib"),
        )
        resumen[objetivo] = {**metricas, "calibrado": calibrado, "n_train_full": int(len(X))}

        if es_clas:
            detalle = (f"acc={metricas.get('accuracy', '—')} "
                       f"logloss={metricas.get('log_loss', '—')} brier={metricas.get('brier', '—')}")
        else:
            detalle = (f"rmse={metricas.get('rmse', '—')} mae={metricas.get('mae', '—')} "
                       f"r2={metricas.get('r2', '—')}")
        print(f"  ✅ {objetivo:<16} {detalle} {'[calibrado]' if calibrado else ''}")
        top = _top_features(base, cols_features)
        if top:
            print(f"     ↳ top: {', '.join(top)}")

    if resumen:
        ruta = os.path.join(config.data.output_dir, "metricas_modelos.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"season": config.data.season,
                       "generado": datetime.datetime.now().isoformat(),
                       "n_juegos": int(len(df)), "modelos": resumen},
                      f, indent=2, ensure_ascii=False)
        print(f"  💾 Métricas persistidas en: {ruta}")
    return resumen


# ---------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------
def _inferir(row_feat: dict) -> dict:
    out: dict = {}
    for objetivo in TARGETS:
        ruta = os.path.join(config.data.model_dir, f"{objetivo}.joblib")
        if not os.path.exists(ruta):
            continue
        obj = joblib.load(ruta)
        X_in = pd.DataFrame([row_feat]).reindex(columns=obj["features"], fill_value=0.0)
        if objetivo == "hom_win":
            probs = obj["modelo"].predict_proba(X_in)[0]
            out["ml_local_%"] = round(probs[1] * 100, 1)
            out["ml_visitante_%"] = round(probs[0] * 100, 1)
        else:
            out[objetivo] = round(float(obj["modelo"].predict(X_in)[0]), 2)
    # Consistencia: totales = suma de individuales
    if "hom_runs" in out and "away_runs" in out:
        out["total_runs"] = round(out["hom_runs"] + out["away_runs"], 2)
    if "hom_p_k" in out and "away_p_k" in out:
        out["total_k"] = round(out["hom_p_k"] + out["away_p_k"], 2)
    return out


def predecir_dia(fecha: str | None = None) -> pd.DataFrame:
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")
    juegos = data.calendario_dia(fecha)
    if juegos.empty:
        return pd.DataFrame()
    proximos = juegos[juegos["status"].isin(ESTADOS_PENDIENTES)]
    if proximos.empty:
        return pd.DataFrame()

    equipos = data.cargar_equipos(proximos)
    cuotas = obtener_cuotas(fecha)
    gdate = pd.Timestamp(fecha)
    predicciones = []

    for _, g in proximos.iterrows():
        if pd.isna(g["hom_team_id"]) or pd.isna(g["away_team_id"]):
            continue
        hid, aid = int(g["hom_team_id"]), int(g["away_team_id"])
        feat = {"park_factor": data.park_factor(g["hom_team_name"])}

        for lado, tid in [("hom", hid), ("away", aid)]:
            eq = equipos.get(tid)
            if not eq:
                continue
            for k, v in data.ultimas_características(eq["features"], gdate).items():
                if k in ("date", "team_id", "opponent_id"):
                    continue
                feat[f"{lado}_{k}"] = v
            feat[f"{lado}_descanso"] = data.dias_descanso(eq["hitting"], gdate)

        row = {
            "fecha": fecha,
            "matchup": f"{g['away_team_name']} @ {g['hom_team_name']}",
            "hom_team": g["hom_team_name"],
            "away_team": g["away_team_name"],
            "hom_pitcher": g["hom_pitcher"],
            "away_pitcher": g["away_pitcher"],
            "park_factor": feat["park_factor"],
        }
        row.update(_inferir(feat))

        row["proj_carreras_local"] = row.get("hom_runs")
        row["proj_carreras_visitante"] = row.get("away_runs")
        row["proj_total_carreras"] = row.get("total_runs")

        info = cuotas.get(row["matchup"], {})
        c_h = info.get("hom_ml_odds", 0.0)
        c_a = info.get("away_ml_odds", 0.0)
        row["cuota_local"] = c_h if c_h > 1.0 else "—"
        row["cuota_visitante"] = c_a if c_a > 1.1 else "—"

        p_h = row.get("ml_local_%", 50.0) / 100.0
        p_a = row.get("ml_visitante_%", 50.0) / 100.0
        row["ev_local"] = round(p_h * c_h - 1.0, 4) if c_h > 1.0 else np.nan
        row["ev_visitante"] = round(p_a * c_a - 1.0, 4) if c_a > 1.0 else np.nan

        linea = info.get("total_line", 0.0)
        if linea > 0 and "total_runs" in row:
            diff = row["total_runs"] - linea
            row["linea_total_mercado"] = linea
            row["over_cuota"] = info.get("over_cuota", 0.0) or "—"
            row["under_cuota"] = info.get("under_cuota", 0.0) or "—"
            row["edge_total_runs"] = round(diff, 2)
            # Probabilidad Over/Under: aprox. normal alrededor del total
            # proyectado (sigma típica de la MLB ~ 3.1 carreras)
            from math import erf, sqrt
            sigma = 3.1
            p_over = 0.5 * (1 + erf(diff / (sigma * sqrt(2))))
            p_over = min(max(p_over, 0.02), 0.98)
            row["prob_over_%"] = round(p_over * 100, 1)
            o_c = info.get("over_cuota", 0.0)
            u_c = info.get("under_cuota", 0.0)
            row["ev_over"] = round(p_over * o_c - 1.0, 4) if o_c > 1.0 else np.nan
            row["ev_under"] = round((1 - p_over) * u_c - 1.0, 4) if u_c > 1.0 else np.nan
        predicciones.append(row)

    return pd.DataFrame(predicciones)

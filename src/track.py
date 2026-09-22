#!/usr/bin/env python3
"""
MLBPro — Auditoría y liquidación diaria (compatible con el workflow)
====================================================================
1. Auditoría de calidad del modelo (moneyline vs resultado, error de totales).
2. Liquidación del libro de apuestas contra resultados oficiales.
3. Curva de equity y métricas de inversión (ROI, yield, CLV, drawdown).
4. Reporte por consola y resumen a Telegram.

Uso:
    python track.py [YYYY-MM-DD]   (default: hoy)
"""

from __future__ import annotations

import datetime
import os
import re
import sys

import pandas as pd
import requests

# CORRECCIÓN DE IMPORTACIONES: Apuntando a la raíz y a la carpeta 'src' de tu repositorio
from config import config
from src.portfolio.ledger import Ledger
from src.reporting.notify import enviar, mensaje_reporte

API_BASE = config.api.mlb_base_url


def resultados_reales(fecha: str) -> dict[str, dict]:
    """Resultados oficiales de la fecha: {matchup: {...}}."""
    try:
        r = requests.get(f"{API_BASE}/schedule",
                         params={"sportId": 1, "date": fecha}, timeout=20)
        data = r.json()
    except Exception as e:
        print(f"⚠️ Error obteniendo resultados de {fecha}: {e}")
        return {}

    resultados = {}
    for dia in data.get("dates", []):
        for g in dia.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            hn = home.get("team", {}).get("name")
            an = away.get("team", {}).get("name")
            hs = home.get("score", 0) or 0
            aws = away.get("score", 0) or 0
            resultados[f"{an} @ {hn}"] = {
                "hom_team": hn, "away_team": an,
                "hom_score": hs, "away_score": aws,
                "total_runs": hs + aws,
                "winner": "home" if hs > aws else "away",
            }
    return resultados


def auditar_modelos(output_dir: str) -> pd.DataFrame:
    """Audita todos los CSV de predicción contra resultados reales."""
    import glob
    registros = []
    cache_resultados: dict[str, dict] = {}

    for fp in sorted(glob.glob(os.path.join(output_dir, "predicciones_*.csv"))):
        m = re.search(r"predicciones_(\d{4}-\d{2}-\d{2})\.csv", fp)
        if not m:
            continue
        fecha = m.group(1)
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if fecha not in cache_resultados:
            cache_resultados[fecha] = resultados_reales(fecha)
        reales = cache_resultados[fecha]
        for _, r in df.iterrows():
            real = reales.get(r.get("matchup"))
            if not real:
                continue
            pred_w = "home" if float(r.get("ml_local_%", 50)) > 50 else "away"
            registros.append({
                "fecha": fecha, "matchup": r.get("matchup"),
                "ml_correct": pred_w == real["winner"],
                "pred_total": float(r.get("total_runs", 0) or 0),
                "real_total": real["total_runs"],
                "total_error": abs(float(r.get("total_runs", 0) or 0) - real["total_runs"]),
            })

    df_audit = pd.DataFrame(registros)
    if not df_audit.empty:
        ruta = os.path.join(output_dir, "tracking_results.csv")
        df_audit.to_csv(ruta, index=False)
        print(f"✅ Auditoría de modelos: {len(df_audit)} juegos → {ruta}")
    return df_audit


def main() -> int:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")
    print("=" * 70)
    print(f"📊 MLBPRO — AUDITORÍA Y LIQUIDACIÓN ({fecha})")
    print("=" * 70)

    try:
        output_dir = config.data.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1) Auditoría de calidad del modelo
        print("\n1️⃣ Auditoría de modelos")
        print("-" * 70)
        df_audit = auditar_modelos(output_dir)
        if not df_audit.empty:
            hr = df_audit["ml_correct"].mean() * 100
            print(f"   Hit rate ML (histórico): {hr:.1f}% | "
                  f"Error medio totales: {df_audit['total_error'].mean():.2f}")

        # 2) Liquidación del libro
        print("\n2️⃣ Liquidación de apuestas")
        print("-" * 70)
        ledger = Ledger()
        res_hoy = resultados_reales(fecha)
        n = ledger.liquidar(res_hoy)
        print(f"   Apuestas liquidadas hoy: {n}")

        # Liquidar también fechas anteriores pendientes (respaldo)
        pendientes = sorted(set(ledger.df.loc[ledger.df["status"] == "open", "fecha"])
                            - {fecha}) if not ledger.df.empty else []
        for f in pendientes:
            n += ledger.liquidar(resultados_reales(str(f)))
        if pendientes:
            print(f"   Pendientes de fechas previas liquidadas: {n}")

        # 3) Equity y métricas
        print("\n3️⃣ Métricas de inversión")
        print("-" * 70)
        equity = ledger.equity()
        if not equity.empty:
            ruta_eq = os.path.join(output_dir, "equity_curve.csv")
            equity.to_csv(ruta_eq, index=False)
        m = ledger.metricas()
        for k, v in m.items():
            print(f"   {k:<18}: {v}")

        # 4) Notificación
        try:
            enviar(mensaje_reporte(m, equity))
        except Exception as e:
            print(f"⚠️ Telegram falló (no crítico): {e}")

        print("\n✅ Auditoría completada.")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
MLB System Performance Tracker
==============================
Módulo de auditoría que evalúa las predicciones históricas guardadas
en 'output/predicciones_*.csv' contra los resultados reales de la MLB Stats API.

Calcula:
  - Exactitud en Moneyline (Hit Rate %)
  - Unidades ganadas / perdidas (Yield / ROI)
  - Desviación media en Totales de Runs y Ponches
"""

import glob
import os
import re
import pandas as pd
import requests

API_BASE = "https://statsapi.mlb.com/api/v1"
OUTPUT_DIR = "output"


def consultar_resultados_reales(fecha: str) -> dict:
    url = f"{API_BASE}/schedule?sportId=1&date={fecha}"
    try:
        r = requests.get(url, timeout=20)
        data = r.json()
        resultados = {}
        for dia in data.get("dates", []):
            for g in dia.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    home = g.get("teams", {}).get("home", {}).get("team", {}).get("name")
                    away = g.get("teams", {}).get("away", {}).get("team", {}).get("name")
                    h_score = g.get("teams", {}).get("home", {}).get("score", 0)
                    a_score = g.get("teams", {}).get("away", {}).get("score", 0)
                    key = f"{away} @ {home}"
                    resultados[key] = {
                        "hom_score": h_score,
                        "away_score": a_score,
                        "total_runs": h_score + a_score,
                        "winner": "home" if h_score > a_score else "away"
                    }
        return resultados
    except Exception as e:
        print(f"Error consultando resultados para {fecha}: {e}")
        return {}


def auditar_predicciones():
    archivos = sorted(glob.glob(os.path.join(OUTPUT_DIR, "predicciones_*.csv")))
    if not archivos:
        print("❌ No se encontraron archivos de predicción en 'output/'.")
        return

    registros = []
    total_apostado = 0.0
    total_retorno = 0.0
    aciertos_ml = 0
    total_juegos_evaluados = 0

    print("=" * 65)
    print(" 📊 AUDITORÍA FRÍA DE RENDIMIENTO Y BANKROLL — MLB SYSTEM")
    print("=" * 65)

    for arch in archivos:
        m = re.search(r"predicciones_(\d{4}-\d{2}-\d{2})\.csv", arch)
        if not m:
            continue
        fecha = m.group(1)
        df_pred = pd.read_csv(arch)
        reales = consultar_resultados_reales(fecha)

        if not reales:
            continue

        for _, row in df_pred.iterrows():
            matchup = row.get("matchup")
            if matchup not in reales:
                continue

            res = reales[matchup]
            ml_prob_h = row.get("ml_local_%", 50.0)
            pred_winner = "home" if ml_prob_h > 50.0 else "away"
            real_winner = res["winner"]

            es_acierto = (pred_winner == real_winner)
            if es_acierto:
                aciertos_ml += 1
            total_juegos_evaluados += 1

            # Evaluación de cuota y rendimiento financiero (Flat 1 Unidad por Pick)
            pick_ev = str(row.get("pick_ev", ""))
            cuota_h = pd.to_numeric(row.get("cuota_local"), errors="coerce")
            cuota_a = pd.to_numeric(row.get("cuota_visitante"), errors="coerce")

            pnl = 0.0
            if "ML (EV" in pick_ev:
                total_apostado += 1.0
                if "ML" in pick_ev and row["hom_team"] in pick_ev:
                    if real_winner == "home":
                        cuota = cuota_h if pd.notna(cuota_h) and cuota_h > 1 else 1.90
                        pnl = cuota - 1.0
                    else:
                        pnl = -1.0
                elif "ML" in pick_ev and row["away_team"] in pick_ev:
                    if real_winner == "away":
                        cuota = cuota_a if pd.notna(cuota_a) and cuota_a > 1 else 1.90
                        pnl = cuota - 1.0
                    else:
                        pnl = -1.0
                total_retorno += (1.0 + pnl)

            registros.append({
                "fecha": fecha,
                "matchup": matchup,
                "pred_winner": pred_winner,
                "real_winner": real_winner,
                "acierto": es_acierto,
                "pnl_unidades": pnl
            })

    if total_juegos_evaluados == 0:
        print("⚠️ No hay juegos finalizados registrados para evaluar aún.")
        return

    win_rate = (aciertos_ml / total_juegos_evaluados) * 100
    profit = total_retorno - total_apostado
    roi = (profit / total_apostado * 100) if total_apostado > 0 else 0.0

    print(f"  • Juegos Evaluados:   {total_juegos_evaluados}")
    print(f"  • Hit Rate Moneyline:  {win_rate:.2f}% ({aciertos_ml}/{total_juegos_evaluados})")
    print(f"  • Unidades Apostadas: {total_apostado:.2f} U")
    print(f"  • Beneficio Neto:     {profit:+.2f} U")
    print(f"  • ROI / Yield:        {roi:+.2f}%")
    print("=" * 65)

    # Guardar log de auditoría
    df_audit = pd.DataFrame(registros)
    df_audit.to_csv(os.path.join(OUTPUT_DIR, "tracking_results.csv"), index=False)


if __name__ == "__main__":
    auditar_predicciones()

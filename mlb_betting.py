#!/usr/bin/env python3
"""
MLBPro — Punto de entrada diario (compatible con el workflow de Actions)
=======================================================================
Orquesta: entrenar → predecir → propuestas de inversión (Kelly + riesgo)
→ persistir predicciones y libro de apuestas → notificar por Telegram.

Uso:
    python mlb_betting.py [YYYY-MM-DD]
"""

from __future__ import annotations

import datetime
import os
import sys

import pandas as pd

from mlbpro.config import config
from mlbpro.models import entrenar_modelos, predecir_dia
from mlbpro.portfolio.ledger import Ledger
from mlbpro.portfolio.sizing import generar_propuestas, propuestas_a_dataframe
from mlbpro.reporting.notify import enviar_lote, mensaje_predicciones


def main() -> int:
    print("=" * 70)
    print("🚀 MLBPRO v3.0 — SISTEMA DE INVERSIÓN MLB")
    print("=" * 70)

    if not config.validate():
        print("❌ Configuración inválida. Abortando.")
        return 1

    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")
    print(f"Fecha objetivo: {fecha}")
    print(f"Bankroll: {config.portfolio.bankroll_units}u | Kelly 1/{int(1/config.portfolio.kelly_fraction)} "
          f"| Stake máx {config.portfolio.max_stake_pct:.1%} | EV mín {config.portfolio.min_ev:.1%}")

    try:
        print("\n📚 FASE 1 — Entrenamiento (evaluación temporal + calibración)")
        print("-" * 70)
        entrenar_modelos()

        print("\n🔮 FASE 2 — Predicción y valor de mercado")
        print("-" * 70)
        df_pred = predecir_dia(fecha)
        if df_pred is None or df_pred.empty:
            print("⚠️ No hay juegos pendientes para la fecha. Finalizando.")
            return 0
        print(f"✅ Predicciones: {len(df_pred)} juegos")

        print("\n💼 FASE 3 — Cartera (Kelly fraccional + límites de riesgo)")
        print("-" * 70)
        ledger = Ledger()
        drawdown = ledger.drawdown_actual()
        bankroll = config.portfolio.bankroll_units + (
            float(ledger.df.loc[ledger.df["status"] == "settled", "pnl_u"].sum())
            if not ledger.df.empty else 0.0)
        print(f"Bankroll efectivo: {bankroll:.2f}u | Drawdown actual: {drawdown:.1%}")

        propuestas = generar_propuestas(df_pred, bankroll, drawdown)
        df_bets = propuestas_a_dataframe(propuestas)
        nuevas = ledger.registrar(df_bets)
        print(f"✅ Propuestas: {len(propuestas)} | Nuevas registradas: {nuevas} "
              f"| Exposición: {df_bets['stake_u'].sum() if not df_bets.empty else 0}u")

        print("\n📤 FASE 4 — Persistencia y notificación")
        print("-" * 70)
        os.makedirs(config.data.output_dir, exist_ok=True)
        ruta_csv = os.path.join(config.data.output_dir, f"predicciones_{fecha}.csv")
        df_pred.to_csv(ruta_csv, index=False)
        print(f"✅ CSV: {ruta_csv}")
        print(f"✅ Ledger: {ledger.path}")

        try:
            enviar_lote(mensaje_predicciones(fecha, df_pred, df_bets))
        except Exception as e:
            print(f"⚠️ Telegram falló (no crítico): {e}")

        print("\n" + "=" * 70)
        print("✅ PROCESO COMPLETADO")
        print("=" * 70)
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido por el usuario.")
        return 1
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

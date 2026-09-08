#!/usr/bin/env python3
"""
Performance Tracking System - Pro Edition
=========================================
Sistema de auditoría, seguimiento de rendimiento y cálculo de ROI/Yield real.
"""

import glob
import os
import re
from typing import Dict

import pandas as pd
import requests

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger('tracking', log_file='logs/tracking.log')

API_BASE = "https://statsapi.mlb.com/api/v1"


class PerformanceTracker:
    """Rastreador de rendimiento del sistema."""
    
    def __init__(self):
        self.output_dir = config.data.output_dir
        self.results_cache: Dict[str, Dict] = {}
    
    def fetch_real_results(self, date: str) -> Dict[str, Dict]:
        """Obtiene resultados reales del schedule oficial de la MLB."""
        if date in self.results_cache:
            return self.results_cache[date]
        
        url = f"{API_BASE}/schedule"
        params = {"sportId": 1, "date": date}
        
        try:
            response = requests.get(url, params=params, timeout=20)
            data = response.json()
            
            results = {}
            
            for day in data.get("dates", []):
                for game in day.get("games", []):
                    # Solo auditar partidos finalizados
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue
                    
                    home = game.get("teams", {}).get("home", {})
                    away = game.get("teams", {}).get("away", {})
                    
                    home_name = home.get("team", {}).get("name")
                    away_name = away.get("team", {}).get("name")
                    
                    h_score = home.get("score", 0)
                    a_score = away.get("score", 0)
                    
                    matchup = f"{away_name} @ {home_name}"
                    
                    results[matchup] = {
                        "hom_score": h_score,
                        "away_score": a_score,
                        "total_runs": h_score + a_score,
                        "winner": "home" if h_score > a_score else "away",
                    }
            
            self.results_cache[date] = results
            return results
            
        except Exception as e:
            logger.error(f"Error obteniendo resultados para {date}: {e}")
            return {}
    
    def audit_predictions(self) -> pd.DataFrame:
        """Audita las predicciones exportadas por mlb_betting.py."""
        prediction_files = sorted(
            glob.glob(os.path.join(self.output_dir, "predicciones_*.csv"))
        )
        
        if not prediction_files:
            logger.warning("No se encontraron archivos de predicción en la carpeta output.")
            return pd.DataFrame()
        
        logger.info(f"Auditando {len(prediction_files)} archivos de predicción...")
        
        audit_records = []
        
        for file_path in prediction_files:
            match = re.search(r"predicciones_(\d{4}-\d{2}-\d{2})\.csv", file_path)
            if not match:
                continue
            
            date_str = match.group(1)
            
            try:
                df_pred = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"Error leyendo {file_path}: {e}")
                continue
            
            real_results = self.fetch_real_results(date_str)
            if not real_results:
                continue
            
            for _, row in df_pred.iterrows():
                matchup = row.get("matchup")
                if matchup not in real_results:
                    continue
                
                real = real_results[matchup]
                
                # Evaluation of Moneyline Model Prediction
                pred_prob_h = float(row.get("ml_local_%", 50.0))
                pred_winner = "home" if pred_prob_h > 50.0 else "away"
                real_winner = real["winner"]
                ml_correct = (pred_winner == real_winner)
                
                # Totals Evaluation
                pred_total = float(row.get("total_runs", 0.0))
                real_total = real["total_runs"]
                total_error = abs(pred_total - real_total)
                
                # Value Betting Evaluation (EV Picks)
                pick_ev = str(row.get("pick_ev", "Sin Valor Claro"))
                stake_str = str(row.get("stake_rec", "0%")).replace("%", "")
                stake_pct = float(stake_str) / 100.0 if stake_str else 0.0
                
                pick_made = stake_pct > 0.0 and pick_ev != "Sin Valor Claro"
                
                pnl = 0.0
                units_staked = 0.0
                
                if pick_made:
                    # Parsear cuotas decimales con respaldo
                    c_loc = pd.to_numeric(row.get("cuota_local"), errors="coerce")
                    c_vis = pd.to_numeric(row.get("cuota_visitante"), errors="coerce")
                    
                    odds_h = float(c_loc) if pd.notna(c_loc) and c_loc > 1.0 else 1.90
                    odds_a = float(c_vis) if pd.notna(c_vis) and c_vis > 1.0 else 1.90
                    
                    units_staked = stake_pct * 100.0  # Normalizado a unidades base
                    
                    if "Local" in pick_ev:
                        if real_winner == "home":
                            pnl = units_staked * (odds_h - 1.0)
                        else:
                            pnl = -units_staked
                    elif "Visitante" in pick_ev:
                        if real_winner == "away":
                            pnl = units_staked * (odds_a - 1.0)
                        else:
                            pnl = -units_staked
                
                audit_records.append({
                    "fecha": date_str,
                    "matchup": matchup,
                    "pred_winner": pred_winner,
                    "real_winner": real_winner,
                    "ml_correct": ml_correct,
                    "pred_total": pred_total,
                    "real_total": real_total,
                    "total_error": total_error,
                    "pick_made": pick_made,
                    "pick_ev": pick_ev,
                    "units_staked": units_staked,
                    "pnl_units": pnl,
                })
        
        df_audit = pd.DataFrame(audit_records)
        if df_audit.empty:
            logger.warning("No hay datos para guardar en la auditoría.")
            return df_audit
        
        audit_path = os.path.join(self.output_dir, "tracking_results.csv")
        df_audit.to_csv(audit_path, index=False)
        logger.info(f"Auditoría ejecutada con éxito. Guardado en: {audit_path}")
        
        return df_audit
    
    def generate_report(self, df_audit: pd.DataFrame):
        """Genera y muestra en logs el reporte cuantitativo de métricas."""
        if df_audit.empty:
            logger.warning("No se encontraron registros auditados.")
            return
        
        total_games = len(df_audit)
        ml_correct = df_audit["ml_correct"].sum()
        hit_rate = (ml_correct / total_games) * 100 if total_games > 0 else 0.0
        avg_total_error = df_audit["total_error"].mean()
        
        picks_made = df_audit[df_audit["pick_made"]]
        total_picks = len(picks_made)
        
        if total_picks > 0:
            total_staked = picks_made["units_staked"].sum()
            profit = picks_made["pnl_units"].sum()
            roi = (profit / total_staked * 100.0) if total_staked > 0 else 0.0
            wins = len(picks_made[picks_made["pnl_units"] > 0])
            picks_hit_rate = (wins / total_picks) * 100.0
        else:
            total_staked = 0.0
            profit = 0.0
            roi = 0.0
            picks_hit_rate = 0.0
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 REPORTE DE AUDITORÍA Y RENDIMIENTO MLB ML")
        logger.info("=" * 70)
        logger.info(f"Juegos Evaluados:          {total_games}")
        logger.info(f"Hit Rate General ML:       {hit_rate:.2f}% ({ml_correct}/{total_games})")
        logger.info(f"Error Promedio Totales:    {avg_total_error:.2f} carreras")
        logger.info("-" * 70)
        logger.info(f"Picks +EV Ejecutados:      {total_picks}")
        logger.info(f"Hit Rate en Picks +EV:     {picks_hit_rate:.2f}%")
        logger.info(f"Capital Apostado:          {total_staked:.2f} U")
        logger.info(f"Beneficio Neto (PnL):      {profit:+.2f} U")
        logger.info(f"ROI / Yield del Sistema:   {roi:+.2f}%")
        logger.info("=" * 70)


def main():
    tracker = PerformanceTracker()
    df_audit = tracker.audit_predictions()
    tracker.generate_report(df_audit)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Performance Tracking System
===========================
Sistema de auditoría y seguimiento de predicciones.

Calcula:
  - Hit Rate (%)
  - ROI / Yield
  - Desviación en totales
  - Análisis de sesgo
"""

import glob
import os
import re
from typing import Dict, List

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
        """
        Obtiene resultados reales de un día.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
        
        Returns:
            Diccionario con resultados por matchup
        """
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
        """
        Audita todas las predicciones históricas.
        
        Returns:
            DataFrame con resultados de auditoría
        """
        prediction_files = sorted(
            glob.glob(os.path.join(self.output_dir, "predicciones_*.csv"))
        )
        
        if not prediction_files:
            logger.warning("No se encontraron archivos de predicción.")
            return pd.DataFrame()
        
        logger.info(f"Auditando {len(prediction_files)} archivos...")
        
        audit_records = []
        
        for file_path in prediction_files:
            match = re.search(r"predicciones_(\d{4}-\d{2}-\d{2})\.csv", file_path)
            if not match:
                continue
            
            date_str = match.group(1)
            
            # Cargar predicciones
            try:
                df_pred = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"Error leyendo {file_path}: {e}")
                continue
            
            # Obtener resultados reales
            real_results = self.fetch_real_results(date_str)
            
            if not real_results:
                continue
            
            # Evaluar cada predicción
            for _, row in df_pred.iterrows():
                matchup = row.get("matchup")
                
                if matchup not in real_results:
                    continue
                
                real = real_results[matchup]
                
                # Evaluar moneyline
                pred_prob_h = row.get("ml_local_%", 50.0)
                pred_winner = "home" if pred_prob_h > 50.0 else "away"
                real_winner = real["winner"]
                ml_correct = (pred_winner == real_winner)
                
                # Evaluar totales
                pred_total = row.get("total_runs", 0.0)
                real_total = real["total_runs"]
                total_error = abs(pred_total - real_total)
                
                # Evaluar EV
                pick_ev = str(row.get("ev_pick", ""))
                stake = row.get("ev_stake", "0%")
                
                pnl = 0.0
                if "%" in stake and stake != "0%":
                    # Hubo una apuesta
                    odds_h = row.get("cuota_hom_ml_odds", 0.0)
                    odds_a = row.get("cuota_away_ml_odds", 0.0)
                    
                    if row.get("hom_team") in pick_ev:
                        if real_winner == "home":
                            pnl = (odds_h - 1.0) if odds_h > 1 else 0.9
                        else:
                            pnl = -1.0
                    elif row.get("away_team") in pick_ev:
                        if real_winner == "away":
                            pnl = (odds_a - 1.0) if odds_a > 1 else 0.9
                        else:
                            pnl = -1.0
                
                audit_records.append({
                    "fecha": date_str,
                    "matchup": matchup,
                    "pred_winner": pred_winner,
                    "real_winner": real_winner,
                    "ml_correct": ml_correct,
                    "pred_total": pred_total,
                    "real_total": real_total,
                    "total_error": total_error,
                    "pick_made": "%" in stake and stake != "0%",
                    "pnl_units": pnl,
                })
        
        df_audit = pd.DataFrame(audit_records)
        
        if df_audit.empty:
            logger.warning("No hay datos de auditoría.")
            return df_audit
        
        # Guardar resultados
        audit_path = os.path.join(self.output_dir, "tracking_results.csv")
        df_audit.to_csv(audit_path, index=False)
        logger.info(f"Resultados de auditoría guardados: {audit_path}")
        
        return df_audit
    
    def generate_report(self, df_audit: pd.DataFrame):
        """
        Genera reporte de rendimiento.
        
        Args:
            df_audit: DataFrame de auditoría
        """
        if df_audit.empty:
            logger.warning("No hay datos para generar reporte.")
            return
        
        total_games = len(df_audit)
        ml_correct = df_audit["ml_correct"].sum()
        hit_rate = (ml_correct / total_games) * 100
        
        avg_total_error = df_audit["total_error"].mean()
        
        picks_made = df_audit[df_audit["pick_made"]]
        total_picks = len(picks_made)
        
        if total_picks > 0:
            total_staked = total_picks  # Asumimos 1U por pick
            total_return = total_staked + picks_made["pnl_units"].sum()
            profit = total_return - total_staked
            roi = (profit / total_staked) * 100
        else:
            total_staked = 0
            total_return = 0
            profit = 0
            roi = 0.0
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 REPORTE DE RENDIMIENTO")
        logger.info("=" * 70)
        logger.info(f"Juegos Evaluados:       {total_games}")
        logger.info(f"Hit Rate Moneyline:     {hit_rate:.2f}% ({ml_correct}/{total_games})")
        logger.info(f"Error Promedio Totales: {avg_total_error:.2f} runs")
        logger.info("-" * 70)
        logger.info(f"Picks Realizados:       {total_picks}")
        logger.info(f"Unidades Apostadas:     {total_staked:.2f} U")
        logger.info(f"Beneficio Neto:         {profit:+.2f} U")
        logger.info(f"ROI / Yield:            {roi:+.2f}%")
        logger.info("=" * 70)


def main():
    """Ejecuta el tracking."""
    tracker = PerformanceTracker()
    df_audit = tracker.audit_predictions()
    tracker.generate_report(df_audit)


if __name__ == "__main__":
    main()

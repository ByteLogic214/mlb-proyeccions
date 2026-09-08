"""
Report Generator
================
Generador de reportes en múltiples formatos.
"""

import os
from datetime import datetime
from typing import List

import pandas as pd

from src.config import config
from src.models.predictor import PredictionResult
from src.utils.logger import logger


class ReportGenerator:
    """Generador de reportes de predicciones."""
    
    def __init__(self):
        self.output_dir = config.data.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_csv(
        self,
        predictions: List[PredictionResult],
        date: str
    ) -> str:
        """
        Genera reporte CSV.
        
        Args:
            predictions: Lista de predicciones
            date: Fecha del reporte
        
        Returns:
            Ruta del archivo generado
        """
        df = pd.DataFrame([p.to_dict() for p in predictions])
        
        csv_path = os.path.join(self.output_dir, f"predicciones_{date}.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"CSV generado: {csv_path}")
        return csv_path
    
    def generate_markdown(
        self,
        predictions: List[PredictionResult],
        date: str
    ) -> str:
        """
        Genera reporte Markdown profesional.
        
        Args:
            predictions: Lista de predicciones
            date: Fecha del reporte
        
        Returns:
            Ruta del archivo generado
        """
        lines = [
            f"# ⚾ Predicciones MLB — {date}",
            "",
            f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Sistema:** MLB Betting ML v2.0  ",
            f"**Modelo:** Random Forest + Park Factors + Odds Analysis",
            "",
            "---",
            "",
            "## 📊 Resumen Ejecutivo",
            "",
            f"- **Juegos Analizados:** {len(predictions)}",
            f"- **Picks EV+:** {sum(1 for p in predictions if 'Sin Valor' not in p.ev_analysis.get('pick', ''))}",
            "",
            "---",
            "",
            "## 🎯 Predicciones Detalladas",
            "",
        ]
        
        for pred in predictions:
            lines.extend([
                f"### {pred.matchup}",
                "",
                "**Abridores:**",
                f"- Visitante: {pred.away_pitcher}",
                f"- Local: {pred.hom_pitcher}",
                "",
                "**Factor de Parque:** " + f"`{pred.park_factor:.2f}`",
                "",
                "#### Moneyline",
                "",
                "| Equipo | Probabilidad | Cuota | EV% |",
                "|--------|--------------|-------|-----|",
            ])
            
            prob_h = pred.predictions.get("ml_local_%", 0)
            prob_a = pred.predictions.get("ml_visitante_%", 0)
            odds_h = pred.odds.get("hom_ml_odds", "—")
            odds_a = pred.odds.get("away_ml_odds", "—")
            
            lines.extend([
                f"| {pred.hom_team} (L) | {prob_h:.1f}% | {odds_h} | — |",
                f"| {pred.away_team} (V) | {prob_a:.1f}% | {odds_a} | — |",
                "",
            ])
            
            if pred.ev_analysis:
                pick = pred.ev_analysis.get("pick", "Sin Valor")
                ev_pct = pred.ev_analysis.get("ev_pct", "—")
                stake = pred.ev_analysis.get("stake", "0%")
                confidence = pred.ev_analysis.get("confidence", "Ninguna")
                
                if "Sin Valor" not in pick:
                    lines.extend([
                        "#### ✅ **PICK RECOMENDADO**",
                        "",
                        f"- **Selección:** {pick}",
                        f"- **Expected Value:** {ev_pct}",
                        f"- **Stake Sugerido:** {stake}",
                        f"- **Confianza:** {confidence}",
                        "",
                    ])
            
            lines.extend([
                "#### Proyecciones",
                "",
                "| Mercado | Proyección |",
                "|---------|------------|",
            ])
            
            markets = [
                ("Total Runs", "total_runs"),
                ("Runs Local", "hom_runs"),
                ("Runs Visitante", "away_runs"),
                ("Hits Local", "hom_hits"),
                ("Hits Visitante", "away_hits"),
                ("K Total", "total_k"),
            ]
            
            for label, key in markets:
                value = pred.predictions.get(key)
                if value is not None:
                    lines.append(f"| {label} | **{value:.1f}** |")
            
            lines.extend(["", "---", ""])
        
        lines.extend([
            "",
            "## ⚠️ Disclaimer",
            "",
            "Este análisis es generado automáticamente con fines educativos.  ",
            "**No constituye asesoramiento financiero ni de apuestas.**  ",
            "Apuesta responsablemente y dentro de tus posibilidades.",
            "",
            "---",
            "",
            f"*Powered by MLB Stats API • {datetime.now().year}*",
        ])
        
        md_path = os.path.join(self.output_dir, f"predicciones_{date}.md")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Markdown generado: {md_path}")
        return md_path
    
    def generate_all(
        self,
        predictions: List[PredictionResult],
        date: str
    ) -> dict:
        """
        Genera todos los formatos de reporte.
        
        Args:
            predictions: Lista de predicciones
            date: Fecha del reporte
        
        Returns:
            Diccionario con rutas de archivos
        """
        csv_path = self.generate_csv(predictions, date)
        md_path = self.generate_markdown(predictions, date)
        
        return {
            "csv": csv_path,
            "markdown": md_path,
        }

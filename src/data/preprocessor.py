"""
Data Preprocessing
==================
Normalización y limpieza de datos.
"""

import numpy as np
import pandas as pd

from src.config import config
from src.utils.logger import logger
from src.utils.validators import DataValidator


class DataPreprocessor:
    """Preprocesador de datos MLB."""
    
    def __init__(self):
        self.validator = DataValidator()
    
    @staticmethod
    def safe_float(value, default=0.0):
        """Convierte a float de forma segura."""
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def parse_innings(value):
        """Parsea innings pitched (formato X.Y donde Y es outs)."""
        if isinstance(value, str) and "." in value:
            try:
                integer, _, fraction = value.partition(".")
                outs_map = {"0": 0.0, "1": 1/3, "2": 2/3}
                return float(integer) + outs_map.get(fraction[:1], 0.0)
            except ValueError:
                pass
        return DataPreprocessor.safe_float(value)
    
    def normalize_game_log(self, df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
        """
        Normaliza un game log (hitting o pitching).
        
        Args:
            df: DataFrame raw
            stat_type: 'hitting' o 'pitching'
        
        Returns:
            DataFrame normalizado
        """
        if df is None or df.empty:
            logger.warning(f"Game log {stat_type} vacío")
            return pd.DataFrame()
        
        df = df.copy()
        
        # Normalizar fechas
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date", "team_id", "opponent_id"])
        
        # Ordenar cronológicamente
        df = df.sort_values("date").reset_index(drop=True)
        
        # Convertir IDs a enteros
        df["team_id"] = df["team_id"].astype(int)
        df["opponent_id"] = df["opponent_id"].astype(int)
        
        # Normalizar columnas numéricas
        stat_cols = (config.features.hitting_stats if stat_type == "hitting" 
                     else config.features.pitching_stats)
        
        for col in stat_cols:
            if col in df.columns:
                if col == "inningsPitched":
                    df[col] = df[col].apply(self.parse_innings)
                else:
                    df[col] = df[col].apply(self.safe_float)
        
        # Métricas derivadas para pitching
        if stat_type == "pitching" and "inningsPitched" in df.columns:
            ip = df["inningsPitched"].replace(0, np.nan)
            df["k_per_9"] = df["strikeOuts"] / (ip / 9)
            df["whip_game"] = (df["hits"] + df["baseOnBalls"]) / ip
            df[["k_per_9", "whip_game"]] = df[["k_per_9", "whip_game"]].fillna(0.0)
        
        # Métricas derivadas para hitting
        if stat_type == "hitting" and "atBats" in df.columns:
            df["avg_game"] = df["hits"] / df["atBats"].replace(0, np.nan)
            df["avg_game"] = df["avg_game"].fillna(0.0)
        
        logger.debug(f"Game log {stat_type} normalizado: {len(df)} registros")
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Elimina outliers usando desviación estándar.
        
        Args:
            df: DataFrame
            column: Columna a limpiar
            n_std: Número de desviaciones estándar
        
        Returns:
            DataFrame sin outliers
        """
        if column not in df.columns:
            return df
        
        mean = df[column].mean()
        std = df[column].std()
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        before = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed = before - len(df)
        
        if removed > 0:
            logger.debug(f"Outliers eliminados en {column}: {removed} registros")
        
        return df

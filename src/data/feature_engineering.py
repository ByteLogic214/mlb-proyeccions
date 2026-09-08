"""
Feature Engineering
===================
Generación avanzada de características para ML.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import config
from src.utils.logger import logger


class FeatureEngineer:
    """Ingeniero de características para datos MLB."""
    
    def __init__(self):
        self.windows = config.model.windows
        self.hitting_stats = config.features.hitting_stats
        self.pitching_stats = config.features.pitching_stats
        self.park_factors = config.park_factors
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Crea características de medias móviles.
        
        Args:
            df: DataFrame con datos temporales
            columns: Columnas para calcular features
            prefix: Prefijo para nombres de columnas
        
        Returns:
            DataFrame con nuevas características
        """
        if df.empty:
            return pd.DataFrame()
        
        features = {
            "date": df["date"].values,
            "team_id": df["team_id"].values,
            "opponent_id": df["opponent_id"].values,
        }
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Medias móviles por ventana
            for window in self.windows:
                rolling_col = f"{prefix}{col}_avg{window}"
                features[rolling_col] = (
                    df[col]
                    .shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                    .values
                )
            
            # Media de temporada (expanding mean)
            season_col = f"{prefix}{col}_season"
            features[season_col] = (
                df[col]
                .expanding()
                .mean()
                .shift(1)
                .values
            )
            
            # Tendencia (últimos 5 vs previos 10)
            last5 = df[col].shift(1).rolling(5, min_periods=1).mean()
            prev10 = df[col].shift(6).rolling(10, min_periods=1).mean()
            trend_col = f"{prefix}{col}_trend"
            features[trend_col] = (last5 - prev10).values
            
            # Volatilidad (std últimos 10 juegos)
            volatility_col = f"{prefix}{col}_volatility"
            features[volatility_col] = (
                df[col]
                .shift(1)
                .rolling(10, min_periods=2)
                .std()
                .fillna(0.0)
                .values
            )
        
        logger.debug(f"Features rolling creadas: {len(features)} columnas")
        return pd.DataFrame(features)
    
    def create_team_features(
        self,
        hitting_df: pd.DataFrame,
        pitching_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Crea características combinadas de bateo y pitcheo.
        
        Args:
            hitting_df: DataFrame de bateo
            pitching_df: DataFrame de pitcheo
        
        Returns:
            DataFrame con características combinadas
        """
        hitting_features = self.create_rolling_features(
            hitting_df,
            self.hitting_stats,
            prefix="h_"
        )
        
        pitching_features = self.create_rolling_features(
            pitching_df,
            self.pitching_stats,
            prefix="p_"
        )
        
        if hitting_features.empty:
            return pitching_features
        if pitching_features.empty:
            return hitting_features
        
        combined = pd.merge(
            hitting_features,
            pitching_features,
            on=["date", "team_id", "opponent_id"],
            how="outer"
        )
        
        logger.debug(f"Features combinadas: {len(combined)} registros")
        return combined
    
    def get_latest_features(
        self,
        features_df: pd.DataFrame,
        target_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Obtiene las últimas características antes de una fecha.
        
        Args:
            features_df: DataFrame de características
            target_date: Fecha objetivo
        
        Returns:
            Diccionario con características
        """
        if features_df is None or features_df.empty:
            return {}
        
        prior = features_df[features_df["date"] < target_date]
        
        if prior.empty:
            return {}
        
        latest = prior.sort_values("date").iloc[-1].to_dict()
        
        # Eliminar columnas de identidad
        for key in ["date", "team_id", "opponent_id"]:
            latest.pop(key, None)
        
        return latest
    
    def calculate_rest_days(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp
    ) -> float:
        """
        Calcula días de descanso desde el último juego.
        
        Args:
            df: DataFrame con fechas de juegos
            target_date: Fecha objetivo
        
        Returns:
            Número de días de descanso
        """
        if df is None or df.empty:
            return 0.0
        
        prior_games = df[df["date"] < target_date]
        
        if prior_games.empty:
            return 0.0
        
        last_game = prior_games["date"].max()
        rest_days = (target_date - last_game).days
        
        return max(0.0, float(rest_days))
    
    def apply_park_factor(
        self,
        value: float,
        team_name: str,
        stat_type: str
    ) -> float:
        """
        Aplica factor de parque a una estadística.
        
        Args:
            value: Valor original
            team_name: Nombre del equipo local
            stat_type: Tipo de estadística
        
        Returns:
            Valor ajustado por parque
        """
        # Solo aplicar a métricas de producción ofensiva
        offensive_stats = ["runs", "hits", "homeRuns", "totalBases"]
        
        if stat_type not in offensive_stats:
            return value
        
        factor = self.park_factors.get_factor(team_name)
        adjusted = value * factor
        
        return adjusted
    
    def create_matchup_features(
        self,
        home_features: Dict[str, float],
        away_features: Dict[str, float],
        park_factor: float,
        home_rest: float,
        away_rest: float
    ) -> Dict[str, float]:
        """
        Crea características para un enfrentamiento específico.
        
        Args:
            home_features: Características del equipo local
            away_features: Características del equipo visitante
            park_factor: Factor de parque
            home_rest: Días de descanso local
            away_rest: Días de descanso visitante
        
        Returns:
            Diccionario con todas las características del matchup
        """
        features = {"park_factor": park_factor}
        
        # Agregar características del local con prefijo
        for key, value in home_features.items():
            features[f"hom_{key}"] = value
        
        # Agregar características del visitante con prefijo
        for key, value in away_features.items():
            features[f"away_{key}"] = value
        
        # Descanso
        features["hom_descanso"] = home_rest
        features["away_descanso"] = away_rest
        
        # Características diferenciales (ventaja local vs visitante)
        common_keys = set(home_features.keys()) & set(away_features.keys())
        for key in common_keys:
            if key.endswith("_season") or key.endswith("_avg5"):
                diff_key = f"diff_{key}"
                features[diff_key] = home_features[key] - away_features[key]
        
        return features


class TargetEngineer:
    """Generador de variables objetivo (targets)."""
    
    @staticmethod
    def create_targets(game_data: Dict) -> Dict[str, float]:
        """
        Crea todas las variables objetivo de un juego.
        
        Args:
            game_data: Diccionario con datos del juego
        
        Returns:
            Diccionario con targets
        """
        hom_score = float(game_data.get("hom_score", 0))
        away_score = float(game_data.get("away_score", 0))
        
        targets = {
            "hom_runs": hom_score,
            "away_runs": away_score,
            "total_runs": hom_score + away_score,
            "hom_win": 1 if hom_score > away_score else 0,
        }
        
        # Agregar otras métricas si están disponibles
        optional_metrics = [
            "hom_hits", "away_hits",
            "hom_total_bases", "away_total_bases",
            "hom_k", "away_k",
            "hom_p_k", "away_p_k",
        ]
        
        for metric in optional_metrics:
            if metric in game_data:
                targets[metric] = float(game_data[metric])
        
        # Total de ponches
        if "hom_k" in targets and "away_k" in targets:
            targets["total_k"] = targets["hom_k"] + targets["away_k"]
        
        return targets

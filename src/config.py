"""
Configuration Management
========================
Centraliza todas las configuraciones del sistema.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class APIConfig:
    """Configuración de APIs externas."""
    mlb_base_url: str = "https://statsapi.mlb.com/api/v1"
    odds_api_url: str = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    timeout: int = 45
    max_retries: int = 4
    retry_delay: int = 10


@dataclass
class DataConfig:
    """Configuración de datos y cache."""
    season: int = 2026
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "output"
    cache_ttl: int = 6 * 3600  # 6 horas


@dataclass
class ModelConfig:
    """Configuración de modelos ML."""
    windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    rf_n_estimators: int = 250
    rf_max_depth_classifier: int = 8
    rf_max_depth_regressor: int = 10
    rf_min_samples_leaf_classifier: int = 4
    rf_min_samples_leaf_regressor: int = 3
    random_state: int = 42
    test_size: float = 0.2
    min_training_samples: int = 30


@dataclass
class FeatureConfig:
    """Configuración de características."""
    hitting_stats: List[str] = field(default_factory=lambda: [
        "runs", "hits", "doubles", "triples", "homeRuns", "totalBases",
        "strikeOuts", "baseOnBalls", "walk", "atBats"
    ])
    pitching_stats: List[str] = field(default_factory=lambda: [
        "runs", "hits", "strikeOuts", "baseOnBalls", "homeRuns",
        "inningsPitched", "era", "whip"
    ])
    targets: List[str] = field(default_factory=lambda: [
        "hom_runs", "away_runs", "total_runs",
        "hom_hits", "away_hits",
        "hom_total_bases", "away_total_bases",
        "hom_k", "away_k",
        "hom_p_k", "away_p_k",
        "total_k",
        "hom_win",
    ])


@dataclass
class ParkFactors:
    """Factores de parque MLB."""
    factors: Dict[str, float] = field(default_factory=lambda: {
        "Colorado Rockies": 1.35,
        "Boston Red Sox": 1.08,
        "Cincinnati Reds": 1.12,
        "Philadelphia Phillies": 1.06,
        "Kansas City Royals": 1.05,
        "Chicago Cubs": 1.04,
        "Baltimore Orioles": 1.02,
        "Texas Rangers": 1.02,
        "Los Angeles Dodgers": 1.01,
        "Atlanta Braves": 1.01,
        "Minnesota Twins": 1.00,
        "Chicago White Sox": 1.00,
        "St. Louis Cardinals": 0.99,
        "Milwaukee Brewers": 0.99,
        "New York Yankees": 0.98,
        "Houston Astros": 0.98,
        "Toronto Blue Jays": 0.98,
        "Arizona Diamondbacks": 0.97,
        "Washington Nationals": 0.97,
        "Los Angeles Angels": 0.96,
        "San Francisco Giants": 0.95,
        "Detroit Tigers": 0.95,
        "Pittsburgh Pirates": 0.95,
        "New York Mets": 0.94,
        "Cleveland Guardians": 0.94,
        "Tampa Bay Rays": 0.93,
        "Miami Marlins": 0.93,
        "San Diego Padres": 0.92,
        "Seattle Mariners": 0.91,
        "Athletics": 0.96,
    })

    def get_factor(self, team_name: str) -> float:
        """Obtiene el factor de parque para un equipo."""
        return self.factors.get(team_name, 1.00)


class Config:
    """Configuración global del sistema."""

    def __init__(self):
        self.api = APIConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.features = FeatureConfig()
        self.park_factors = ParkFactors()

    def validate(self) -> bool:
        """Valida la configuración."""
        try:
            os.makedirs(self.data.data_dir, exist_ok=True)
            os.makedirs(self.data.model_dir, exist_ok=True)
            os.makedirs(self.data.output_dir, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Error validando configuración: {e}")
            return False


# Instancia global
config = Config()

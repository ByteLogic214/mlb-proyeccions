"""
Configuración central del sistema
=================================
Toda la configuración vive aquí. Los parámetros de inversión (bankroll,
Kelly, límites de riesgo) pueden sobreescribirse por variables de entorno
sin tocar código — ideal para GitHub Actions secrets/vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class APIConfig:
    """APIs externas."""
    mlb_base_url: str = "https://statsapi.mlb.com/api/v1"
    odds_api_url: str = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    timeout: int = 45
    max_retries: int = 4
    retry_delay: int = 10


@dataclass
class TelegramConfig:
    """Notificaciones Telegram (opcional, nunca crítico)."""
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    max_len: int = 4000


@dataclass
class DataConfig:
    """Datos y cache."""
    season: int = 2026
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "output"
    cache_ttl: int = 6 * 3600  # 6 horas


@dataclass
class ModelConfig:
    """Modelos ML."""
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
class PortfolioConfig:
    """
    Parámetros de gestión de inversión (bankroll y riesgo).

    Env overrides: BANKROLL_UNITS, KELLY_FRACTION, MAX_STAKE_PCT,
    MAX_DAILY_EXPOSURE_PCT, MIN_EV, MIN_ODDS, MAX_ODDS, MAX_BETS_PER_DAY,
    MAX_DRAWDOWN_PCT.
    """
    bankroll_units: float = field(default_factory=lambda: _env_float("BANKROLL_UNITS", 100.0))
    kelly_fraction: float = field(default_factory=lambda: _env_float("KELLY_FRACTION", 0.25))
    max_stake_pct: float = field(default_factory=lambda: _env_float("MAX_STAKE_PCT", 0.02))
    max_daily_exposure_pct: float = field(default_factory=lambda: _env_float("MAX_DAILY_EXPOSURE_PCT", 0.06))
    min_ev: float = field(default_factory=lambda: _env_float("MIN_EV", 0.03))
    min_odds: float = field(default_factory=lambda: _env_float("MIN_ODDS", 1.50))
    max_odds: float = field(default_factory=lambda: _env_float("MAX_ODDS", 3.00))
    max_bets_per_day: int = field(default_factory=lambda: _env_int("MAX_BETS_PER_DAY", 3))
    max_drawdown_pct: float = field(default_factory=lambda: _env_float("MAX_DRAWDOWN_PCT", 0.25))


class Config:
    """Configuración global."""

    def __init__(self):
        self.api = APIConfig()
        self.telegram = TelegramConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.portfolio = PortfolioConfig()

    def validate(self) -> bool:
        """Valida coherencia de parámetros y prepara directorios."""
        try:
            os.makedirs(self.data.data_dir, exist_ok=True)
            os.makedirs(self.data.model_dir, exist_ok=True)
            os.makedirs(self.data.output_dir, exist_ok=True)
        except OSError:
            return False
        p = self.portfolio
        return (
            self.data.season > 2000
            and 0 < p.kelly_fraction <= 1.0
            and 0 < p.max_stake_pct <= 0.25
            and p.min_odds >= 1.01
            and p.max_odds > p.min_odds
            and p.min_ev >= 0
            and 0 < p.max_drawdown_pct <= 1.0
        )


config = Config()

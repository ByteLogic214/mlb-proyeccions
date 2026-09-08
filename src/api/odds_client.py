"""
Odds API Client
===============
Cliente para obtener cuotas de apuestas en vivo.
"""

from typing import Dict

import requests

from src.config import config
from src.utils.logger import logger


class OddsAPIError(Exception):
    """Excepción para errores de Odds API."""
    pass


class OddsClient:
    """Cliente para Odds-API.io."""
    
    def __init__(self):
        self.api_key = config.api.odds_api_key
        self.base_url = config.api.odds_api_url
        self.timeout = 15
    
    def get_odds(self, date: str) -> Dict[str, Dict[str, float]]:
        """
        Obtiene cuotas para una fecha específica.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
        
        Returns:
            Diccionario con cuotas por partido
        """
        if not self.api_key:
            logger.warning("ODDS_API_KEY no configurada. Omitiendo cuotas.")
            return {}
        
        params = {
            "apiKey": self.api_key,
            "sport": "baseball_mlb",
            "date": date
        }
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"Error consultando Odds API: {response.status_code}")
                return {}
            
            return self._parse_odds(response.json())
            
        except requests.RequestException as e:
            logger.error(f"Error en petición a Odds API: {e}")
            return {}
    
    def _parse_odds(self, data: Dict) -> Dict[str, Dict[str, float]]:
        """Parsea la respuesta de la API de odds."""
        odds_map = {}
        
        for game in data.get("data", []):
            home = game.get("home_team")
            away = game.get("away_team")
            
            if not home or not away:
                continue
            
            match_key = f"{away} @ {home}"
            bookmakers = game.get("bookmakers", [])
            
            if not bookmakers:
                continue
            
            # Tomar el primer bookmaker principal
            bm = bookmakers[0]
            markets = bm.get("markets", {})
            
            odds_map[match_key] = {
                "hom_ml_odds": float(markets.get("h2h", {}).get("home", 0.0)),
                "away_ml_odds": float(markets.get("h2h", {}).get("away", 0.0)),
                "total_line": float(markets.get("totals", {}).get("line", 0.0)),
            }
        
        logger.info(f"Cuotas obtenidas para {len(odds_map)} partidos")
        return odds_map

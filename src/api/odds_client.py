"""
Odds API Client
===============
Cliente para obtener cuotas de apuestas de the-odds-api.com.

Referencia: https://the-odds-api.com/
Endpoint: https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/
"""

from typing import Dict

import requests

from src.config import config
from src.utils.logger import logger


class OddsAPIError(Exception):
    """Excepción para errores de Odds API."""
    pass


class OddsClient:
    """Cliente para the-odds-api.com."""

    def __init__(self):
        self.api_key = config.api.odds_api_key
        self.base_url = config.api.odds_api_url
        self.timeout = 15

    def get_odds(self, date: str) -> Dict[str, Dict[str, float]]:
        """
        Obtiene cuotas para una fecha específica.

        Endpoint:
            GET https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/
            ?apiKey=YOUR_KEY&regions=us&markets=h2h,totals&date=YYYY-MM-DD

        Args:
            date: Fecha en formato YYYY-MM-DD

        Returns:
            Diccionario con cuotas por matchup.
            Ejemplo: {"Photo Rockies @ LAD D. Dodgers": {"hom_ml_odds": -150, ...}}
        """
        if not self.api_key:
            logger.warning("ODDS_API_KEY no configurada. Omitiendo cuotas.")
            return {}

        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,totals",
            "oddsFormat": "american",
            "date": date,
        }

        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Error consultando The Odds API: "
                    f"HTTP {response.status_code}"
                )
                return {}

            data = response.json()
            return self._parse_odds(data)

        except requests.RequestException as e:
            logger.error(f"Error en petición a The Odds API: {e}")
            return {}

    def _parse_odds(self, data: list) -> Dict[str, Dict[str, float]]:
        """
        Parsea la respuesta de the-odds-api.com.

        Formato de respuesta (array de juegos):
        [
            {
                "id": "...",
                "sport_key": "baseball_mlb",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "bookmakers": [
                    {
                        "key": "betmgm",
                        "title": "BetMGM",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Red Sox", "price": +130},
                                    {"name": "New York Yankees", "price": -150}
                                ]
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": -110, "point": 8.5},
                                    {"name": "Under", "price": -110, "point": 8.5}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        """
        odds_map: Dict[str, Dict[str, float]] = {}

        if not isinstance(data, list):
            logger.warning("Respuesta de The Odds API no es una lista")
            return odds_map

        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")

            if not home or not away:
                continue

            match_key = f"{away} @ {home}"
            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            # Tomar el primer bookmaker disponible
            bm = bookmakers[0]
            markets = {m["key"]: m for m in bm.get("markets", [])}

            # --- H2H (Moneyline) ---
            h2h_market = markets.get("h2h", {})
            h2h_outcomes = {
                o["name"]: o.get("price", 0.0)
                for o in h2h_market.get("outcomes", [])
            }
            hom_ml_odds = h2h_outcomes.get(home, 0.0)
            away_ml_odds = h2h_outcomes.get(away, 0.0)

            # --- Totals ---
            totals_market = markets.get("totals", {})
            totals_outcomes = totals_market.get("outcomes", [])
            total_line = 0.0
            for outcome in totals_outcomes:
                if "point" in outcome:
                    total_line = float(outcome["point"])
                    break

            odds_map[match_key] = {
                "hom_ml_odds": float(hom_ml_odds),
                "away_ml_odds": float(away_ml_odds),
                "total_line": float(total_line),
            }

        logger.info(
            f"Cuotas obtenidas para {len(odds_map)} partidos "
            f"(fuente: the-odds-api.com)"
        )
        return odds_map

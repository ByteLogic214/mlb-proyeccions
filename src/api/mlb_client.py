"""
MLB Stats API Client
====================
Cliente robusto para la API de MLB con retry y cache.
"""

import time
from typing import Any, Dict, Optional

import pandas as pd
import requests

from src.config import config
from src.utils.cache import CacheManager
from src.utils.logger import logger


class MLBAPIError(Exception):
    """Excepción personalizada para errores de API."""
    pass


class MLBClient:
    """Cliente para MLB Stats API."""
    
    def __init__(self):
        self.base_url = config.api.mlb_base_url
        self.timeout = config.api.timeout
        self.max_retries = config.api.max_retries
        self.retry_delay = config.api.retry_delay
        self.cache = CacheManager(config.data.data_dir, config.data.cache_ttl)
        self.session = requests.Session()
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Realiza una petición HTTP con retry exponential backoff.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
        
        Returns:
            Respuesta JSON
        
        Raises:
            MLBAPIError: Si falla la petición
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit alcanzado. Esperando {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(f"Intento {attempt + 1}/{self.max_retries} fallido: {e}")
                
                if attempt == self.max_retries - 1:
                    raise MLBAPIError(f"Error en petición a {endpoint}: {e}")
                
                time.sleep(self.retry_delay * (attempt + 1))
        
        return {}
    
    def get_schedule(
        self,
        season: Optional[int] = None,
        date: Optional[str] = None,
        game_type: str = "R"
    ) -> pd.DataFrame:
        """
        Obtiene el calendario de juegos.
        
        Args:
            season: Temporada (default: configuración)
            date: Fecha específica (YYYY-MM-DD)
            game_type: Tipo de juego (R=Regular, P=Playoffs)
        
        Returns:
            DataFrame con calendario
        """
        season = season or config.data.season
        cache_key = f"schedule_{season}_{date or 'full'}_{game_type}"
        
        # Intentar obtener del cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Consultar API
        params = {
            "sportId": 1,
            "gameType": game_type,
            "hydrate": "probablePitcher"
        }
        
        if date:
            params["date"] = date
        else:
            params["season"] = season
        
        try:
            data = self._request("schedule", params)
            df = self._parse_schedule(data)
            
            if not df.empty:
                self.cache.set(cache_key, df)
            
            logger.info(f"Calendario obtenido: {len(df)} juegos")
            return df
            
        except MLBAPIError as e:
            logger.error(f"Error obteniendo calendario: {e}")
            return pd.DataFrame()
    
    def _parse_schedule(self, data: Dict) -> pd.DataFrame:
        """Parsea la respuesta del calendario."""
        games = []
        
        for day in data.get("dates", []):
            for game in day.get("games", []):
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                
                games.append({
                    "game_pk": game.get("gamePk"),
                    "game_date": (game.get("gameDate") or "")[:10],
                    "status": game.get("status", {}).get("abstractGameState"),
                    "hom_team_id": home.get("team", {}).get("id"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "hom_team_name": home.get("team", {}).get("name"),
                    "away_team_name": away.get("team", {}).get("name"),
                    "hom_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
                    "away_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
                    "hom_pitcher_id": home.get("probablePitcher", {}).get("id"),
                    "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
                    "hom_score": home.get("score"),
                    "away_score": away.get("score"),
                })
        
        return pd.DataFrame(games)
    
    def get_team_stats(
        self,
        team_id: int,
        stat_group: str,
        season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Obtiene estadísticas de un equipo.
        
        Args:
            team_id: ID del equipo
            stat_group: Grupo de estadísticas (hitting, pitching)
            season: Temporada
        
        Returns:
            DataFrame con estadísticas
        """
        season = season or config.data.season
        cache_key = f"team_{team_id}_{season}_{stat_group}"
        
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        params = {
            "season": season,
            "group": stat_group,
            "stats": "gameLog"
        }
        
        try:
            data = self._request(f"teams/{team_id}/stats", params)
            df = self._parse_team_stats(data, team_id)
            
            if not df.empty:
                self.cache.set(cache_key, df)
            
            logger.debug(f"Estadísticas obtenidas para equipo {team_id} ({stat_group}): {len(df)} juegos")
            return df
            
        except MLBAPIError as e:
            logger.error(f"Error obteniendo stats del equipo {team_id}: {e}")
            return pd.DataFrame()
    
    def _parse_team_stats(self, data: Dict, team_id: int) -> pd.DataFrame:
        """Parsea estadísticas de equipo."""
        rows = []
        
        for stat_block in data.get("stats", []):
            for split in stat_block.get("splits", []):
                row = {
                    "date": (split.get("date") or "")[:10],
                    "team_id": team_id,
                    "opponent_id": split.get("opponent", {}).get("id"),
                }
                row.update(split.get("stat", {}))
                rows.append(row)
        
        return pd.DataFrame(rows)

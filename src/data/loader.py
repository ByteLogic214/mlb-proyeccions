"""
Data Loader
===========
Cargador centralizado de datos de equipos.
"""

from typing import Dict, Optional

import pandas as pd

from src.api.mlb_client import MLBClient
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.utils.logger import logger


class TeamData:
    """Contenedor de datos de un equipo."""
    
    def __init__(
        self,
        team_id: int,
        hitting: pd.DataFrame,
        pitching: pd.DataFrame,
        features: pd.DataFrame
    ):
        self.team_id = team_id
        self.hitting = hitting
        self.pitching = pitching
        self.features = features
    
    def __repr__(self) -> str:
        return (
            f"TeamData(team_id={self.team_id}, "
            f"hitting={len(self.hitting)}, "
            f"pitching={len(self.pitching)}, "
            f"features={len(self.features)})"
        )


class DataLoader:
    """Cargador de datos para equipos MLB."""
    
    def __init__(self):
        self.api_client = MLBClient()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self._cache: Dict[int, TeamData] = {}
    
    def load_team(self, team_id: int, force_refresh: bool = False) -> Optional[TeamData]:
        """
        Carga datos completos de un equipo.
        
        Args:
            team_id: ID del equipo
            force_refresh: Forzar recarga desde API
        
        Returns:
            TeamData o None si falla
        """
        # Usar cache si está disponible
        if not force_refresh and team_id in self._cache:
            logger.debug(f"Usando cache para equipo {team_id}")
            return self._cache[team_id]
        
        try:
            # Cargar datos raw
            hitting_raw = self.api_client.get_team_stats(team_id, "hitting")
            pitching_raw = self.api_client.get_team_stats(team_id, "pitching")
            
            # Normalizar
            hitting = self.preprocessor.normalize_game_log(hitting_raw, "hitting")
            pitching = self.preprocessor.normalize_game_log(pitching_raw, "pitching")
            
            # Generar características
            features = self.feature_engineer.create_team_features(hitting, pitching)
            
            # Crear objeto TeamData
            team_data = TeamData(
                team_id=team_id,
                hitting=hitting,
                pitching=pitching,
                features=features
            )
            
            # Guardar en cache
            self._cache[team_id] = team_data
            
            logger.info(f"Equipo {team_id} cargado correctamente")
            return team_data
            
        except Exception as e:
            logger.error(f"Error cargando equipo {team_id}: {e}", exc_info=True)
            return None
    
    def load_teams(
        self,
        team_ids: list[int],
        force_refresh: bool = False
    ) -> Dict[int, TeamData]:
        """
        Carga múltiples equipos.
        
        Args:
            team_ids: Lista de IDs de equipos
            force_refresh: Forzar recarga
        
        Returns:
            Diccionario de TeamData por ID
        """
        teams = {}
        
        for team_id in team_ids:
            team_data = self.load_team(team_id, force_refresh)
            if team_data:
                teams[team_id] = team_data
        
        logger.info(f"Cargados {len(teams)}/{len(team_ids)} equipos")
        return teams
    
    def load_teams_from_schedule(
        self,
        schedule: pd.DataFrame,
        force_refresh: bool = False
    ) -> Dict[int, TeamData]:
        """
        Carga equipos desde un calendario.
        
        Args:
            schedule: DataFrame con calendario
            force_refresh: Forzar recarga
        
        Returns:
            Diccionario de TeamData
        """
        team_ids = pd.unique(
            schedule[["hom_team_id", "away_team_id"]].values.ravel()
        )
        
        # Filtrar valores nulos
        team_ids = [int(tid) for tid in team_ids if pd.notna(tid)]
        
        return self.load_teams(team_ids, force_refresh)
    
    def clear_cache(self):
        """Limpia el cache de equipos."""
        self._cache.clear()
        logger.info("Cache de equipos limpiado")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas del cache."""
        return {
            "cached_teams": len(self._cache),
            "team_ids": list(self._cache.keys())
        }

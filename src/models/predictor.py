"""
Prediction Engine
=================
Motor de predicción con gestión de modelos y cache.
"""

import os
from typing import Dict, List, Optional

import joblib
import pandas as pd

from src.config import config
from src.data.loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.api.mlb_client import MLBClient
from src.api.odds_client import OddsClient
from src.utils.logger import logger


class PredictionResult:
    """Contenedor de resultado de predicción."""
    
    def __init__(self, game_data: Dict):
        self.fecha = game_data.get("fecha")
        self.matchup = game_data.get("matchup")
        self.hom_team = game_data.get("hom_team")
        self.away_team = game_data.get("away_team")
        self.hom_pitcher = game_data.get("hom_pitcher", "TBD")
        self.away_pitcher = game_data.get("away_pitcher", "TBD")
        self.park_factor = game_data.get("park_factor", 1.0)
        self.predictions: Dict[str, float] = {}
        self.odds: Dict[str, float] = {}
        self.ev_analysis: Dict[str, any] = {}
    
    def add_prediction(self, target: str, value: float):
        """Agrega una predicción."""
        self.predictions[target] = round(value, 2)
    
    def add_odds(self, odds_data: Dict[str, float]):
        """Agrega información de cuotas."""
        self.odds.update(odds_data)
    
    def calculate_ev(self):
        """Calcula Expected Value."""
        if "ml_local_%" not in self.predictions or "hom_ml_odds" not in self.odds:
            self.ev_analysis = {
                "pick": "Sin datos de cuotas",
                "stake": "0%"
            }
            return
        
        prob_h = self.predictions["ml_local_%"] / 100.0
        prob_a = self.predictions.get("ml_visitante_%", 100 - self.predictions["ml_local_%"]) / 100.0
        
        h_odds = self.odds.get("hom_ml_odds", 0.0)
        a_odds = self.odds.get("away_ml_odds", 0.0)
        
        ev_h = (prob_h * h_odds - 1) * 100 if h_odds > 1.0 else -999.0
        ev_a = (prob_a * a_odds - 1) * 100 if a_odds > 1.0 else -999.0
        
        threshold = 3.0  # Umbral de EV+ mínimo
        
        if ev_h > threshold:
            self.ev_analysis = {
                "pick": f"{self.hom_team} ML",
                "ev_pct": f"+{ev_h:.1f}%",
                "stake": "1.5% Bankroll",
                "confidence": "Alta" if ev_h > 8.0 else "Media"
            }
        elif ev_a > threshold:
            self.ev_analysis = {
                "pick": f"{self.away_team} ML",
                "ev_pct": f"+{ev_a:.1f}%",
                "stake": "1.5% Bankroll",
                "confidence": "Alta" if ev_a > 8.0 else "Media"
            }
        else:
            self.ev_analysis = {
                "pick": "Sin Valor",
                "ev_pct": "—",
                "stake": "0%",
                "confidence": "Ninguna"
            }
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            "fecha": self.fecha,
            "matchup": self.matchup,
            "hom_team": self.hom_team,
            "away_team": self.away_team,
            "hom_pitcher": self.hom_pitcher,
            "away_pitcher": self.away_pitcher,
            "park_factor": self.park_factor,
            **self.predictions,
            **{f"cuota_{k}": v for k, v in self.odds.items()},
            **{f"ev_{k}": v for k, v in self.ev_analysis.items()},
        }


class MLBPredictor:
    """Motor de predicción para mercados MLB."""
    
    def __init__(self):
        self.mlb_client = MLBClient()
        self.odds_client = OddsClient()
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.models: Dict[str, Dict] = {}
        self._load_models()
    
    def _load_models(self):
        """Carga modelos desde disco."""
        if not os.path.exists(config.data.model_dir):
            logger.warning(f"Directorio de modelos no existe: {config.data.model_dir}")
            return
        
        for target in config.features.targets:
            model_path = os.path.join(config.data.model_dir, f"{target}.joblib")
            
            if os.path.exists(model_path):
                try:
                    self.models[target] = joblib.load(model_path)
                    logger.debug(f"Modelo cargado: {target}")
                except Exception as e:
                    logger.error(f"Error cargando modelo {target}: {e}")
        
        logger.info(f"Modelos cargados: {len(self.models)}/{len(config.features.targets)}")
    
    def predict_game(
        self,
        game_data: Dict,
        teams_data: Dict
    ) -> Optional[PredictionResult]:
        """
        Predice un juego individual.
        
        Args:
            game_data: Datos del juego
            teams_data: Diccionario de TeamData
        
        Returns:
            PredictionResult o None si falla
        """
        try:
            game_date = pd.Timestamp(game_data["game_date"])
            home_id = int(game_data["hom_team_id"])
            away_id = int(game_data["away_team_id"])
            
            home_team = teams_data.get(home_id)
            away_team = teams_data.get(away_id)
            
            if not home_team or not away_team:
                logger.warning(f"Datos faltantes para juego: {game_data.get('matchup')}")
                return None
            
            # Obtener características
            home_features = self.feature_engineer.get_latest_features(
                home_team.features,
                game_date
            )
            away_features = self.feature_engineer.get_latest_features(
                away_team.features,
                game_date
            )
            
            if not home_features or not away_features:
                logger.warning(f"Features incompletas para: {game_data.get('matchup')}")
                return None
            
            # Calcular descanso
            home_rest = self.feature_engineer.calculate_rest_days(
                home_team.hitting,
                game_date
            )
            away_rest = self.feature_engineer.calculate_rest_days(
                away_team.hitting,
                game_date
            )
            
            # Factor de parque
            park_factor = config.park_factors.get_factor(game_data["hom_team_name"])
            
            # Crear features del matchup
            features = self.feature_engineer.create_matchup_features(
                home_features,
                away_features,
                park_factor,
                home_rest,
                away_rest
            )
            
            # Crear resultado
            result = PredictionResult({
                "fecha": game_data["game_date"],
                "matchup": f"{game_data['away_team_name']} @ {game_data['hom_team_name']}",
                "hom_team": game_data["hom_team_name"],
                "away_team": game_data["away_team_name"],
                "hom_pitcher": game_data.get("hom_pitcher", "TBD"),
                "away_pitcher": game_data.get("away_pitcher", "TBD"),
                "park_factor": park_factor,
            })
            
            # Realizar predicciones
            for target, model_data in self.models.items():
                try:
                    X_row = pd.DataFrame([features]).reindex(
                        columns=model_data["features"],
                        fill_value=0.0
                    )
                    
                    model = model_data["modelo"]
                    
                    if target == "hom_win":
                        proba = model.predict_proba(X_row)[0]
                        prob = proba[list(model.classes_).index(1)] if 1 in model.classes_ else 0.5
                        result.add_prediction("ml_local_%", prob * 100)
                        result.add_prediction("ml_visitante_%", (1 - prob) * 100)
                    else:
                        value = float(model.predict(X_row)[0])
                        
                        # Aplicar park factor a métricas ofensivas
                        offensive_metrics = ["runs", "hits", "homeRuns", "totalBases"]
                        if any(metric in target for metric in offensive_metrics):
                            value = value * park_factor
                        
                        result.add_prediction(target, value)
                
                except Exception as e:
                    logger.error(f"Error prediciendo {target}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}", exc_info=True)
            return None
    
    def predict_date(self, date: Optional[str] = None) -> List[PredictionResult]:
        """
        Predice todos los juegos de una fecha.
        
        Args:
            date: Fecha en formato YYYY-MM-DD (default: hoy)
        
        Returns:
            Lista de PredictionResult
        """
        import datetime
        date = date or datetime.date.today().strftime("%Y-%m-%d")
        
        logger.info(f"Generando predicciones para: {date}")
        
        # Obtener calendario
        schedule = self.mlb_client.get_schedule(date=date)
        
        if schedule.empty:
            logger.warning(f"No hay juegos programados para {date}")
            return []
        
        # Filtrar juegos pendientes
        upcoming = schedule[
            schedule["status"].isin(["Scheduled", "Preview", "Pre-Game"])
        ].copy()
        
        if upcoming.empty:
            logger.info(f"No hay juegos pendientes para {date}")
            return []
        
        logger.info(f"Juegos pendientes: {len(upcoming)}")
        
        # Cargar datos de equipos
        teams_data = self.data_loader.load_teams_from_schedule(upcoming)
        
        # Obtener cuotas
        odds_data = self.odds_client.get_odds(date)
        
        # Generar predicciones
        results = []
        
        for _, game in upcoming.iterrows():
            result = self.predict_game(game.to_dict(), teams_data)
            
            if result:
                # Agregar cuotas si están disponibles
                match_key = result.matchup
                if match_key in odds_data:
                    result.add_odds(odds_data[match_key])
                    result.calculate_ev()
                
                results.append(result)
        
        logger.info(f"Predicciones generadas: {len(results)}")
        return results

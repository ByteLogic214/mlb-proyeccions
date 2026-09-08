"""
Model Trainer
=============
Entrenamiento de modelos ML con validación.
"""

import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report
)

from src.config import config
from src.data.loader import DataLoader
from src.data.feature_engineering import FeatureEngineer, TargetEngineer
from src.utils.logger import logger


class ModelTrainer:
    """Entrenador de modelos para mercados MLB."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.target_engineer = TargetEngineer()
        self.model_dir = config.data.model_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    def build_training_dataset(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Construye dataset de entrenamiento desde calendario.
        
        Args:
            schedule: DataFrame con juegos finalizados
        
        Returns:
            DataFrame con features y targets
        """
        # Filtrar solo juegos finales con scores
        finished = schedule[schedule["status"] == "Final"].copy()
        finished = finished.dropna(subset=["hom_score", "away_score"])
        
        if finished.empty:
            logger.warning("No hay juegos finalizados para entrenar")
            return pd.DataFrame()
        
        logger.info(f"Construyendo dataset con {len(finished)} juegos")
        
        # Cargar datos de todos los equipos
        teams = self.data_loader.load_teams_from_schedule(finished)
        
        rows = []
        
        for idx, game in finished.iterrows():
            game_date = pd.Timestamp(game["game_date"])
            home_id = int(game["hom_team_id"])
            away_id = int(game["away_team_id"])
            
            home_team = teams.get(home_id)
            away_team = teams.get(away_id)
            
            if not home_team or not away_team:
                continue
            
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
                continue
            
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
            park_factor = config.park_factors.get_factor(game["hom_team_name"])
            
            # Crear features del matchup
            features = self.feature_engineer.create_matchup_features(
                home_features,
                away_features,
                park_factor,
                home_rest,
                away_rest
            )
            
            features["game_date"] = game_date
            
            # Crear targets
            targets = self.target_engineer.create_targets({
                "hom_score": game["hom_score"],
                "away_score": game["away_score"],
            })
            
            # Combinar
            row = {**features, **targets}
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            logger.warning("Dataset de entrenamiento vacío")
            return df
        
        logger.info(f"Dataset construido: {len(df)} muestras, {len(df.columns)} columnas")
        return df
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_name: str,
        is_classification: bool = False
    ) -> Tuple[object, Dict[str, float]]:
        """
        Entrena un modelo individual.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            target_name: Nombre del target
            is_classification: Si es clasificación o regresión
        
        Returns:
            Tupla (modelo, métricas)
        """
        logger.info(f"Entrenando modelo para: {target_name}")
        
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=config.model.rf_n_estimators,
                max_depth=config.model.rf_max_depth_classifier,
                min_samples_leaf=config.model.rf_min_samples_leaf_classifier,
                random_state=config.model.random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=config.model.rf_n_estimators,
                max_depth=config.model.rf_max_depth_regressor,
                min_samples_leaf=config.model.rf_min_samples_leaf_regressor,
                random_state=config.model.random_state,
                n_jobs=-1
            )
        
        model.fit(X_train, y_train)
        
        # Calcular métricas en train
        y_pred = model.predict(X_train)
        
        if is_classification:
            metrics = {
                "accuracy": accuracy_score(y_train, y_pred),
                "n_samples": len(X_train)
            }
        else:
            metrics = {
                "mse": mean_squared_error(y_train, y_pred),
                "mae": mean_absolute_error(y_train, y_pred),
                "r2": r2_score(y_train, y_pred),
                "n_samples": len(X_train)
            }
        
        logger.info(f"Modelo {target_name} entrenado: {metrics}")
        return model, metrics
    
    def train_all_models(
        self,
        schedule: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Entrena todos los modelos para los mercados objetivo.
        
        Args:
            schedule: DataFrame con calendario
        
        Returns:
            Diccionario con modelos y métricas
        """
        # Construir dataset
        df = self.build_training_dataset(schedule)
        
        if len(df) < config.model.min_training_samples:
            logger.error(
                f"Insuficientes muestras para entrenar: "
                f"{len(df)} < {config.model.min_training_samples}"
            )
            return {}
        
        # Ordenar cronológicamente
        df = df.sort_values("game_date").reset_index(drop=True)
        
        # Separar features y targets
        feature_cols = [
            c for c in df.columns
            if c not in config.features.targets and c != "game_date"
        ]
        
        X = df[feature_cols].fillna(0.0)
        
        # Split temporal (80/20)
        split_idx = int(len(df) * 0.8)
        X_train = X.iloc[:split_idx]
        
        models = {}
        
        for target in config.features.targets:
            if target not in df.columns:
                logger.warning(f"Target {target} no encontrado en dataset")
                continue
            
            is_classification = (target == "hom_win")
            
            y = df[target]
            if is_classification:
                y = y.astype(int)
            else:
                y = y.astype(float)
            
            y_train = y.iloc[:split_idx]
            
            try:
                model, metrics = self.train_model(
                    X_train,
                    y_train,
                    target,
                    is_classification
                )
                
                # Guardar modelo
                model_path = os.path.join(self.model_dir, f"{target}.joblib")
                joblib.dump(
                    {
                        "modelo": model,
                        "features": feature_cols,
                        "metrics": metrics,
                        "target": target
                    },
                    model_path
                )
                
                models[target] = {
                    "model": model,
                    "features": feature_cols,
                    "metrics": metrics,
                    "path": model_path
                }
                
                logger.info(f"✓ Modelo {target} guardado en {model_path}")
                
            except Exception as e:
                logger.error(f"Error entrenando {target}: {e}", exc_info=True)
        
        logger.info(f"Entrenamiento completado: {len(models)} modelos")
        return models

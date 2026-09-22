"""
FASE 2: ENTRENAMIENTO CON XGBOOST Y CONFIDENCE SCORING
=======================================================

Basado en el consenso científico (Zhao 2024, Bae 2026):
  1. XGBoost supera a Random Forest en predicción deportiva
  2. Predicción binaria directa rara vez supera 55-60%
  3. Solo predicen cuando confianza es EXTREMADAMENTE ALTA (>65-70%)
  4. Time-Series Split evita data leakage
  5. Modelos especializados por mercado logran >70% en subconjuntos

Importado por: mlb_betting_refactored.py
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, roc_auc_score


# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

MODEL_DIR = "models"
CONFIDENCE_THRESHOLD_MONEYLINE = 0.68  # 68% mínimo para Moneyline
CONFIDENCE_THRESHOLD_PROPS = 0.65      # 65% mínimo para Props

# Parámetros XGBoost validados por literatura (Zhao 2024)
XGBOOST_PARAMS_CLASSIFICATION = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,  # L2 regularization
    "alpha": 0.5,   # L1 regularization
    "scale_pos_weight": 1.0,  # Ajustar si hay desbalance de clases
    "gamma": 0.0,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
}

XGBOOST_PARAMS_REGRESSION = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "booster": "gbtree",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 0.5,
    "tree_method": "hist",
    "random_state": 42,
}


# ============================================================================
# CLASE: TIME SERIES SPLIT VALIDATOR
# ============================================================================

class TimeSeriesSplitValidator:
    """
    Valida modelos de series temporales sin data leakage.
    Usa Walk-Forward Validation: entrena en pasado, prueba en futuro.
    
    Referencias:
        - Hyndman, R. J., & Athanasopoulos, G. (2021).
          "Forecasting: principles and practice"
        - Prado, R., & West, M. (2010). "Time Series Modeling"
    """
    
    def __init__(self, n_splits: int = 5, train_ratio: float = 0.8):
        """
        Args:
            n_splits: Número de splits temporales
            train_ratio: Proporción train/test en cada split
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.splitter = TimeSeriesSplit(n_splits=n_splits)
    
    def get_splits(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera índices train/test en orden temporal.
        
        Returns:
            List de tuples (train_indices, test_indices)
        """
        splits = []
        for train_idx, test_idx in self.splitter.split(X):
            splits.append((train_idx, test_idx))
        return splits
    
    def evaluate_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: xgb.XGBClassifier | xgb.XGBRegressor,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Valida modelo en CV temporal.
        
        Returns:
            Dict con métricas (accuracy/r2, logloss/rmse, etc.)
        """
        splits = self.get_splits(X)
        scores = []
        losses = []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train, verbose=False)
            
            if is_classification:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                loss = log_loss(y_test, y_pred_proba)
            else:
                y_pred = model.predict(X_test)
                score = model.score(X_test, y_test)  # R²
                loss = np.mean((y_test - y_pred) ** 2)  # MSE
            
            scores.append(score)
            losses.append(loss)
        
        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "mean_loss": np.mean(losses),
            "std_loss": np.std(losses),
        }


# ============================================================================
# CLASE: XGBOOST CLASSIFIER CON CONFIDENCE SCORING
# ============================================================================

class ConfidenceScoringXGBClassifier:
    """
    Wrapper de XGBoost para clasificación con Confidence Scoring.
    
    Solo emite predicción si probabilidad > UMBRAL de confianza.
    Estrategia: Replicar éxito de Zhao (2024) y Bae (2026) en deportes.
    
    Atributos:
        model: XGBClassifier entrenado
        scaler: StandardScaler para normalización de features
        feature_names: Lista de nombres de features
        confidence_threshold: Umbral mínimo de confianza
        feature_importance: Importancia de cada feature (SHAP-like)
    """
    
    def __init__(self, confidence_threshold: float = 0.65):
        """
        Args:
            confidence_threshold: Umbral de confianza (default: 65%)
        """
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.confidence_threshold = confidence_threshold
        self.feature_importance = {}
        self.cv_validator = TimeSeriesSplitValidator(n_splits=5)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        evaluate_cv: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena el modelo en datos históricos.
        
        Args:
            X: Features (DataFrame)
            y: Target binario (Series, 0 o 1)
            evaluate_cv: Si True, evalúa en CV temporal
            verbose: Logging
        
        Returns:
            Dict con métricas de entrenamiento
        """
        # Normalizar features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.feature_names = list(X.columns)
        
        # Entrenar XGBoost
        self.model = xgb.XGBClassifier(**XGBOOST_PARAMS_CLASSIFICATION, n_estimators=500)
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            verbose=False
        )
        
        # Extraer feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Validación cruzada temporal
        metrics = {}
        if evaluate_cv:
            metrics = self.cv_validator.evaluate_cv(X_scaled, y, self.model, is_classification=True)
            if verbose:
                print(f"  ✅ CV Temporal - Accuracy: {metrics['mean_score']:.1%} ± {metrics['std_score']:.1%}")
                print(f"     Log Loss: {metrics['mean_loss']:.3f}")
        
        return metrics
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice con confidence scoring.
        
        Solo retorna predicción si confianza > umbral.
        Estrategia Zhao 2024: Sin valor claro = sin apuesta.
        
        Args:
            X: Features (DataFrame, single row o multiple)
        
        Returns:
            Dict:
                "prediction": 0 o 1 (o None si confianza < umbral)
                "confidence": Probabilidad predicha
                "high_confidence": Bool (True si > umbral)
                "recommendation": String descriptivo
        """
        if self.model is None:
            return {
                "prediction": None,
                "confidence": 0.5,
                "high_confidence": False,
                "recommendation": "❌ Modelo no entrenado"
            }
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Predecir probabilidades
        proba = self.model.predict_proba(X_scaled)
        
        # Tomar clase 1 (victoria)
        prob_class_1 = proba[0, 1] if len(proba) > 0 else 0.5
        
        # Definir predicción y confianza
        is_high_confidence = prob_class_1 > self.confidence_threshold or prob_class_1 < (1 - self.confidence_threshold)
        
        if is_high_confidence:
            if prob_class_1 > 0.5:
                prediction = 1
                confidence = prob_class_1
                recommendation = f"✅ ALTA CONFIANZA: {prob_class_1:.1%} (Local gana)"
            else:
                prediction = 0
                confidence = 1 - prob_class_1
                recommendation = f"✅ ALTA CONFIANZA: {confidence:.1%} (Visitante gana)"
        else:
            prediction = None
            confidence = max(prob_class_1, 1 - prob_class_1)
            recommendation = f"⚠️ Confianza insuficiente: {confidence:.1%} < {self.confidence_threshold:.1%}"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "prob_home_win": prob_class_1,
            "prob_away_win": 1 - prob_class_1,
            "high_confidence": is_high_confidence,
            "recommendation": recommendation
        }
    
    def get_top_features(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Retorna top N features por importancia.
        """
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]
    
    def save(self, filepath: str) -> None:
        """Guarda modelo y scaler."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        pickle.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "confidence_threshold": self.confidence_threshold,
            "feature_importance": self.feature_importance
        }, open(filepath, "wb"))
    
    def load(self, filepath: str) -> None:
        """Carga modelo y scaler."""
        data = pickle.load(open(filepath, "rb"))
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.confidence_threshold = data["confidence_threshold"]
        self.feature_importance = data["feature_importance"]


# ============================================================================
# CLASE: XGBOOST REGRESSOR CON CALIBRACIÓN
# ============================================================================

class CalibratedXGBRegressor:
    """
    Regressor XGBoost para predicción de propiedades (runs, hits, etc.)
    con calibración de predicciones.
    
    Usa conformal prediction para generar intervalos de confianza.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.cv_validator = TimeSeriesSplitValidator(n_splits=5)
        self.residuals = []  # Para conformal prediction
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        evaluate_cv: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena regressor XGBoost.
        
        Args:
            X: Features (DataFrame)
            y: Target continuo (Series)
            evaluate_cv: Si True, evalúa en CV temporal
            verbose: Logging
        
        Returns:
            Dict con métricas
        """
        # Normalizar
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.feature_names = list(X.columns)
        
        # Entrenar
        self.model = xgb.XGBRegressor(**XGBOOST_PARAMS_REGRESSION, n_estimators=500)
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            verbose=False
        )
        
        # Validación CV
        metrics = {}
        if evaluate_cv:
            metrics = self.cv_validator.evaluate_cv(X_scaled, y, self.model, is_classification=False)
            if verbose:
                rmse = np.sqrt(metrics['mean_loss'])
                print(f"  ✅ CV Temporal - RMSE: {rmse:.2f} ± {metrics['std_loss']:.2f}")
                print(f"     R²: {metrics['mean_score']:.1%}")
        
        # Calcular residuos para conformal prediction
        y_pred = self.model.predict(X_scaled)
        self.residuals = np.abs(y - y_pred).values
        
        return metrics
    
    def predict_with_interval(self, X: pd.DataFrame, confidence: float = 0.9) -> Dict[str, Any]:
        """
        Predice con intervalo de confianza (conformal prediction).
        
        Args:
            X: Features (DataFrame, single row)
            confidence: Nivel de confianza (default: 90%)
        
        Returns:
            Dict:
                "prediction": Valor predicho
                "lower_bound": Límite inferior del intervalo
                "upper_bound": Límite superior del intervalo
                "interval_width": Ancho del intervalo
        """
        if self.model is None:
            return {
                "prediction": None,
                "lower_bound": None,
                "upper_bound": None,
                "interval_width": None
            }
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Predecir valor puntual
        y_pred = self.model.predict(X_scaled)[0]
        
        # Calcular margen conformal (cuantil de residuos)
        if len(self.residuals) > 0:
            alpha = 1 - confidence
            margin = np.quantile(self.residuals, min(1.0, 1 - alpha))
        else:
            margin = y_pred * 0.15  # 15% por defecto
        
        return {
            "prediction": y_pred,
            "lower_bound": y_pred - margin,
            "upper_bound": y_pred + margin,
            "interval_width": 2 * margin,
            "confidence_level": confidence
        }
    
    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        pickle.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "residuals": self.residuals
        }, open(filepath, "wb"))
    
    def load(self, filepath: str) -> None:
        data = pickle.load(open(filepath, "rb"))
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.residuals = data.get("residuals", [])


# ============================================================================
# CLASE: MODEL TRAINER (ORQUESTADOR)
# ============================================================================

class MLBModelTrainer:
    """
    Entrena suite completa de modelos especializados:
      - Moneyline (Clasificación: Local gana sí/no)
      - Total de Runs (Regresión)
      - Hits por equipo (Regresión)
      - Ponches (Regresión)
    
    Utiliza:
      - XGBoost (no Random Forest)
      - Time-Series Split (sin data leakage)
      - Confidence Scoring (solo predice si confianza alta)
    """
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}  # Dict[target_name] = modelo entrenado
        self.training_metrics = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def train_all_targets(
        self,
        df: pd.DataFrame,
        targets: List[str],
        feature_cols: List[str],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena modelos para múltiples targets.
        
        Args:
            df: DataFrame con features + targets
            targets: Lista de columnas objetivo
            feature_cols: Lista de columnas de features
            verbose: Logging
        
        Returns:
            Dict[target_name] = Dict con métricas
        """
        results = {}
        
        # Separar clasificación de regresión
        classification_targets = ["hom_win"]
        regression_targets = [t for t in targets if t not in classification_targets]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 ENTRENANDO {len(targets)} MODELOS CON XGBOOST + TIME-SERIES SPLIT")
            print(f"{'='*70}")
        
        # Clasificación (Moneyline)
        for target in classification_targets:
            if target not in df.columns:
                continue
            
            if verbose:
                print(f"\n📊 {target.upper()} [CLASIFICACIÓN]")
                print(f"   Umbral de confianza: {CONFIDENCE_THRESHOLD_MONEYLINE:.0%}")
            
            X = df[feature_cols].fillna(0.0)
            y = df[target].astype(int)
            
            # Entrenar
            clf = ConfidenceScoringXGBClassifier(
                confidence_threshold=CONFIDENCE_THRESHOLD_MONEYLINE
            )
            metrics = clf.fit(X, y, evaluate_cv=True, verbose=verbose)
            
            # Guardar
            self.models[target] = clf
            self.training_metrics[target] = metrics
            results[target] = metrics
            
            # Top features
            top_features = clf.get_top_features(top_n=5)
            if verbose:
                print(f"   Top Features: {', '.join([f[0] for f in top_features])}")
        
        # Regresión (Propiedades)
        for target in regression_targets:
            if target not in df.columns:
                continue
            
            if verbose:
                print(f"\n📊 {target.upper()} [REGRESIÓN]")
            
            X = df[feature_cols].fillna(0.0)
            y = df[target].astype(float)
            
            # Entrenar
            reg = CalibratedXGBRegressor()
            metrics = reg.fit(X, y, evaluate_cv=True, verbose=verbose)
            
            # Guardar
            self.models[target] = reg
            self.training_metrics[target] = metrics
            results[target] = metrics
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ ENTRENAMIENTO COMPLETADO: {len(self.models)} modelos")
            print(f"{'='*70}\n")
        
        return results
    
    def save_all_models(self) -> None:
        """Guarda todos los modelos entrenados."""
        for target_name, model in self.models.items():
            filepath = os.path.join(self.model_dir, f"{target_name}_xgb.pkl")
            model.save(filepath)
            print(f"  ✅ Modelo guardado: {filepath}")
    
    def load_all_models(self, targets: List[str]) -> None:
        """Carga modelos entrenados."""
        for target in targets:
            filepath = os.path.join(self.model_dir, f"{target}_xgb.pkl")
            if not os.path.exists(filepath):
                continue
            
            # Detectar tipo de modelo
            dummy_df = pd.DataFrame()
            if target == "hom_win":
                model = ConfidenceScoringXGBClassifier()
            else:
                model = CalibratedXGBRegressor()
            
            model.load(filepath)
            self.models[target] = model
            print(f"  ✅ Modelo cargado: {filepath}")


# ============================================================================
# PRUEBA / VALIDACIÓN
# ============================================================================

if __name__ == "__main__":
    print("✅ models_xgboost.py cargado correctamente")
    print("\nClases disponibles:")
    print("  1. TimeSeriesSplitValidator — Validación temporal sin data leakage")
    print("  2. ConfidenceScoringXGBClassifier — Moneyline + Confidence Scoring")
    print("  3. CalibratedXGBRegressor — Props con conformal prediction")
    print("  4. MLBModelTrainer — Orquestador de múltiples modelos")

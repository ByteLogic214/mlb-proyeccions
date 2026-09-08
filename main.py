#!/usr/bin/env python3
"""
MLB Betting System - Main Entry Point
======================================
Sistema profesional de predicción de mercados MLB.

Uso:
    python main.py [YYYY-MM-DD]
"""

import sys
from datetime import date

from src.config import config
from src.models.trainer import ModelTrainer
from src.models.predictor import MLBPredictor
from src.reporting.generator import ReportGenerator
from src.utils.logger import logger


def main():
    """Punto de entrada principal."""
    logger.info("=" * 70)
    logger.info("🚀 MLB BETTING SYSTEM - Inicio")
    logger.info("=" * 70)
    
    # Validar configuración
    if not config.validate():
        logger.error("Configuración inválida. Abortando.")
        sys.exit(1)
    
    # Fecha objetivo
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m-%d")
    logger.info(f"Fecha objetivo: {target_date}")
    
    try:
        # 1️⃣ Entrenar modelos
        logger.info("\n📚 FASE 1: Entrenamiento de Modelos")
        logger.info("-" * 70)
        
        trainer = ModelTrainer()
        
        # Obtener calendario histórico
        schedule = trainer.mlb_client.get_schedule()
        
        if schedule.empty:
            logger.error("No se pudo obtener el calendario. Abortando.")
            sys.exit(1)
        
        models = trainer.train_all_models(schedule)
        
        if not models:
            logger.error("No se entrenó ningún modelo. Abortando.")
            sys.exit(1)
        
        logger.info(f"✅ Modelos entrenados: {len(models)}")
        
        # 2️⃣ Generar predicciones
        logger.info("\n🔮 FASE 2: Generación de Predicciones")
        logger.info("-" * 70)
        
        predictor = MLBPredictor()
        predictions = predictor.predict_date(target_date)
        
        if not predictions:
            logger.warning("No hay predicciones para generar.")
            sys.exit(0)
        
        logger.info(f"✅ Predicciones generadas: {len(predictions)}")
        
        # 3️⃣ Generar reportes
        logger.info("\n📊 FASE 3: Generación de Reportes")
        logger.info("-" * 70)
        
        generator = ReportGenerator()
        reports = generator.generate_all(predictions, target_date)
        
        logger.info("✅ Reportes generados:")
        for format_type, path in reports.items():
            logger.info(f"  • {format_type.upper()}: {path}")
        
        # Resumen final
        logger.info("\n" + "=" * 70)
        logger.info("✅ PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)
        logger.info(f"Juegos analizados: {len(predictions)}")
        logger.info(f"Picks EV+: {sum(1 for p in predictions if 'Sin Valor' not in p.ev_analysis.get('pick', ''))}")
        logger.info(f"Archivos generados: {len(reports)}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Proceso interrumpido por el usuario")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ Error crítico: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

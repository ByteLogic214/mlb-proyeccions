"""
Logging Utilities
=================
Sistema de logging profesional con rotación y niveles.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter con colores para consola."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configura un logger con handlers para consola y archivo.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging
        log_file: Ruta del archivo de log
        max_bytes: Tamaño máximo del archivo antes de rotar
        backup_count: Número de archivos de respaldo
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Formato
    fmt = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    
    # Handler consola con colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(fmt, date_fmt))
    logger.addHandler(console_handler)
    
    # Handler archivo con rotación
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, date_fmt))
        logger.addHandler(file_handler)
    
    return logger


# Logger global
logger = setup_logger('mlb_system', log_file='logs/mlb_system.log')

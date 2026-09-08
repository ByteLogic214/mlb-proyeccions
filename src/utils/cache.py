"""
Cache Management
================
Sistema de cache con TTL y gestión de archivos.
"""

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import logger


class CacheManager:
    """Gestor de cache con TTL para DataFrames."""
    
    def __init__(self, cache_dir: str, ttl: int = 3600):
        """
        Inicializa el gestor de cache.
        
        Args:
            cache_dir: Directorio de cache
            ttl: Tiempo de vida en segundos
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos del cache si están vigentes.
        
        Args:
            key: Clave del cache
        
        Returns:
            DataFrame si existe y está vigente, None en caso contrario
        """
        cache_path = self.cache_dir / f"{key}.csv"
        
        if not cache_path.exists():
            logger.debug(f"Cache miss: {key}")
            return None
        
        # Verificar TTL
        age = time.time() - cache_path.stat().st_mtime
        if age > self.ttl:
            logger.debug(f"Cache expirado: {key} (edad: {age:.0f}s)")
            return None
        
        try:
            df = pd.read_csv(cache_path)
            logger.debug(f"Cache hit: {key}")
            return df
        except Exception as e:
            logger.warning(f"Error leyendo cache {key}: {e}")
            return None
    
    def set(self, key: str, df: pd.DataFrame) -> bool:
        """
        Guarda datos en cache.
        
        Args:
            key: Clave del cache
            df: DataFrame a guardar
        
        Returns:
            True si se guardó correctamente
        """
        cache_path = self.cache_dir / f"{key}.csv"
        
        try:
            df.to_csv(cache_path, index=False)
            logger.debug(f"Cache guardado: {key}")
            return True
        except Exception as e:
            logger.error(f"Error guardando cache {key}: {e}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Limpia el cache.
        
        Args:
            pattern: Patrón de archivos a eliminar (ej: "schedule_*")
        
        Returns:
            Número de archivos eliminados
        """
        count = 0
        glob_pattern = pattern or "*"
        
        for file_path in self.cache_dir.glob(f"{glob_pattern}.csv"):
            try:
                file_path.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error eliminando {file_path}: {e}")
        
        logger.info(f"Cache limpiado: {count} archivos eliminados")
        return count
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas del cache."""
        files = list(self.cache_dir.glob("*.csv"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }

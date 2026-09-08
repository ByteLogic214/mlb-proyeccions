"""
Data Validators
===============
Validadores de datos y esquemas.
"""

from typing import List, Optional

import pandas as pd

from src.utils.logger import logger


class DataValidator:
    """Validador de DataFrames y datos."""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List[str],
        min_rows: int = 1,
        name: str = "DataFrame"
    ) -> bool:
        """
        Valida un DataFrame.
        
        Args:
            df: DataFrame a validar
            required_columns: Columnas requeridas
            min_rows: Número mínimo de filas
            name: Nombre del DataFrame para logging
        
        Returns:
            True si es válido
        """
        if df is None:
            logger.error(f"{name} es None")
            return False
        
        if df.empty:
            logger.error(f"{name} está vacío")
            return False
        
        if len(df) < min_rows:
            logger.error(f"{name} tiene menos de {min_rows} filas")
            return False
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"{name} le faltan columnas: {missing_cols}")
            return False
        
        logger.debug(f"{name} validado correctamente ({len(df)} filas)")
        return True
    
    @staticmethod
    def validate_numeric_column(
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """Valida que una columna sea numérica y esté en rango."""
        if column not in df.columns:
            logger.error(f"Columna {column} no existe")
            return False
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.error(f"Columna {column} no es numérica")
            return False
        
        if min_value is not None and df[column].min() < min_value:
            logger.warning(f"Columna {column} tiene valores < {min_value}")
        
        if max_value is not None and df[column].max() > max_value:
            logger.warning(f"Columna {column} tiene valores > {max_value}")
        
        return True
    
    @staticmethod
    def validate_date_column(df: pd.DataFrame, column: str) -> bool:
        """Valida que una columna sea de tipo fecha."""
        if column not in df.columns:
            logger.error(f"Columna {column} no existe")
            return False
        
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            logger.error(f"Columna {column} no es tipo datetime")
            return False
        
        return True

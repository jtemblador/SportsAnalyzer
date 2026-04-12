# src/nfl/features/v5/__init__.py
from src.nfl.features.v5.config import VERSION
from src.nfl.features.v5.engineer import build_features

__all__ = ['VERSION', 'build_features']

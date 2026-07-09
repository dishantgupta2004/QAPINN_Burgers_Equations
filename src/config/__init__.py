"""Configuration layer: enums + the immutable BurgersConfig dataclass."""

from src.config.config import BurgersConfig
from src.config.enums import ElementType, TimeIntegrator, ICKind, BCType, BCSpec

__all__ = [
    "BurgersConfig",
    "ElementType",
    "TimeIntegrator",
    "ICKind",
    "BCType",
    "BCSpec",
]

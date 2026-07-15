from __future__ import annotations
from enum import Enum
from typing import Tuple, Union


class ElementType(str, Enum):
    LAGRANGE = "Lagrange"         
    P = "Lagrange"              


class TimeIntegrator(str, Enum):
    BACKWARD_EULER = "backward_euler"
    CRANK_NICOLSON = "crank_nicolson"   
    BDF2 = "bdf2"                     


class ICKind(str, Enum):
    SIN = "sin"
    GAUSSIAN = "gaussian"
    SQUARE = "square"
    SHOCK = "shock"
    RANDOM_SMOOTH = "random_smooth"
    CUSTOM = "custom"


class BCType(str, Enum):
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
    PERIODIC = "Periodic"
    ROBIN = "Robin"

BCSpec = Tuple[str, Union[float, Tuple[float, float]]]

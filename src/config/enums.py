"""
src/config/enums.py
===================

Closed vocabularies (enumerations) for the configuration surface.

Enums make invalid states unrepresentable and give editors autocompletion.
Strings are still accepted at the config boundary (see :class:`BurgersConfig`)
and normalized against these enums, so the ergonomic ``initial_condition="sin"``
form from the experiment file keeps working.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple, Union


class ElementType(str, Enum):
    """Supported finite-element families (basix names)."""
    LAGRANGE = "Lagrange"          # standard continuous CG element
    P = "Lagrange"                 # alias


class TimeIntegrator(str, Enum):
    """Time-stepping schemes. Only Backward Euler is fully implemented; the
    others are declared so the config surface is forward-compatible."""
    BACKWARD_EULER = "backward_euler"
    CRANK_NICOLSON = "crank_nicolson"   # architecture placeholder
    BDF2 = "bdf2"                       # architecture placeholder


class ICKind(str, Enum):
    """Named initial-condition presets. ``CUSTOM`` defers to a user callable."""
    SIN = "sin"
    GAUSSIAN = "gaussian"
    SQUARE = "square"
    SHOCK = "shock"
    RANDOM_SMOOTH = "random_smooth"
    CUSTOM = "custom"


class BCType(str, Enum):
    """Boundary-condition families. Dirichlet & Neumann are fully implemented;
    Periodic & Robin are wired into the architecture as explicit NotImplemented
    paths so extension points are obvious."""
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
    PERIODIC = "Periodic"
    ROBIN = "Robin"


# A boundary condition specification is a (type, value) tuple, e.g.
#   ("Dirichlet", 0.0)            -> u = 0 on that boundary
#   ("Neumann", 0.0)             -> du/dn = 0 (natural, do-nothing)
#   ("Robin", (alpha, beta))     -> alpha*u + du/dn = beta   (architecture)
BCSpec = Tuple[str, Union[float, Tuple[float, float]]]

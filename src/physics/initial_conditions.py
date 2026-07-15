from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from src.config.config import BurgersConfig
from src.config.enums import ICKind


class InitialConditionFactory:
    """Turn a :class:`BurgersConfig` into a spatial callable ``u0(x)``."""

    def __init__(self, config: BurgersConfig) -> None:
        self.cfg = config

    def build(self) -> Callable[[np.ndarray], np.ndarray]:
        kind = self.cfg.initial_condition
        dispatch: Dict[str, Callable[[], Callable[[np.ndarray], np.ndarray]]] = {
            ICKind.SIN.value: self._sin,
            ICKind.GAUSSIAN.value: self._gaussian,
            ICKind.SQUARE.value: self._square,
            ICKind.SHOCK.value: self._shock,
            ICKind.RANDOM_SMOOTH.value: self._random_smooth,
            ICKind.CUSTOM.value: self._custom,
        }
        if kind not in dispatch:
            raise ValueError(
                f"Unknown initial_condition '{kind}'. "
                f"Valid: {sorted(dispatch)}."
            )
        return dispatch[kind]()

    # -- per-axis helpers -----------------------------------------------------
    def _axis_mins(self) -> List[float]:
        return [lo for lo, _ in self.cfg.axis_bounds()]

    def _axis_spans(self) -> List[float]:
        return [hi - lo for lo, hi in self.cfg.axis_bounds()]

    # -- presets --------------------------------------------------------------
    def _sin(self) -> Callable[[np.ndarray], np.ndarray]:
        """Separable sine: product of one full period along each active axis.

        1D: a·sin(2π (x-xmin)/Lx)
        2D: a·sin(2π (x-xmin)/Lx)·sin(2π (y-ymin)/Ly)     (and so on in 3D)
        """
        a = self.cfg.ic_amplitude
        d = self.cfg.dimension
        mins, spans = self._axis_mins(), self._axis_spans()

        def field(x: np.ndarray) -> np.ndarray:
            out = a * np.ones_like(x[0], dtype=np.float64)
            for i in range(d):
                out = out * np.sin(2.0 * np.pi * (x[i] - mins[i]) / spans[i])
            return out
        return field

    def _gaussian(self) -> Callable[[np.ndarray], np.ndarray]:
        """Radial Gaussian bump centered at ``ic_gaussian_center`` on each axis."""
        a, c, w = (self.cfg.ic_amplitude, self.cfg.ic_gaussian_center,
                   self.cfg.ic_gaussian_width)
        d = self.cfg.dimension

        def field(x: np.ndarray) -> np.ndarray:
            r2 = np.zeros_like(x[0], dtype=np.float64)
            for i in range(d):
                r2 = r2 + (x[i] - c) ** 2
            return a * np.exp(-r2 / (2.0 * w * w))
        return field

    def _square(self) -> Callable[[np.ndarray], np.ndarray]:
        """Axis-aligned top-hat (box); the CG projection regularizes corners."""
        a, lo, hi = (self.cfg.ic_amplitude, self.cfg.ic_square_lo,
                     self.cfg.ic_square_hi)
        d = self.cfg.dimension

        def field(x: np.ndarray) -> np.ndarray:
            mask = np.ones_like(x[0], dtype=bool)
            for i in range(d):
                mask &= (x[i] >= lo) & (x[i] <= hi)
            return a * mask.astype(np.float64)
        return field

    def _shock(self) -> Callable[[np.ndarray], np.ndarray]:
        """Riemann-type step along x: left state +a, right state -a about mid-x."""
        a = self.cfg.ic_amplitude
        mid = 0.5 * (self.cfg.xmin + self.cfg.xmax)
        return lambda x: a * np.where(x[0] < mid, 1.0, -1.0)

    def _random_smooth(self) -> Callable[[np.ndarray], np.ndarray]:
        """Band-limited random field: a sum of low-frequency Fourier modes with
        random phases/amplitudes, summed independently over each active axis.
        Deterministic given ``ic_random_seed``.
        """
        rng = np.random.default_rng(self.cfg.ic_random_seed)
        modes = int(self.cfg.ic_random_modes)
        d = self.cfg.dimension
        mins, spans = self._axis_mins(), self._axis_spans()

        # Independent mode amplitudes/phases per axis (deterministic order).
        amps = [rng.standard_normal(modes) / np.arange(1, modes + 1)
                for _ in range(d)]
        phases = [rng.uniform(0.0, 2.0 * np.pi, size=modes) for _ in range(d)]

        def field(x: np.ndarray) -> np.ndarray:
            out = np.zeros_like(x[0], dtype=np.float64)
            for i in range(d):
                xi = (x[i] - mins[i]) / spans[i]
                for k in range(modes):
                    out += amps[i][k] * np.sin(2.0 * np.pi * (k + 1) * xi
                                                + phases[i][k])
            return self.cfg.ic_amplitude * out
        return field

    def _custom(self) -> Callable[[np.ndarray], np.ndarray]:
        user_fn = self.cfg.ic_function
        assert user_fn is not None  # guaranteed by config validation
        # For 1D we pass the x-row (so simple ``lambda x: np.sin(2*np.pi*x)``
        # forms keep working). For 2D/3D we pass the full ``(gdim, n)`` array so
        # the user can index x[0], x[1], x[2].
        if self.cfg.dimension == 1:
            return lambda x: np.asarray(user_fn(x[0]), dtype=np.float64)
        return lambda x: np.asarray(user_fn(x), dtype=np.float64)

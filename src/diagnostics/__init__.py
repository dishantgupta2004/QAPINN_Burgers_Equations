"""Diagnostics layer: integral quantities and stability guards."""

from src.diagnostics.diagnostics import (
    compute_diagnostics,
    check_stability,
    check_stability_cfg,
)

__all__ = ["compute_diagnostics", "check_stability", "check_stability_cfg"]

from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from . import models
from .common import PHYSICS

def _fft_power(signal: np.ndarray, dxi: float):
    signal = signal - signal.mean()
    n = signal.shape[0]
    fft = np.fft.rfft(signal)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=dxi) 
    total = power.sum() + 1e-30
    return freqs, power / total


def _spectral_summary(freqs: np.ndarray, power: np.ndarray, energy_thresh: float = 0.99) -> Dict:
    cum = np.cumsum(power)
    cum = cum / (cum[-1] + 1e-30)
    k = int(np.searchsorted(cum, energy_thresh))
    k = min(k, len(freqs) - 1)
    centroid = float(np.sum(freqs * power) / (np.sum(power) + 1e-30))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / (np.sum(power) + 1e-30)))
    return {
        "max_frequency": float(freqs[k]),
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "peak_frequency": float(freqs[int(np.argmax(power))]),
    }

def _eval_line(model, axis: str, fixed: float, n: int = 512):
    if axis == "x":
        coord = np.linspace(PHYSICS.xmin, PHYSICS.xmax, n)
        xs, ts = coord, np.full(n, fixed)
    else:
        coord = np.linspace(PHYSICS.tmin, PHYSICS.T, n)
        ts, xs = coord, np.full(n, fixed)
    xn = torch.tensor(PHYSICS.norm_x(xs).reshape(-1, 1), dtype=torch.float32)
    tn = torch.tensor(PHYSICS.norm_t(ts).reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        u = model(xn, tn).cpu().numpy().reshape(-1)
    return coord, u


def _eval_quantum_probs_line(model: models.QAPINN, axis: str, fixed: float, n: int = 512):
    if axis == "x":
        xs = np.linspace(PHYSICS.xmin, PHYSICS.xmax, n)
        ts = np.full(n, fixed)
    else:
        ts = np.linspace(PHYSICS.tmin, PHYSICS.T, n)
        xs = np.full(n, fixed)
    xn = torch.tensor(PHYSICS.norm_x(xs).reshape(-1, 1), dtype=torch.float32)
    tn = torch.tensor(PHYSICS.norm_t(ts).reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        _, feats = model(xn, tn, return_features=True)
    return feats["quantum_probs"].cpu().numpy()  # (n, 2**q)

def fourier_spectrum(
    model,
    label: str,
    axis: str = "x",
    fixed: float = 0.5,
    n: int = 512,
    is_quantum: bool = False,
) -> Dict:
    span = (PHYSICS.xmax - PHYSICS.xmin) if axis == "x" else (PHYSICS.T - PHYSICS.tmin)
    dxi = span / (n - 1)

    coord, u = _eval_line(model, axis, fixed, n)
    freqs, power = _fft_power(u, dxi)
    result = {
        "label": label,
        "axis": axis,
        "fixed": fixed,
        "freqs": freqs.tolist(),
        "output_power": power.tolist(),
        "output_summary": _spectral_summary(freqs, power),
    }

    if is_quantum:
        probs = _eval_quantum_probs_line(model, axis, fixed, n)
        agg = np.zeros_like(freqs)
        for c in range(probs.shape[1]):
            _, p = _fft_power(probs[:, c], dxi)
            agg += p
        agg /= probs.shape[1]
        result["quantum_probs_power"] = agg.tolist()
        result["quantum_probs_summary"] = _spectral_summary(freqs, agg)
    return result


def compare_spectra(
    qubit_range=(3, 4, 5),
    axis: str = "x",
    fixed: float = 0.5,
    n: int = 512,
) -> Dict:
   
    out: Dict[str, Dict] = {}

    try:
        cmodel, _ = models.load_classical()
        out["Classical PINN"] = fourier_spectrum(cmodel, "Classical PINN", axis, fixed, n)
    except Exception as exc:  # pragma: no cover
        out["Classical PINN"] = {"error": str(exc)}

    for q in qubit_range:
        try:
            qmodel, _ = models.load_qapinn(q)
            out[f"QA-PINN {q}q"] = fourier_spectrum(
                qmodel, f"QA-PINN {q}q", axis, fixed, n, is_quantum=True
            )
        except Exception as exc:  # pragma: no cover
            out[f"QA-PINN {q}q"] = {"error": str(exc)}

    table = []
    for label, r in out.items():
        if "error" in r:
            continue
        row = {"model": label, **r["output_summary"]}
        if "quantum_probs_summary" in r:
            row["quantum_max_frequency"] = r["quantum_probs_summary"]["max_frequency"]
        table.append(row)
    out["_summary_table"] = table
    return out

def gradient_dynamics(
    model: models.QAPINN,
    n_points: int = 400,
    seed: int = 0,
) -> Dict:
    rng = np.random.default_rng(seed)
    model = model.train()

    # collocation points on the normalised interior domain
    xc = torch.tensor(rng.uniform(-1, 1, (n_points, 1)), dtype=torch.float32, requires_grad=True)
    tc = torch.tensor(rng.uniform(-1, 1, (n_points, 1)), dtype=torch.float32, requires_grad=True)

    u = model(xc, tc)
    u_th = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True)[0]
    u_xh = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
    u_xxh = torch.autograd.grad(u_xh, xc, torch.ones_like(u_xh), create_graph=True)[0]
    u_t = u_th * PHYSICS.st
    u_x = u_xh * PHYSICS.sx
    u_xx = u_xxh * PHYSICS.sx * PHYSICS.sx
    residual = u_t + u * u_x - PHYSICS.nu * u_xx
    loss = (residual ** 2).mean()

    model.zero_grad()
    loss.backward()

    groups = {
        "encoder": ["encoder.weight", "encoder.bias"],
        "quantum": ["qlayer.weights"],
        "decoder": ["dec_fc1.weight", "dec_fc1.bias", "dec_fc2.weight", "dec_fc2.bias"],
    }
    named = dict(model.named_parameters())
    group_norms = {}
    for gname, keys in groups.items():
        sq = 0.0
        count = 0
        for k in keys:
            p = named.get(k)
            if p is not None and p.grad is not None:
                sq += float((p.grad ** 2).sum())
                count += p.numel()
        group_norms[gname] = {
            "grad_l2": float(np.sqrt(sq)),
            "grad_rms": float(np.sqrt(sq / max(count, 1))),
            "n_params": count,
        }
    model.eval()
    q = group_norms["quantum"]["grad_rms"]
    c = group_norms["decoder"]["grad_rms"]
    return {
        "n_qubits": model.n_qubits,
        "loss": float(loss.item()),
        "group_grad_norms": group_norms,
        "quantum_to_decoder_rms_ratio": float(q / (c + 1e-30)),
    }


def gradient_dynamics_suite(qubit_range=(3, 4, 5), seed: int = 0) -> Dict:
    """Gradient-magnitude fingerprint for every QA-PINN checkpoint."""
    out = {}
    for q in qubit_range:
        try:
            model, _ = models.load_qapinn(q)
            out[str(q)] = gradient_dynamics(model, seed=seed)
        except Exception as exc:  # pragma: no cover
            out[str(q)] = {"error": str(exc)}
    return out

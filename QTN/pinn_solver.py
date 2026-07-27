"""Physics-informed neural network solver for the 1D viscous Burgers equation.

The network approximates u(x, t) on a rectangular space-time domain and is
trained on a composite objective combining the PDE residual with initial and
boundary condition penalties.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from torch import Tensor, nn

from classical_gmres import (
    BurgersConfig,
    InitialConditionFactory,
    linf_error,
    relative_l2_error,
)

__all__ = [
    "PINNArchitecture",
    "TrainingConfig",
    "SamplingConfig",
    "TrainingHistory",
    "MLP",
    "BurgersPINN",
    "plot_loss_curves",
    "plot_prediction_vs_reference",
    "plot_residual_heatmap",
    "plot_solution_surface",
    "plot_slice_comparison",
]

logger = logging.getLogger(__name__)
if not logger.handlers:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

ActivationName = Literal["tanh", "sin", "gelu", "silu"]
SchedulerName = Literal["cosine", "plateau", "step", "none"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PINNArchitecture:
    """Fully connected network geometry."""

    n_hidden_layers: int = 3
    width: int = 50
    activation: ActivationName = "tanh"
    input_dim: int = 2
    output_dim: int = 1
    fourier_features: int = 0
    fourier_scale: float = 5.0

    def __post_init__(self) -> None:
        if self.n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1.")
        if self.width < 1:
            raise ValueError("width must be >= 1.")
        if self.fourier_features < 0:
            raise ValueError("fourier_features must be non-negative.")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Collocation point counts and resampling policy."""

    n_collocation: int = 10_000
    n_initial: int = 400
    n_boundary: int = 400
    n_validation: int = 2_000
    resample_every: int = 0  # 0 disables resampling
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.n_collocation, self.n_initial, self.n_boundary) < 1:
            raise ValueError("All sample counts must be >= 1.")
        if self.resample_every < 0:
            raise ValueError("resample_every must be non-negative.")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimiser, scheduler, and stopping options."""

    epochs: int = 10_000
    learning_rate: float = 1e-3
    weight_pde: float = 1.0
    weight_ic: float = 1.0
    weight_bc: float = 1.0
    scheduler: SchedulerName = "cosine"
    scheduler_step_size: int = 2_000
    scheduler_gamma: float = 0.5
    min_learning_rate: float = 1e-6
    grad_clip: float | None = 1.0
    early_stopping_patience: int = 0  # 0 disables
    early_stopping_min_delta: float = 0.0
    log_every: int = 500
    validate_every: int = 250
    use_amp: bool = False
    lbfgs_steps: int = 0
    device: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if min(self.weight_pde, self.weight_ic, self.weight_bc) < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if self.log_every < 1 or self.validate_every < 1:
            raise ValueError("log_every and validate_every must be >= 1.")
        if self.lbfgs_steps < 0:
            raise ValueError("lbfgs_steps must be non-negative.")


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch loss traces and run metrics."""

    epoch: list[int] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    pde: list[float] = field(default_factory=list)
    ic: list[float] = field(default_factory=list)
    bc: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    validation_epoch: list[int] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    peak_memory_mb: float = 0.0

    def record(
        self,
        epoch: int,
        losses: dict[str, float],
        learning_rate: float,
    ) -> None:
        self.epoch.append(epoch)
        self.total.append(losses["total"])
        self.pde.append(losses["pde"])
        self.ic.append(losses["ic"])
        self.bc.append(losses["bc"])
        self.learning_rate.append(learning_rate)

    def record_validation(self, epoch: int, value: float) -> None:
        self.validation_epoch.append(epoch)
        self.validation.append(value)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class _Sine(nn.Module):
    """Sinusoidal activation."""

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return torch.sin(x)


class _FourierEncoding(nn.Module):
    """Random Fourier feature lifting of the input coordinates."""

    def __init__(self, input_dim: int, n_features: int, scale: float) -> None:
        super().__init__()
        b = torch.randn(input_dim, n_features) * scale
        self.register_buffer("b", b)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        proj = 2.0 * torch.pi * (x @ self.b)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)

    @property
    def output_dim(self) -> int:
        return int(self.b.shape[0] + 2 * self.b.shape[1])


class MLP(nn.Module):
    """Configurable fully connected network with Xavier initialisation."""

    def __init__(self, arch: PINNArchitecture) -> None:
        super().__init__()
        self.arch = arch

        if arch.fourier_features > 0:
            self.encoding: nn.Module | None = _FourierEncoding(
                arch.input_dim, arch.fourier_features, arch.fourier_scale
            )
            in_dim = self.encoding.output_dim
        else:
            self.encoding = None
            in_dim = arch.input_dim

        activation = (
            _Sine if arch.activation == "sin" else _ACTIVATIONS[arch.activation]
        )
        layers: list[nn.Module] = []
        for _ in range(arch.n_hidden_layers):
            layers.extend([nn.Linear(in_dim, arch.width), activation()])
            in_dim = arch.width
        layers.append(nn.Linear(in_dim, arch.output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        if self.encoding is not None:
            x = self.encoding(x)
        return self.net(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BurgersPINN:
    """Trains and evaluates a physics-informed network for Burgers' equation."""

    def __init__(
        self,
        problem: BurgersConfig | None = None,
        architecture: PINNArchitecture | None = None,
        training: TrainingConfig | None = None,
        sampling: SamplingConfig | None = None,
    ) -> None:
        self.problem = problem or BurgersConfig()
        self.architecture = architecture or PINNArchitecture()
        self.training = training or TrainingConfig()
        self.sampling = sampling or SamplingConfig()

        torch.manual_seed(self.training.seed)
        np.random.seed(self.training.seed)

        self.device = _resolve_device(self.training.device)
        self.model = MLP(self.architecture).to(self.device)
        self.history = TrainingHistory()
        self._rng = np.random.default_rng(self.sampling.seed)
        self._u0 = InitialConditionFactory.callable_for(self.problem)
        self._amp_enabled = self.training.use_amp and self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

        logger.info(
            "PINN initialised on %s with %d parameters.",
            self.device,
            self.model.n_parameters,
        )

    # -- sampling ----------------------------------------------------------- #
    def _tensor(self, array: np.ndarray, requires_grad: bool = False) -> Tensor:
        t = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        return t.requires_grad_(True) if requires_grad else t

    def sample_interior(self, n: int | None = None) -> tuple[Tensor, Tensor]:
        """Uniformly sample interior collocation points."""
        n = n or self.sampling.n_collocation
        cfg = self.problem
        x = self._rng.uniform(cfg.x_min, cfg.x_max, size=(n, 1))
        t = self._rng.uniform(0.0, cfg.t_final, size=(n, 1))
        return self._tensor(x, True), self._tensor(t, True)

    def sample_initial(self, n: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Sample points on the t=0 slice together with target values."""
        n = n or self.sampling.n_initial
        cfg = self.problem
        x = self._rng.uniform(cfg.x_min, cfg.x_max, size=(n, 1))
        u = self._u0(x)
        t = np.zeros_like(x)
        return self._tensor(x), self._tensor(t), self._tensor(u)

    def sample_boundary(self, n: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Sample points on both Dirichlet boundaries."""
        n = n or self.sampling.n_boundary
        cfg = self.problem
        half = max(n // 2, 1)
        t = self._rng.uniform(0.0, cfg.t_final, size=(2 * half, 1))
        x = np.concatenate(
            [np.full((half, 1), cfg.x_min), np.full((half, 1), cfg.x_max)]
        )
        u = np.concatenate(
            [np.full((half, 1), cfg.bc_left), np.full((half, 1), cfg.bc_right)]
        )
        return self._tensor(x), self._tensor(t), self._tensor(u)

    # -- physics ------------------------------------------------------------ #
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Evaluate the network at the given coordinates."""
        return self.model(torch.cat([x, t], dim=1))

    def pde_residual(self, x: Tensor, t: Tensor) -> Tensor:
        """Residual u_t + u u_x - nu u_xx evaluated by automatic differentiation."""
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return u_t + u * u_x - self.problem.viscosity * u_xx

    def compute_losses(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute PDE, IC, BC, and weighted total losses."""
        residual = self.pde_residual(batch["x_f"], batch["t_f"])
        loss_pde = torch.mean(residual**2)
        loss_ic = torch.mean(
            (self.forward(batch["x_i"], batch["t_i"]) - batch["u_i"]) ** 2
        )
        loss_bc = torch.mean(
            (self.forward(batch["x_b"], batch["t_b"]) - batch["u_b"]) ** 2
        )
        total = (
            self.training.weight_pde * loss_pde
            + self.training.weight_ic * loss_ic
            + self.training.weight_bc * loss_bc
        )
        return {"total": total, "pde": loss_pde, "ic": loss_ic, "bc": loss_bc}

    def _build_batch(self) -> dict[str, Tensor]:
        x_f, t_f = self.sample_interior()
        x_i, t_i, u_i = self.sample_initial()
        x_b, t_b, u_b = self.sample_boundary()
        return {
            "x_f": x_f, "t_f": t_f,
            "x_i": x_i, "t_i": t_i, "u_i": u_i,
            "x_b": x_b, "t_b": t_b, "u_b": u_b,
        }

    def _make_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        name = self.training.scheduler
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.training.epochs, eta_min=self.training.min_learning_rate
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.training.scheduler_step_size,
                gamma=self.training.scheduler_gamma,
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.training.scheduler_gamma,
                patience=max(self.training.scheduler_step_size // 10, 1),
                min_lr=self.training.min_learning_rate,
            )
        return None

    # -- validation --------------------------------------------------------- #
    def validation_residual(self, n: int | None = None) -> float:
        """Mean squared PDE residual on a freshly sampled validation set."""
        n = n or self.sampling.n_validation
        x, t = self.sample_interior(n)
        residual = self.pde_residual(x, t)
        return float(torch.mean(residual**2).detach().cpu())

    # -- training ----------------------------------------------------------- #
    def train(self) -> TrainingHistory:
        """Run Adam training followed by optional L-BFGS refinement."""
        cfg = self.training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        scheduler = self._make_scheduler(optimizer)

        batch = self._build_batch()
        best_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0

        tracemalloc.start()
        t_start = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            if self.sampling.resample_every and epoch % self.sampling.resample_every == 0:
                batch = self._build_batch()

            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            # Second-order autograd is unsupported under fp16, so the residual is
            # always evaluated at full precision; AMP wraps the forward pass only.
            with torch.amp.autocast("cuda", enabled=False):
                losses = self.compute_losses(batch)

            if self._amp_enabled:
                self._scaler.scale(losses["total"]).backward()
                if cfg.grad_clip is not None:
                    self._scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                losses["total"].backward()
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimizer.step()

            scalar_losses = {k: float(v.detach().cpu()) for k, v in losses.items()}
            current_lr = float(optimizer.param_groups[0]["lr"])
            self.history.record(epoch, scalar_losses, current_lr)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(scalar_losses["total"])
                else:
                    scheduler.step()

            if epoch % cfg.validate_every == 0:
                self.history.record_validation(epoch, self.validation_residual())

            if epoch % cfg.log_every == 0 or epoch == 1:
                logger.info(
                    "epoch %6d | total %.4e | pde %.4e | ic %.4e | bc %.4e | lr %.2e",
                    epoch,
                    scalar_losses["total"],
                    scalar_losses["pde"],
                    scalar_losses["ic"],
                    scalar_losses["bc"],
                    current_lr,
                )

            if scalar_losses["total"] < best_loss - cfg.early_stopping_min_delta:
                best_loss = scalar_losses["total"]
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if cfg.early_stopping_patience and stale_epochs >= cfg.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        if cfg.lbfgs_steps > 0:
            self._refine_lbfgs(batch)

        wall = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if best_state is not None and cfg.lbfgs_steps == 0:
            self.model.load_state_dict(best_state)

        self.history.wall_time_s = wall
        self.history.peak_memory_mb = peak / 1024**2
        logger.info("Training finished in %.2fs (best loss %.4e).", wall, best_loss)
        return self.history

    def _refine_lbfgs(self, batch: dict[str, Tensor]) -> None:
        """Polish the converged Adam solution with L-BFGS."""
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.training.lbfgs_steps,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = self.compute_losses(batch)["total"]
            loss.backward()
            return loss

        logger.info("Running L-BFGS refinement (%d steps).", self.training.lbfgs_steps)
        optimizer.step(closure)

    # -- inference ---------------------------------------------------------- #
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Evaluate the trained model on arbitrary coordinate arrays."""
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1, 1)
        if x_arr.shape != t_arr.shape:
            raise ValueError("x and t must contain the same number of points.")
        self.model.eval()
        with torch.no_grad():
            u = self.forward(self._tensor(x_arr), self._tensor(t_arr))
        return u.cpu().numpy().ravel()

    def predict_on_grid(
        self, x: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Evaluate on the tensor product grid, returning shape (n_times, n_x)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        times = np.asarray(times, dtype=np.float64).ravel()
        xx, tt = np.meshgrid(x, times, indexing="xy")
        flat = self.predict(xx.ravel(), tt.ravel())
        return flat.reshape(len(times), len(x))

    def residual_on_grid(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        """PDE residual magnitude on a space-time grid, shape (n_times, n_x)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        times = np.asarray(times, dtype=np.float64).ravel()
        xx, tt = np.meshgrid(x, times, indexing="xy")
        x_t = self._tensor(xx.reshape(-1, 1), True)
        t_t = self._tensor(tt.reshape(-1, 1), True)
        residual = self.pde_residual(x_t, t_t).detach().cpu().numpy().ravel()
        return residual.reshape(len(times), len(x))

    def evaluate(
        self, reference: np.ndarray, x: np.ndarray, t: float | None = None
    ) -> dict[str, float]:
        """Error and runtime metrics against a reference profile."""
        t_eval = self.problem.t_final if t is None else t
        pred = self.predict(x, np.full_like(np.asarray(x, dtype=np.float64), t_eval))
        return {
            "relative_l2": relative_l2_error(pred, reference),
            "linf": linf_error(pred, reference),
            "train_wall_time_s": self.history.wall_time_s,
            "peak_memory_mb": self.history.peak_memory_mb,
            "n_parameters": float(self.model.n_parameters),
        }

    # -- persistence -------------------------------------------------------- #
    def save_checkpoint(self, path: str | Path) -> Path:
        """Write model weights, configuration, and history to disk."""
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "problem": asdict(self.problem),
                "architecture": asdict(self.architecture),
                "training": asdict(self.training),
                "sampling": asdict(self.sampling),
                "history": self.history.as_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)
        return path

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, device: str | None = None
    ) -> "BurgersPINN":
        """Reconstruct a solver instance from a checkpoint file."""
        path = Path(path).with_suffix(".pt")
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}.")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        training = TrainingConfig(**payload["training"])
        if device is not None:
            training = TrainingConfig(**{**payload["training"], "device": device})
        solver = cls(
            problem=BurgersConfig(**payload["problem"]),
            architecture=PINNArchitecture(**payload["architecture"]),
            training=training,
            sampling=SamplingConfig(**payload["sampling"]),
        )
        solver.model.load_state_dict(payload["state_dict"])
        solver.model.to(solver.device)
        solver.history = TrainingHistory(**payload["history"])
        logger.info("Checkpoint loaded from %s", path)
        return solver

    def export_history(self, path: str | Path) -> Path:
        """Write the training history to a JSON file."""
        path = Path(path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.history.as_dict(), indent=2))
        return path


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def _finalise(fig: Figure, save_path: str | Path | None) -> Figure:
    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Figure written to %s", p)
    return fig


def plot_loss_curves(
    history: TrainingHistory, save_path: str | Path | None = None
) -> Figure:
    """Loss components, validation residual, and learning-rate schedule."""
    if not history.epoch:
        raise ValueError("History is empty; train the model first.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogy(history.epoch, history.total, label="total")
    axes[0].semilogy(history.epoch, history.pde, label="PDE", alpha=0.8)
    axes[0].semilogy(history.epoch, history.ic, label="IC", alpha=0.8)
    axes[0].semilogy(history.epoch, history.bc, label="BC", alpha=0.8)
    if history.validation:
        axes[0].semilogy(
            history.validation_epoch, history.validation, "k--", label="val residual"
        )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(history.epoch, history.learning_rate, color="tab:purple")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("Learning-rate schedule")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_prediction_vs_reference(
    x: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    time_label: str = "",
    save_path: str | Path | None = None,
) -> Figure:
    """Overlay the PINN prediction on a reference profile with error inset."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(x, reference, "k-", lw=2, label="reference")
    axes[0].plot(x, prediction, "r--", lw=2, label="PINN")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].set_title(f"Prediction vs reference {time_label}".strip())
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(x, np.abs(prediction - reference) + 1e-18)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|u_pred - u_ref|")
    axes[1].set_title(
        f"Pointwise error (rel. $L_2$ = {relative_l2_error(prediction, reference):.3e})"
    )
    axes[1].grid(alpha=0.3, which="both")
    return _finalise(fig, save_path)


def plot_residual_heatmap(
    x: np.ndarray,
    times: np.ndarray,
    residual: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """Log-scaled heatmap of the PDE residual over the space-time domain."""
    if residual.shape != (len(times), len(x)):
        raise ValueError(
            f"residual shape {residual.shape} does not match "
            f"({len(times)}, {len(x)})."
        )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    mesh = ax.pcolormesh(
        x, times, np.log10(np.abs(residual) + 1e-16), shading="auto", cmap="magma"
    )
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}|r|$")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("PDE residual magnitude")
    return _finalise(fig, save_path)


def plot_solution_surface(
    x: np.ndarray,
    times: np.ndarray,
    field: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """Combined 3D surface and 2D contour view of the predicted field."""
    if field.shape != (len(times), len(x)):
        raise ValueError(
            f"field shape {field.shape} does not match ({len(times)}, {len(x)})."
        )
    xx, tt = np.meshgrid(x, times, indexing="xy")
    fig = plt.figure(figsize=(13, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(xx, tt, field, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("t")
    ax3d.set_zlabel("u")
    ax3d.set_title("Solution surface")

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(xx, tt, field, levels=40, cmap="viridis")
    fig.colorbar(contour, ax=ax2d, label="u")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("t")
    ax2d.set_title("Contour view")
    return _finalise(fig, save_path)


def plot_slice_comparison(
    x: np.ndarray,
    times: Iterable[float],
    field: np.ndarray,
    time_axis: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot predicted profiles at selected times."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("plasma")
    times = list(times)
    for k, t in enumerate(times):
        idx = int(np.argmin(np.abs(time_axis - t)))
        ax.plot(
            x,
            field[idx],
            color=cmap(k / max(len(times) - 1, 1)),
            label=f"t = {time_axis[idx]:.3f}",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("PINN profiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finalise(fig, save_path)


if __name__ == "__main__":  # pragma: no cover
    problem = BurgersConfig(
        n_points=128, viscosity=0.01, t_final=0.1, initial_condition="smooth_step"
    )
    pinn = BurgersPINN(
        problem=problem,
        architecture=PINNArchitecture(n_hidden_layers=3, width=50),
        training=TrainingConfig(epochs=500, log_every=100),
    )
    pinn.train()
    grid = problem.grid()
    print("final residual:", pinn.validation_residual())
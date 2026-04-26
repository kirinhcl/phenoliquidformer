"""Latent Neural ODE for Timothy plant dynamics.

Models the plant as a continuous-time dynamical system where a latent
physiological state h(t) evolves under the forcing of multimodal observations
and optionally experimental context (WHC, genotype).

Architecture:
    - Encoder: initial observation + optional context -> h_0
    - Vector field f_theta: (h, observations(t), context) -> dh/dt
    - ODE solver: torchdiffeq (dopri5 adaptive)
    - Decoder (for reconstruction loss): h(t) -> reconstructed observation
    - Yield head: h(T_cut) -> (dry weight, flowering count)

Designed for small-sample, irregular time-series plant phenotyping data.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchdiffeq import odeint


class PiecewiseConstantInterpolator:
    """Piecewise-constant interpolation of observations between measured times.

    Given observations o_i at times t_i, returns o(t) = o_i where
    t_i is the latest measured time at or before t. This serves as the
    "forcing term" in the ODE, keeping the multimodal input available to
    the vector field at any t during integration.

    Note: this is NOT a torch.nn.Module; it's a stateful callable that
    holds reference tensors in a closure-like way.
    """

    def __init__(self, times: Tensor, observations: Tensor) -> None:
        """
        Args:
            times: (T,) float tensor of measurement times (sorted ascending)
            observations: (B, T, obs_dim) tensor of observations at those times
        """
        self.times = times
        self.observations = observations

    def __call__(self, t: Tensor) -> Tensor:
        """Query observation at time t (scalar).

        Returns:
            (B, obs_dim) observation at the most recent measurement at or before t
        """
        idx = torch.searchsorted(self.times, t.detach().unsqueeze(0) if t.dim() == 0 else t.detach(),
                                 right=True) - 1
        idx = idx.clamp(min=0, max=len(self.times) - 1)
        if idx.dim() == 0:
            return self.observations[:, idx]
        return self.observations[:, idx[0]]


class VectorField(nn.Module):
    """Neural ODE vector field: dh/dt = f(h, obs(t), context).

    2-layer MLP with tanh activation for bounded derivatives (stability).
    Context (if provided) is concatenated as additional static input.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        context_dim: int = 0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.context_dim = context_dim

        input_dim = latent_dim + obs_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Initialize the final layer small so initial dynamics are near-zero
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

        # These are set externally before calling odeint
        self._obs_interp: PiecewiseConstantInterpolator | None = None
        self._context: Tensor | None = None

    def set_drivers(
        self,
        obs_interp: PiecewiseConstantInterpolator,
        context: Tensor | None,
    ) -> None:
        """Set the observation interpolator and context for the next ODE solve."""
        self._obs_interp = obs_interp
        self._context = context

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        """Compute dh/dt at time t.

        Args:
            t: scalar time
            h: (B, latent_dim) current latent state

        Returns:
            (B, latent_dim) time derivative
        """
        if self._obs_interp is None:
            raise RuntimeError("Must call set_drivers() before integration")

        obs_t = self._obs_interp(t)  # (B, obs_dim)
        inputs = [h, obs_t]
        if self._context is not None:
            inputs.append(self._context)
        x = torch.cat(inputs, dim=-1)
        return self.net(x)


class LatentODE(nn.Module):
    """Latent Neural ODE model for plant dynamics.

    Workflow:
        1. Encode initial observation (+ optional context) -> h_0
        2. Integrate dh/dt = f(h, obs(t), context) from t_0 to t_final
        3. At each measurement time, the latent state h(t_i) is available
        4. Decoder reconstructs observations from h(t_i) (auxiliary loss)
        5. Yield head predicts (DW, flowering) from h(T_cut)
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        context_dim: int = 0,
        vf_hidden: int = 64,
        decoder_hidden: int = 64,
        ode_method: str = "dopri5",
        ode_rtol: float = 1e-3,
        ode_atol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        # Encoder: (obs_0, context) -> h_0
        enc_in = obs_dim + context_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, vf_hidden),
            nn.Tanh(),
            nn.Linear(vf_hidden, latent_dim),
        )

        # Vector field
        self.vector_field = VectorField(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            context_dim=context_dim,
            hidden_dim=vf_hidden,
        )

        # Decoder: h -> obs_hat (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, obs_dim),
        )

    def integrate(
        self,
        observations: Tensor,
        times: Tensor,
        active_mask: Tensor,
        context: Tensor | None = None,
        t_cut: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Integrate the latent ODE and return trajectory at measurement times.

        Args:
            observations: (B, T, obs_dim)
            times: (T,) float, measurement times (DAS values, sorted)
            active_mask: (B, T) bool, True = valid measurement
            context: (B, context_dim) static context or None
            t_cut: if given, integrate only up to this time

        Returns:
            h_traj: (B, T_used, latent_dim) latent at each measurement time (before cut)
            time_used: (T_used,) actual times used
        """
        # Apply truncation
        if t_cut is not None:
            mask = times <= t_cut
            time_used = times[mask]
            obs_used = observations[:, : len(time_used)]
        else:
            time_used = times
            obs_used = observations

        # Ensure times start from a valid point; use first time as t_0
        t_0 = time_used[0]

        # Encode initial state from first observation
        enc_in = observations[:, 0]  # (B, obs_dim)
        if context is not None:
            enc_in = torch.cat([enc_in, context], dim=-1)
        h_0 = self.encoder(enc_in)  # (B, latent_dim)

        # Set up observation interpolator and context for vector field
        interp = PiecewiseConstantInterpolator(time_used, obs_used)
        self.vector_field.set_drivers(interp, context)

        # Integrate
        h_traj = odeint(
            self.vector_field,
            h_0,
            time_used,
            method=self.ode_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )  # (T_used, B, latent_dim)

        # Reshape to (B, T_used, latent_dim)
        h_traj = h_traj.transpose(0, 1)

        return h_traj, time_used

    def reconstruction_loss(
        self,
        h_traj: Tensor,
        observations: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        """Compute MSE between decoded latent and actual observations.

        Args:
            h_traj: (B, T, latent_dim)
            observations: (B, T_full, obs_dim) original observations
            active_mask: (B, T_full) validity mask
        """
        T = h_traj.shape[1]
        obs_target = observations[:, :T]
        mask = active_mask[:, :T].unsqueeze(-1).float()

        obs_pred = self.decoder(h_traj)  # (B, T, obs_dim)
        diff = (obs_pred - obs_target) ** 2
        return (diff * mask).sum() / mask.sum().clamp(min=1.0)

    def smoothness_loss(self, h_traj: Tensor) -> Tensor:
        """Penalize large consecutive changes in latent state (regularizer)."""
        if h_traj.shape[1] < 2:
            return torch.tensor(0.0, device=h_traj.device)
        diffs = h_traj[:, 1:] - h_traj[:, :-1]
        return (diffs ** 2).mean()

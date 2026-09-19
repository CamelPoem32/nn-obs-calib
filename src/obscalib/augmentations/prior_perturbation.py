"""Perturbation of model calibration priors independently of true events."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from obscalib.augmentations.config import PriorPerturbationConfig
from obscalib.augmentations.sampling import sample_isotropic_perturbation, sample_signed_scalar_perturbation
from obscalib.calibration.state import CalibrationState
from obscalib.geometry.lie import se3_exp


class CalibrationPriorPerturber:
    """
    Perturb the calibration state supplied to the model.

    Prior perturbation does not represent a physical calibration-change event.
    It only makes the model start from an imperfect current calibration.

    Spatial perturbations follow the package's left-multiplicative convention:

        T_prior = Exp(delta_xi_prior) @ T_true

    and temporal perturbations are additive:

        tau_prior = tau_true + delta_tau_prior.
    """

    def __init__(self, config: PriorPerturbationConfig) -> None:
        self.config = config

    def __call__(
        self,
        calibration_truth: Mapping[str, CalibrationState],
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, CalibrationState], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Sample independent prior perturbations for every calibration key.

        Returns:
            calibration_priors:
                Calibration states that will be supplied to the model.

            prior_perturbation_xi_by_key:
                Sampled [phi, rho] spatial perturbations with shape [B, 6].

            prior_perturbation_tau_by_key:
                Sampled temporal perturbations with shape [B, 1].
        """

        _validate_generator(generator)
        _validate_calibration(calibration_truth)

        if not self.config.enabled:
            return dict(calibration_truth), {}, {}

        calibration_priors: dict[str, CalibrationState] = {}
        prior_perturbation_xi_by_key: dict[str, torch.Tensor] = {}
        prior_perturbation_tau_by_key: dict[str, torch.Tensor] = {}

        for calibration_key, true_state in calibration_truth.items():
            probability = self.config.probability_by_key.get(calibration_key, self.config.probability)

            prior_state, perturbation_xi, perturbation_tau = self._perturb_state(
                true_state=true_state,
                probability=probability,
                generator=generator,
            )

            calibration_priors[calibration_key] = prior_state
            prior_perturbation_xi_by_key[calibration_key] = perturbation_xi
            prior_perturbation_tau_by_key[calibration_key] = perturbation_tau

        return calibration_priors, prior_perturbation_xi_by_key, prior_perturbation_tau_by_key

    def _perturb_state(
        self,
        true_state: CalibrationState,
        probability: float,
        generator: torch.Generator | None,
    ) -> tuple[CalibrationState, torch.Tensor, torch.Tensor]:
        """Perturb one batched calibration state."""

        batch_size = true_state.transform.shape[0]
        device = true_state.transform.device
        dtype = true_state.transform.dtype

        # One Bernoulli decision controls whether the current prior is imperfect
        # for each batch item. Configured spatial/temporal components are then
        # all sampled for positive items.
        perturbation_occurred = torch.rand(
            batch_size,
            1,
            device=device,
            dtype=dtype,
            generator=generator,
        ) < probability

        perturbation_mask = perturbation_occurred.to(dtype=dtype)

        delta_phi = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        delta_rho = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        delta_tau = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        if self.config.rotation is not None:
            delta_phi = sample_isotropic_perturbation(
                config=self.config.rotation,
                batch_size=batch_size,
                dimension=3,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            delta_phi = delta_phi * perturbation_mask

        if self.config.translation is not None:
            delta_rho = sample_isotropic_perturbation(
                config=self.config.translation,
                batch_size=batch_size,
                dimension=3,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            delta_rho = delta_rho * perturbation_mask

        if self.config.time_offset is not None:
            delta_tau = sample_signed_scalar_perturbation(
                config=self.config.time_offset,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            delta_tau = delta_tau * perturbation_mask

        perturbation_xi = torch.cat((delta_phi, delta_rho), dim=-1)
        perturbation_transform = se3_exp(perturbation_xi)

        prior_state = CalibrationState(
            transform=perturbation_transform @ true_state.transform,
            time_offset=true_state.time_offset + delta_tau,
        )

        return prior_state, perturbation_xi, delta_tau


def _validate_calibration(calibration: Mapping[str, CalibrationState]) -> None:
    """Validate calibration states required by prior perturbation."""

    if not calibration:
        raise ValueError("At least one calibration state is required.")

    for calibration_key, state in calibration.items():
        if not calibration_key:
            raise ValueError("Calibration keys must be non-empty.")

        state.validate()

        if not torch.is_floating_point(state.transform):
            raise TypeError(f"Calibration transform {calibration_key!r} must have floating-point dtype.")

        if not torch.is_floating_point(state.time_offset):
            raise TypeError(f"Calibration time offset {calibration_key!r} must have floating-point dtype.")

        if state.transform.device != state.time_offset.device:
            raise ValueError(f"Calibration transform and time offset {calibration_key!r} must be on the same device.")

        if state.transform.dtype != state.time_offset.dtype:
            raise ValueError(f"Calibration transform and time offset {calibration_key!r} must have the same dtype.")


def _validate_generator(generator: torch.Generator | None) -> None:
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")
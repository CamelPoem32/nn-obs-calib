"""Perturbation of model calibration priors independently of true events."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from obscalib.augmentations.config import PriorPerturbationConfig
from obscalib.augmentations.sampling import sample_isotropic_perturbation, sample_signed_scalar_perturbation
from obscalib.calibration.state import CalibrationState
from obscalib.geometry.lie import se3_exp


def apply_prior_perturbation(
    calibration_truth: Mapping[str, CalibrationState],
    delta_xi_by_key: Mapping[str, torch.Tensor],
    delta_tau_by_key: Mapping[str, torch.Tensor],
) -> dict[str, CalibrationState]:
    """
    Apply explicit left calibration-prior perturbations.

    The convention is

        T_prior = Exp(delta_xi) @ T_true
        tau_prior = tau_true + delta_tau

    This is the same convention used by CalibrationPriorPerturber.
    """

    _validate_calibration(calibration_truth)

    calibration_keys = set(calibration_truth)

    if set(delta_xi_by_key) != calibration_keys:
        raise KeyError(
            "delta_xi_by_key must contain exactly the calibration-truth keys. "
            f"Expected {sorted(calibration_keys)}, got {sorted(delta_xi_by_key)}."
        )

    if set(delta_tau_by_key) != calibration_keys:
        raise KeyError(
            "delta_tau_by_key must contain exactly the calibration-truth keys. "
            f"Expected {sorted(calibration_keys)}, got {sorted(delta_tau_by_key)}."
        )

    perturbed_calibration = {}

    for calibration_key, true_state in calibration_truth.items():
        true_state.validate()

        delta_xi = delta_xi_by_key[calibration_key]
        delta_tau = delta_tau_by_key[calibration_key]

        batch_size = true_state.transform.shape[0]

        if delta_xi.shape != (batch_size, 6):
            raise ValueError(
                f"delta_xi_by_key[{calibration_key!r}] must have shape "
                f"{(batch_size, 6)}, got {tuple(delta_xi.shape)}."
            )

        if delta_tau.shape != (batch_size, 1):
            raise ValueError(
                f"delta_tau_by_key[{calibration_key!r}] must have shape "
                f"{(batch_size, 1)}, got {tuple(delta_tau.shape)}."
            )

        if delta_xi.device != true_state.transform.device:
            raise ValueError(
                f"delta_xi_by_key[{calibration_key!r}] and calibration transform "
                "must be on the same device."
            )

        if delta_tau.device != true_state.time_offset.device:
            raise ValueError(
                f"delta_tau_by_key[{calibration_key!r}] and calibration time offset "
                "must be on the same device."
            )

        if delta_xi.dtype != true_state.transform.dtype:
            raise ValueError(
                f"delta_xi_by_key[{calibration_key!r}] and calibration transform "
                "must have the same dtype."
            )

        if delta_tau.dtype != true_state.time_offset.dtype:
            raise ValueError(
                f"delta_tau_by_key[{calibration_key!r}] and calibration time offset "
                "must have the same dtype."
            )

        perturbed_calibration[calibration_key] = CalibrationState(
            transform=se3_exp(delta_xi) @ true_state.transform,
            time_offset=true_state.time_offset + delta_tau,
        )

        perturbed_calibration[calibration_key].validate()

    return perturbed_calibration


class CalibrationPriorPerturber:
    """
    Perturb the calibration state supplied to the model.

    Prior perturbation does not represent a physical calibration-change event.
    It only makes the model start from an imperfect current calibration.

    Spatial perturbations follow the package's left-multiplicative convention:

        T_prior = Exp(delta_xi_prior) @ T_true

    and temporal perturbations are additive:

        tau_prior = tau_true + delta_tau_prior.

    By default, every calibration key is perturbed independently, preserving
    the original augmentation behavior.

    When ``one_calibration_per_sample`` is enabled, one calibration key is
    sampled uniformly for every batch item and only that calibration is eligible
    for perturbation.

    When ``one_rotation_axis_per_sample`` is enabled, the rotational
    perturbation is a signed scalar applied along one randomly sampled
    Cartesian tangent-space axis for every batch item.
    """

    def __init__(self, config: PriorPerturbationConfig) -> None:
        self.config = config

    def __call__(self, calibration_truth: Mapping[str, CalibrationState], generator: torch.Generator | None = None) -> tuple[dict[str, CalibrationState], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Sample calibration-prior perturbations.

        Returns:
            calibration_priors:
                Calibration states supplied to the model.

            prior_perturbation_xi_by_key:
                Sampled [phi, rho] spatial perturbations with shape [B, 6].

            prior_perturbation_tau_by_key:
                Sampled temporal perturbations with shape [B, 1].
        """

        _validate_generator(generator)
        _validate_calibration(calibration_truth)

        if not self.config.enabled:
            return dict(calibration_truth), {}, {}

        calibration_keys = tuple(calibration_truth)
        reference_state = calibration_truth[calibration_keys[0]]

        batch_size = reference_state.transform.shape[0]
        device = reference_state.transform.device

        for calibration_key in calibration_keys[1:]:
            state = calibration_truth[calibration_key]

            if state.transform.shape[0] != batch_size:
                raise ValueError("All calibration states must have the same batch size.")

            if state.transform.device != device:
                raise ValueError("All calibration states must be on the same device.")

        calibration_selection_masks = self._sample_calibration_selection_masks(calibration_keys=calibration_keys, batch_size=batch_size, device=device, generator=generator)

        calibration_priors: dict[str, CalibrationState] = {}
        prior_perturbation_xi_by_key: dict[str, torch.Tensor] = {}
        prior_perturbation_tau_by_key: dict[str, torch.Tensor] = {}

        for calibration_key, true_state in calibration_truth.items():
            probability = self.config.probability_by_key.get(calibration_key, self.config.probability)

            prior_state, perturbation_xi, perturbation_tau = self._perturb_state(true_state=true_state, probability=probability, calibration_selection_mask=calibration_selection_masks[calibration_key], generator=generator)

            calibration_priors[calibration_key] = prior_state
            prior_perturbation_xi_by_key[calibration_key] = perturbation_xi
            prior_perturbation_tau_by_key[calibration_key] = perturbation_tau

        return calibration_priors, prior_perturbation_xi_by_key, prior_perturbation_tau_by_key

    def _sample_calibration_selection_masks(self, calibration_keys: tuple[str, ...], batch_size: int, device: torch.device, generator: torch.Generator | None) -> dict[str, torch.Tensor]:
        """
        Return one [B, 1] eligibility mask per calibration key.

        With ordinary independent augmentation every mask is True.

        With one_calibration_per_sample enabled, exactly one key is selected
        uniformly for every batch item. The selected key is then still subject
        to its configured Bernoulli perturbation probability.
        """

        if not self.config.one_calibration_per_sample:
            return {calibration_key: torch.ones(batch_size, 1, dtype=torch.bool, device=device) for calibration_key in calibration_keys}

        selected_calibration_ids = torch.randint(low=0, high=len(calibration_keys), size=(batch_size,), device=device, generator=generator)

        return {calibration_key: (selected_calibration_ids == calibration_id).unsqueeze(-1) for calibration_id, calibration_key in enumerate(calibration_keys)}

    def _perturb_state(self, true_state: CalibrationState, probability: float, calibration_selection_mask: torch.Tensor, generator: torch.Generator | None) -> tuple[CalibrationState, torch.Tensor, torch.Tensor]:
        """Perturb one batched calibration state."""

        batch_size = true_state.transform.shape[0]
        device = true_state.transform.device
        dtype = true_state.transform.dtype

        if calibration_selection_mask.shape != (batch_size, 1):
            raise ValueError(f"calibration_selection_mask must have shape {(batch_size, 1)}, got {tuple(calibration_selection_mask.shape)}.")

        if calibration_selection_mask.device != device:
            raise ValueError("calibration_selection_mask and calibration state must be on the same device.")

        perturbation_occurred = torch.rand(batch_size, 1, device=device, dtype=dtype, generator=generator) < probability
        perturbation_mask = (calibration_selection_mask & perturbation_occurred).to(dtype=dtype)

        delta_phi = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        delta_rho = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        delta_tau = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        if self.config.rotation is not None:
            delta_phi = self._sample_rotation_perturbation(batch_size=batch_size, device=device, dtype=dtype, generator=generator)
            delta_phi = delta_phi * perturbation_mask

        if self.config.translation is not None:
            delta_rho = sample_isotropic_perturbation(config=self.config.translation, batch_size=batch_size, dimension=3, device=device, dtype=dtype, generator=generator)
            delta_rho = delta_rho * perturbation_mask

        if self.config.time_offset is not None:
            delta_tau = sample_signed_scalar_perturbation(config=self.config.time_offset, batch_size=batch_size, device=device, dtype=dtype, generator=generator)
            delta_tau = delta_tau * perturbation_mask

        perturbation_xi = torch.cat((delta_phi, delta_rho), dim=-1)
        perturbation_transform = se3_exp(perturbation_xi)

        prior_state = CalibrationState(transform=perturbation_transform @ true_state.transform, time_offset=true_state.time_offset + delta_tau)
        prior_state.validate()

        return prior_state, perturbation_xi, delta_tau

    def _sample_rotation_perturbation(self, batch_size: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None) -> torch.Tensor:
        """
        Sample either an arbitrary tangent-space rotation or one random axis.

        The one-axis mode samples both the axis and signed angle independently
        for every batch item.
        """

        if self.config.rotation is None:
            raise RuntimeError("Rotation perturbation requested without rotation configuration.")

        if not self.config.one_rotation_axis_per_sample:
            return sample_isotropic_perturbation(config=self.config.rotation, batch_size=batch_size, dimension=3, device=device, dtype=dtype, generator=generator)

        selected_axes = torch.randint(low=0, high=3, size=(batch_size, 1), device=device, generator=generator)
        signed_angles = sample_signed_scalar_perturbation(config=self.config.rotation, batch_size=batch_size, device=device, dtype=dtype, generator=generator)

        delta_phi = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        delta_phi.scatter_(dim=1, index=selected_axes, src=signed_angles)

        return delta_phi


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
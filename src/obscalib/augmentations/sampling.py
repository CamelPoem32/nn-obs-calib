"""Reusable random samplers for synthetic calibration augmentation."""

from __future__ import annotations

import math

import torch

from obscalib.augmentations.config import PerturbationDistribution, PerturbationMagnitudeConfig


def sample_uniform_so3(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Haar-uniform rotations on SO3.

    Returns:
        Rotation matrices with shape [B, 3, 3].
    """

    _validate_batch_size(batch_size)
    _validate_floating_dtype(dtype)
    _validate_generator(generator)

    # A normalized isotropic Gaussian quaternion is uniform on the unit
    # 3-sphere. Identifying q and -q induces the uniform Haar measure on SO3.
    quats = torch.randn(batch_size, 4, device=device, dtype=dtype, generator=generator)
    quats = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)

    w, x, y, z = quats.unbind(dim=-1)

    R = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )

    return R.reshape(batch_size, 3, 3)


def sample_random_unit_vectors(
    batch_size: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample isotropic unit vectors.

    Returns:
        Unit vectors with shape [B, dimension].
    """

    _validate_batch_size(batch_size)
    _validate_floating_dtype(dtype)
    _validate_generator(generator)

    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    vectors = torch.randn(batch_size, dimension, device=device, dtype=dtype, generator=generator)
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)

    return vectors / norms


def sample_perturbation_magnitudes(
    config: PerturbationMagnitudeConfig,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample nonnegative bounded perturbation magnitudes.

    Returns:
        Magnitudes with shape [B, 1].

    The distributions are interpreted as:

        UNIFORM:
            magnitude ~ Uniform(0, maximum)

        TRUNCATED_NORMAL:
            magnitude ~ |Normal(0, scale)| conditioned on magnitude <= maximum

        TRUNCATED_STUDENT_T:
            magnitude ~ scale * |StudentT(df)| conditioned on
            magnitude <= maximum
    """

    _validate_batch_size(batch_size)
    _validate_floating_dtype(dtype)
    _validate_generator(generator)

    if config.distribution == PerturbationDistribution.UNIFORM:
        return config.maximum * torch.rand(batch_size, 1, device=device, dtype=dtype, generator=generator)

    if config.distribution == PerturbationDistribution.TRUNCATED_NORMAL:
        return _sample_truncated_half_normal(
            batch_size=batch_size,
            scale=config.scale,
            maximum=config.maximum,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    if config.distribution == PerturbationDistribution.TRUNCATED_STUDENT_T:
        return _sample_truncated_absolute_student_t(
            batch_size=batch_size,
            scale=config.scale,
            maximum=config.maximum,
            degrees_of_freedom=config.degrees_of_freedom,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    raise ValueError(f"Unsupported perturbation distribution: {config.distribution!r}")


def sample_isotropic_perturbation(
    config: PerturbationMagnitudeConfig,
    batch_size: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample an isotropic bounded perturbation vector.

    A nonnegative magnitude is sampled according to `config`, while its
    direction is sampled uniformly on the unit sphere.

    Returns:
        Perturbation vectors with shape [B, dimension].
    """

    magnitudes = sample_perturbation_magnitudes(
        config=config,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    directions = sample_random_unit_vectors(
        batch_size=batch_size,
        dimension=dimension,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    return magnitudes * directions


def sample_signed_scalar_perturbation(
    config: PerturbationMagnitudeConfig,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample signed bounded scalar perturbations.

    This is intended primarily for temporal offsets.

    Returns:
        Signed perturbations with shape [B, 1].
    """

    magnitudes = sample_perturbation_magnitudes(
        config=config,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    signs = torch.randint(
        low=0,
        high=2,
        size=(batch_size, 1),
        device=device,
        generator=generator,
    )

    signs = signs.to(dtype=dtype) * 2.0 - 1.0

    return magnitudes * signs


def _sample_truncated_half_normal(
    batch_size: int,
    scale: float | None,
    maximum: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample a half-normal distribution truncated to [0, maximum]."""

    if scale is None:
        raise ValueError("TRUNCATED_NORMAL requires scale.")

    # The half-normal CDF is erf(x / (sqrt(2) * scale)).
    # Sampling its truncated inverse CDF avoids rejection sampling.
    sqrt_two = math.sqrt(2.0)

    maximum_normalized = torch.tensor(maximum / (sqrt_two * scale), device=device, dtype=dtype)
    maximum_cdf = torch.erf(maximum_normalized)

    uniform = torch.rand(batch_size, 1, device=device, dtype=dtype, generator=generator)

    return sqrt_two * scale * torch.erfinv(uniform * maximum_cdf)


def _sample_truncated_absolute_student_t(
    batch_size: int,
    scale: float | None,
    maximum: float,
    degrees_of_freedom: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample a bounded absolute Student-t magnitude using rejection sampling."""

    if scale is None:
        raise ValueError("TRUNCATED_STUDENT_T requires scale.")

    samples = torch.empty(batch_size, 1, device=device, dtype=dtype)
    unresolved = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Rejection sampling is appropriate here because augmentation bounds should
    # normally be several characteristic scales wide.
    while torch.any(unresolved):
        count = int(unresolved.sum().item())

        candidates = scale * torch.abs(
            _sample_standard_student_t(
                count=count,
                degrees_of_freedom=degrees_of_freedom,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )

        accepted = candidates[:, 0] <= maximum

        unresolved_indices = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        accepted_indices = unresolved_indices[accepted.squeeze(-1)]

        samples[accepted_indices] = candidates[accepted]
        unresolved[accepted_indices] = False

    return samples


def _sample_standard_student_t(
    count: int,
    degrees_of_freedom: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample a standard Student-t distribution with integer degrees of freedom."""

    # If z ~ N(0, 1) and v ~ ChiSquare(nu), then
    #
    #     z / sqrt(v / nu)
    #
    # follows StudentT(nu). For integer nu, ChiSquare(nu) is the sum of nu
    # independent squared standard-normal variables.
    numerator = torch.randn(count, 1, device=device, dtype=dtype, generator=generator)
    chi_square_components = torch.randn(count, degrees_of_freedom, device=device, dtype=dtype, generator=generator)
    chi_square = torch.sum(chi_square_components * chi_square_components, dim=-1, keepdim=True)

    return numerator / torch.sqrt(chi_square / float(degrees_of_freedom))


def _validate_batch_size(batch_size: int) -> None:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point.")


def _validate_generator(generator: torch.Generator | None) -> None:
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")
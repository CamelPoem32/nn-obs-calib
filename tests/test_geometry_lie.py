"""Tests for differentiable SO(3) and SE(3) Lie-group operations."""

import math

import torch

from obscalib.geometry import (
    SE3LogMap,
    SO3LogMap,
    VectorObservationMap,
    se3_exp,
    se3_hat,
    se3_log,
    se3_vee,
    so3_exp,
    so3_hat,
    so3_log,
    so3_vee,
)


def test_so3_hat_and_vee_preserve_batch_dimensions() -> None:
    phi = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, 0.5, 2.0], [0.1, -0.2, 0.3]],
        ],
        dtype=torch.float64,
    )

    phi_hat = so3_hat(phi)

    # Each [..., 3] vector receives its own [..., 3, 3] skew matrix.
    assert phi.shape == (2, 2, 3)
    assert phi_hat.shape == (2, 2, 3, 3)

    expected_first = torch.tensor(
        [
            [0.0, -3.0, 2.0],
            [3.0, 0.0, -1.0],
            [-2.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(phi_hat[0, 0], expected_first)
    torch.testing.assert_close(so3_vee(phi_hat), phi)


def test_so3_exp_log_round_trip() -> None:
    torch.manual_seed(0)

    # Keep rotations comfortably inside the principal logarithm region.
    phi = 0.4 * torch.randn(16, 3, dtype=torch.float64)

    rotation = so3_exp(phi)
    recovered_phi = so3_log(rotation)

    assert rotation.shape == (16, 3, 3)
    assert recovered_phi.shape == phi.shape

    torch.testing.assert_close(
        recovered_phi,
        phi,
        atol=1e-9,
        rtol=1e-9,
    )


def test_so3_log_near_pi_reconstructs_rotation() -> None:
    axis = torch.tensor(
        [1.0, 2.0, -1.0],
        dtype=torch.float64,
    )
    axis = axis / torch.linalg.vector_norm(axis)

    phi = axis * (math.pi - 1e-4)

    rotation = so3_exp(phi)
    recovered_phi = so3_log(rotation)
    reconstructed_rotation = so3_exp(recovered_phi)

    # Compare rotations instead of vectors because the axis representation
    # becomes sign-ambiguous at exactly pi.
    torch.testing.assert_close(
        reconstructed_rotation,
        rotation,
        atol=1e-8,
        rtol=1e-8,
    )


def test_se3_hat_and_vee_use_phi_rho_ordering() -> None:
    xi = torch.tensor(
        [0.1, -0.2, 0.3, 1.0, 2.0, 3.0],
        dtype=torch.float64,
    )

    xi_hat = se3_hat(xi)

    # xi = [phi, rho].
    torch.testing.assert_close(
        xi_hat[:3, :3],
        so3_hat(xi[:3]),
    )
    torch.testing.assert_close(
        xi_hat[:3, 3],
        xi[3:],
    )
    torch.testing.assert_close(
        xi_hat[3],
        torch.zeros(4, dtype=torch.float64),
    )

    torch.testing.assert_close(se3_vee(xi_hat), xi)


def test_se3_exp_pure_translation() -> None:
    xi = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, -2.0, 3.0],
        dtype=torch.float64,
    )

    transform = se3_exp(xi)

    expected = torch.eye(4, dtype=torch.float64)
    expected[:3, 3] = xi[3:]

    torch.testing.assert_close(
        transform,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_se3_exp_matches_torch_matrix_exp() -> None:
    torch.manual_seed(1)

    # Moderate rotations avoid logarithmic singularities while still testing
    # the full coupled SE(3) exponential.
    xi = 0.3 * torch.randn(32, 6, dtype=torch.float64)

    closed_form = se3_exp(xi)
    matrix_exp_reference = torch.matrix_exp(se3_hat(xi))

    torch.testing.assert_close(
        closed_form,
        matrix_exp_reference,
        atol=1e-10,
        rtol=1e-10,
    )


def test_se3_exp_log_round_trip() -> None:
    torch.manual_seed(2)

    xi = 0.3 * torch.randn(32, 6, dtype=torch.float64)

    transform = se3_exp(xi)
    recovered_xi = se3_log(transform)

    torch.testing.assert_close(
        recovered_xi,
        xi,
        atol=1e-9,
        rtol=1e-9,
    )


def test_se3_operations_preserve_arbitrary_leading_dimensions() -> None:
    torch.manual_seed(3)

    xi = 0.2 * torch.randn(
        2,
        4,
        5,
        6,
        dtype=torch.float64,
    )

    transform = se3_exp(xi)
    recovered_xi = se3_log(transform)

    assert transform.shape == (2, 4, 5, 4, 4)
    assert recovered_xi.shape == (2, 4, 5, 6)

    torch.testing.assert_close(
        recovered_xi,
        xi,
        atol=1e-9,
        rtol=1e-9,
    )


def test_geometry_maps_use_lie_operations() -> None:
    torch.manual_seed(4)

    phi = 0.2 * torch.randn(3, 5, 3, dtype=torch.float64)
    rotation = so3_exp(phi)

    xi = 0.2 * torch.randn(3, 5, 6, dtype=torch.float64)
    transform = se3_exp(xi)

    vectors = torch.randn(3, 5, 7, dtype=torch.float64)

    so3_map = SO3LogMap()
    se3_map = SE3LogMap()
    vector_map = VectorObservationMap()

    torch.testing.assert_close(
        so3_map(rotation),
        so3_log(rotation),
    )
    torch.testing.assert_close(
        se3_map(transform),
        se3_log(transform),
    )
    torch.testing.assert_close(
        vector_map(vectors),
        vectors,
    )


def test_se3_exp_is_differentiable() -> None:
    torch.manual_seed(5)

    xi = (
        0.2 * torch.randn(8, 6, dtype=torch.float64)
    ).requires_grad_()

    transform = se3_exp(xi)

    # Use a non-constant weighted combination of transform entries so the
    # backward pass exercises both rotational and translational components.
    weights = torch.arange(
        1,
        17,
        dtype=torch.float64,
    ).reshape(1, 4, 4)

    loss = (transform * weights).sum()
    loss.backward()

    assert xi.grad is not None
    assert xi.grad.shape == xi.shape
    assert torch.isfinite(xi.grad).all()
from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import (
    GeometryType,
    MeasurementType,
    SensorMetadata,
    SensorStream,
    SensorStreamBatch,
    WindowBatch,
)
from obscalib.observability.estimators import ObservabilityEstimator
from obscalib.observability.layout import CalibrationParameterLayout
from obscalib.observability.mappings import ObservabilityMapper
from obscalib.observability.numerics import ObservabilityNumericsConfig
from obscalib.observability.structures import (
    BatchedObservabilityMatrix,
    ObservabilityResult,
    WindowObservabilityMatrix,
)
from obscalib.observability.timebase import (
    ReferenceTimeConfig,
    ReferenceTimePolicy,
    ReferenceTimebase,
    select_reference_timebase_single_window,
)
from obscalib.pipeline.window_step import WindowStep


def _lidar_stream(timestamps: list[float]) -> SensorStream:
    end_times = torch.tensor(timestamps, dtype=torch.float64)
    return SensorStream(
        values=torch.eye(4, dtype=torch.float64).repeat(len(timestamps), 1, 1),
        timestamps=end_times,
        interval_start_timestamps=end_times - 0.1,
    )


def test_calibration_parameter_layout_uses_explicit_key_order() -> None:
    layout = CalibrationParameterLayout.from_calibration_keys(("rear_lidar_custom", "front_imu_custom"))
    assert layout.block_for("rear_lidar_custom").parameter_names == ("phi_x", "phi_y", "phi_z", "rho_x", "rho_y", "rho_z", "tau")

    assert layout.calibration_keys == ("rear_lidar_custom", "front_imu_custom")
    assert layout.total_dimension == 14
    assert layout.block_for("rear_lidar_custom").parameter_slice == slice(0, 7)
    assert layout.block_for("front_imu_custom").spatial_slice == slice(7, 13)
    assert layout.block_for("front_imu_custom").time_offset_slice == slice(13, 14)

    with pytest.raises(ValueError, match="unique"):
        CalibrationParameterLayout.from_calibration_keys(("shared", "shared"))


def test_reference_timebase_selection_has_no_fixed_sensor_names() -> None:
    """
    Select an explicitly configured LiDAR reference without relying on sensor names.

    The high-level pipeline chooses one LiDAR as its reference stream. Automatic
    lowest-rate selection remains an optional policy, but it is not an
    architectural invariant of this test.
    """

    streams = {
        "gyro_any_name": SensorStream(
            values=torch.zeros(5, 3, dtype=torch.float64),
            timestamps=torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
        ),
        "slow_pose_any_name": _lidar_stream([0.2, 0.7, 1.0]),
        "fast_pose_any_name": _lidar_stream([0.1, 0.4, 0.7, 1.0]),
    }
    metadata = {
        "gyro_any_name": SensorMetadata(MeasurementType.IMU_GYROSCOPE, GeometryType.VECTOR, "physical_imu"),
        "slow_pose_any_name": SensorMetadata(MeasurementType.LIDAR_POSE, GeometryType.SE3, "physical_lidar_slow"),
        "fast_pose_any_name": SensorMetadata(MeasurementType.LIDAR_POSE, GeometryType.SE3, "physical_lidar_fast"),
    }

    selected = select_reference_timebase_single_window(
        streams,
        metadata,
        ReferenceTimeConfig(
            policy=ReferenceTimePolicy.EXPLICIT_STREAM,
            explicit_stream_key="slow_pose_any_name",
        ),
    )

    assert selected.stream_key == "slow_pose_any_name"
    torch.testing.assert_close(
        selected.timestamps,
        streams["slow_pose_any_name"].timestamps,
    )


def test_scientific_matrix_structures_preserve_layout_and_cpu_boundary() -> None:
    layout = CalibrationParameterLayout.from_calibration_keys(("calibration_any_name",))
    timebase = ReferenceTimebase("reference_any_name", torch.tensor([0.0, 0.5], dtype=torch.float64))
    timebase.validate()

    window_result = WindowObservabilityMatrix(
        fisher_information_matrix=torch.eye(7, dtype=torch.float64),
        projected_calibration_jacobian=torch.ones(3, 7, dtype=torch.float64),
        layout=layout,
        reference_timebase=timebase,
    )
    batched_result = BatchedObservabilityMatrix(
        fisher_information_matrix=torch.eye(7, dtype=torch.float64).repeat(2, 1, 1),
        layout=layout,
        reference_timebases=(timebase, timebase),
    )
    compatible_result = ObservabilityResult(raw=window_result)

    assert window_result.layout.calibration_keys == ("calibration_any_name",)
    assert batched_result.fisher_information_matrix.shape == (2, 7, 7)
    assert compatible_result.raw is window_result

    with pytest.raises(ValueError, match="detached"):
        WindowObservabilityMatrix(
            fisher_information_matrix=torch.eye(7, dtype=torch.float64, requires_grad=True),
            layout=layout,
            reference_timebase=timebase,
        )


def test_numerical_policy_rejects_invalid_tolerances_and_scales() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ObservabilityNumericsConfig(relative_tolerance=-1.0)

    with pytest.raises(ValueError, match="strictly positive"):
        ObservabilityNumericsConfig(parameter_scales=(1.0, 0.0))


class _RecordingEstimator(ObservabilityEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.received_metadata: Mapping[str, SensorMetadata] | None = None

    def forward(
        self,
        measurements: Mapping[str, SensorStreamBatch],
        calibration: Mapping[str, CalibrationState],
        metadata: Mapping[str, SensorMetadata],
    ) -> ObservabilityResult:
        self.received_metadata = metadata
        return ObservabilityResult(raw={"stream_keys": tuple(measurements)})


class _IdentityMapper(ObservabilityMapper):
    def forward(self, result: ObservabilityResult) -> ObservabilityResult:
        return result


def test_window_step_passes_sensor_metadata_to_estimator() -> None:
    estimator = _RecordingEstimator()
    step = object.__new__(WindowStep)
    step.observability_estimator = estimator
    step.observability_mapper = _IdentityMapper()

    metadata = {
        "stream_any_name": SensorMetadata(
            measurement_type=MeasurementType.IMU_GYROSCOPE,
            geometry_type=GeometryType.VECTOR,
            calibration_key="physical_sensor_any_name",
        )
    }
    window = WindowBatch(
        streams={
            "stream_any_name": SensorStreamBatch(
                values=torch.zeros(1, 2, 3, dtype=torch.float64),
                timestamps=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
                sample_mask=torch.ones(1, 2, dtype=torch.bool),
            )
        },
        current_calibration={},
        metadata=metadata,
    )

    result = step._compute_observability(window, {})

    assert result is not None
    assert estimator.received_metadata is metadata

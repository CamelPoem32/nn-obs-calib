"""Raw-window observability-matrix estimation for batched pipeline inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Protocol

import torch
from torch import nn

from obscalib.calibration.state import CalibrationState
from obscalib.data.structures import SensorMetadata, SensorStream, SensorStreamBatch
from obscalib.observability.layout import CalibrationParameterLayout
from obscalib.observability.linearization import linearize_single_window
from obscalib.observability.matrix import compute_observability_matrix_single_window
from obscalib.observability.numerics import ObservabilityNumericsConfig
from obscalib.observability.structures import BatchedObservabilityMatrix, ObservabilityResult
from obscalib.observability.timebase import ReferenceTimeConfig, ReferenceTimebase, select_reference_timebase_single_window


class TrajectoryProvider(Protocol):
    """
    Provide body trajectory poses at the selected reference timestamps.

    The observability equations require an actual trajectory linearization point
    ``T_W_B(t_k)``. That source is deliberately injected rather than guessed from
    a relative-pose stream inside the estimator.

    Implementations may use scan-to-map poses, an external odometry estimate, a
    factor-graph trajectory, or another validated trajectory source.
    """

    def __call__(
        self,
        measurements: Mapping[str, SensorStream],
        metadata: Mapping[str, SensorMetadata],
        calibration: Mapping[str, CalibrationState],
        reference_timebase: ReferenceTimebase,
    ) -> torch.Tensor:
        """
        Return body poses with shape ``[N_reference, 4, 4]``.
        """


class ObservabilityEstimator(nn.Module, ABC):
    """
    Common interface for scientific observability estimation from raw windows.

    Estimation happens before geometry preprocessing and outside autograd.
    """

    @abstractmethod
    def forward(
        self,
        measurements: Mapping[str, SensorStreamBatch],
        calibration: Mapping[str, CalibrationState],
        metadata: Mapping[str, SensorMetadata],
    ) -> ObservabilityResult:
        """
        Compute one scientific observability result per batch element.
        """


def _validate_batched_inputs(
    measurements: Mapping[str, SensorStreamBatch],
    calibration: Mapping[str, CalibrationState],
    metadata: Mapping[str, SensorMetadata],
    calibration_layout: CalibrationParameterLayout,
) -> int:
    """
    Validate raw batched inputs and return the shared batch size.
    """

    if not measurements:
        raise ValueError("At least one sensor stream is required.")
    if set(measurements) != set(metadata):
        raise ValueError("measurements and metadata must contain identical stream keys.")

    batch_sizes: list[int] = []

    for stream_key, stream in measurements.items():
        stream.validate()
        batch_sizes.append(stream.values.shape[0])

        calibration_key = metadata[stream_key].calibration_key

        if calibration_key not in calibration:
            raise KeyError(f"Stream {stream_key!r} requires missing calibration key {calibration_key!r}.")
        if calibration_key not in calibration_layout.calibration_keys:
            raise KeyError(f"Stream {stream_key!r} uses calibration key {calibration_key!r} outside the calibration layout.")

    for calibration_key in calibration_layout.calibration_keys:
        if calibration_key not in calibration:
            raise KeyError(f"Calibration layout requires missing calibration key {calibration_key!r}.")

    for state in calibration.values():
        state.validate()
        batch_sizes.append(state.transform.shape[0])

    if not batch_sizes:
        raise ValueError("Could not infer batch size.")
    if any(batch_size != batch_sizes[0] for batch_size in batch_sizes[1:]):
        raise ValueError("All sensor streams and calibration states must share batch size.")

    return batch_sizes[0]


def _unbatch_sensor_streams(measurements: Mapping[str, SensorStreamBatch], batch_index: int) -> dict[str, SensorStream]:
    """
    Extract one real, unpadded sensor window from a padded minibatch.
    """

    result: dict[str, SensorStream] = {}

    for stream_key, stream in measurements.items():
        valid_mask = stream.sample_mask[batch_index]

        values = stream.values[batch_index, valid_mask]
        timestamps = stream.timestamps[batch_index, valid_mask]
        interval_start_timestamps = None

        if stream.interval_start_timestamps is not None:
            interval_start_timestamps = stream.interval_start_timestamps[batch_index, valid_mask]

        single_stream = SensorStream(values=values, timestamps=timestamps, interval_start_timestamps=interval_start_timestamps)
        single_stream.validate()
        result[stream_key] = single_stream

    return result


def _unbatch_calibration(calibration: Mapping[str, CalibrationState], batch_index: int) -> dict[str, CalibrationState]:
    """
    Extract one batch-size-one calibration state mapping.
    """

    return {
        calibration_key: CalibrationState(transform=state.transform[batch_index : batch_index + 1], time_offset=state.time_offset[batch_index : batch_index + 1])
        for calibration_key, state in calibration.items()
    }


def _select_batched_tensor(tensor: torch.Tensor, batch_index: int, *, unbatched_ndim: int, name: str) -> torch.Tensor:
    """
    Select one item from a tensor that may be shared or batched.

    A tensor with ``unbatched_ndim`` dimensions is shared by all batch elements.
    A tensor with one additional leading dimension is indexed by ``batch_index``.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if tensor.ndim == unbatched_ndim:
        return tensor

    if tensor.ndim == unbatched_ndim + 1:
        if batch_index >= tensor.shape[0]:
            raise IndexError(f"{name} does not contain batch index {batch_index}.")
        return tensor[batch_index]

    raise ValueError(f"{name} must have {unbatched_ndim} dimensions when shared or {unbatched_ndim + 1} dimensions when batched.")


def _select_tensor_mapping(
    mapping: Mapping[str, torch.Tensor] | None,
    batch_index: int,
    *,
    unbatched_ndim: int,
    name: str,
) -> dict[str, torch.Tensor] | None:
    """
    Select one batch element from an optional named tensor mapping.
    """

    if mapping is None:
        return None

    return {
        key: _select_batched_tensor(value, batch_index, unbatched_ndim=unbatched_ndim, name=f"{name}[{key!r}]")
        for key, value in mapping.items()
    }

def _select_single_sensor_stream(stream: SensorStreamBatch, batch_index: int) -> SensorStream:
    """
    Extract one unpadded scientific sensor stream from a minibatch.

    Padding belongs only to the neural minibatch representation and must never
    enter the scientific observability pipeline.
    """

    stream.validate()

    valid_mask = stream.sample_mask[
        batch_index
    ]

    values = stream.values[
        batch_index,
        valid_mask,
    ]

    timestamps = stream.timestamps[
        batch_index,
        valid_mask,
    ]

    if stream.interval_start_timestamps is None:
        interval_start_timestamps = None
    else:
        interval_start_timestamps = stream.interval_start_timestamps[
            batch_index,
            valid_mask,
        ]

    return SensorStream(
        values=values,
        timestamps=timestamps,
        interval_start_timestamps=interval_start_timestamps,
    )

class ObservabilityMatrixEstimator(ObservabilityEstimator):
    """
    Compute one calibration Fisher matrix per raw temporal window.

    The estimator owns orchestration only:

        raw padded batch
            -> remove padding per batch element
            -> choose reference trajectory timestamps
            -> obtain trajectory poses from ``trajectory_provider``
            -> factor linearization
            -> whitening and nuisance marginalization
            -> calibration Fisher matrix.

    The trajectory source, residual covariance model, and numerical policy are
    explicit dependencies so none of them are silently invented here.
    """

    def __init__(
        self,
        calibration_layout: CalibrationParameterLayout,
        trajectory_provider: TrajectoryProvider,
        residual_covariance_by_stream: Mapping[str, torch.Tensor],
        *,
        reference_time_config: ReferenceTimeConfig | None = None,
        numerics: ObservabilityNumericsConfig | None = None,
        gravity_world: torch.Tensor | None = None,
        gyro_bias_by_calibration_key: Mapping[str, torch.Tensor] | None = None,
        njit: bool = False,
    ) -> None:
        super().__init__()

        if not callable(trajectory_provider):
            raise TypeError("trajectory_provider must be callable.")
        if not residual_covariance_by_stream:
            raise ValueError("residual_covariance_by_stream must not be empty.")

        self.calibration_layout = calibration_layout
        self.trajectory_provider = trajectory_provider
        self.residual_covariance_by_stream = dict(residual_covariance_by_stream)
        self.reference_time_config = reference_time_config if reference_time_config is not None else ReferenceTimeConfig()
        self.numerics = numerics if numerics is not None else ObservabilityNumericsConfig()
        self.gravity_world = gravity_world
        self.gyro_bias_by_calibration_key = None if gyro_bias_by_calibration_key is None else dict(gyro_bias_by_calibration_key)
        self.njit = bool(njit)

    def _estimate_single_window(
        self,
        measurements: Mapping[str, SensorStream],
        calibration: Mapping[str, CalibrationState],
        metadata: Mapping[str, SensorMetadata],
        *,
        residual_covariance_by_stream: Mapping[str, torch.Tensor],
        gyro_bias_by_calibration_key: Mapping[str, torch.Tensor] | None,
        gravity_world: torch.Tensor | None,
    ):
        """
        Compute the complete scientific observability result for one batch item.
        """

        reference_timebase = select_reference_timebase_single_window(measurements, metadata, self.reference_time_config)
        trajectory_poses = self.trajectory_provider(measurements, metadata, calibration, reference_timebase)

        if not isinstance(trajectory_poses, torch.Tensor):
            raise TypeError("trajectory_provider must return a torch.Tensor.")
        if trajectory_poses.ndim == 4 and trajectory_poses.shape[0] == 1:
            trajectory_poses = trajectory_poses[0]
        if trajectory_poses.shape != (reference_timebase.timestamps.numel(), 4, 4):
            raise ValueError(f"trajectory_provider must return shape [{reference_timebase.timestamps.numel()}, 4, 4]; got {tuple(trajectory_poses.shape)}.")

        lidar_interval_start_timestamps_by_stream = {
            stream_key: stream.interval_start_timestamps
            for stream_key, stream in measurements.items()
            if stream.interval_start_timestamps is not None
        }

        linearization = linearize_single_window(
            measurements,
            metadata,
            calibration,
            self.calibration_layout,
            reference_timebase,
            trajectory_poses=trajectory_poses,
            gyro_bias_by_calibration_key=gyro_bias_by_calibration_key,
            residual_covariance_by_stream=residual_covariance_by_stream,
            gravity_world=gravity_world,
            lidar_interval_start_timestamps_by_stream=lidar_interval_start_timestamps_by_stream,
            njit=self.njit,
        )

        return compute_observability_matrix_single_window(linearization, self.numerics)

    def forward(
        self,
        measurements: Mapping[str, SensorStreamBatch],
        calibration: Mapping[str, CalibrationState],
        metadata: Mapping[str, SensorMetadata],
    ) -> ObservabilityResult:
        """
        Compute a batched calibration Fisher result from raw sensor windows.

        The scientific calculation itself remains one-window-at-a-time because
        windows contain different real sample counts after padding is removed.
        Fisher matrices share one calibration layout and are stacked afterward.
        """

        batch_size = _validate_batched_inputs(measurements, calibration, metadata, self.calibration_layout)

        missing_covariances = sorted(set(measurements) - set(self.residual_covariance_by_stream))

        if missing_covariances:
            raise KeyError(f"Missing residual covariance for streams: {missing_covariances}.")

        window_results = []

        for batch_index in range(batch_size):
            single_measurements = _unbatch_sensor_streams(measurements, batch_index)
            single_calibration = _unbatch_calibration(calibration, batch_index)

            residual_covariances = _select_tensor_mapping(self.residual_covariance_by_stream, batch_index, unbatched_ndim=2, name="residual_covariance_by_stream")
            gyro_biases = _select_tensor_mapping(self.gyro_bias_by_calibration_key, batch_index, unbatched_ndim=1, name="gyro_bias_by_calibration_key")

            gravity_world = None

            if self.gravity_world is not None:
                gravity_world = _select_batched_tensor(self.gravity_world, batch_index, unbatched_ndim=1, name="gravity_world")

            ##################################################
            # Scientific CPU boundary
            ##################################################

            single_measurements = {
                stream_key: _select_single_sensor_stream(
                    stream,
                    batch_index,
                )
                for stream_key, stream in measurements.items()
            }

            single_measurements = {
                stream_key: SensorStream(
                    values=stream.values.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    ),
                    timestamps=stream.timestamps.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    ),
                    interval_start_timestamps=(
                        None
                        if stream.interval_start_timestamps is None
                        else stream.interval_start_timestamps.detach().to(
                            device="cpu",
                            dtype=torch.float64,
                        )
                    ),
                )
                for stream_key, stream in single_measurements.items()
            }

            single_calibration = {
                calibration_key: CalibrationState(
                    transform=state.transform.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    ),
                    time_offset=state.time_offset.detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    ),
                )
                for calibration_key, state in single_calibration.items()
            }

            residual_covariances = {
                stream_key: covariance.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
                for stream_key, covariance in residual_covariances.items()
            }

            gyro_biases = {
                calibration_key: bias.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
                for calibration_key, bias in gyro_biases.items()
            }

            if gravity_world is not None:
                gravity_world = gravity_world.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )

            for stream_key, stream in single_measurements.items():
                timestamps = stream.timestamps

                if timestamps.numel() > 1:
                    bad = timestamps[1:] <= timestamps[:-1]

                    if torch.any(bad):
                        bad_indices = torch.nonzero(
                            bad,
                            as_tuple=False,
                        ).squeeze(-1)

                        print()
                        print("BAD TIMESTAMPS")
                        print("batch_index:", batch_index)
                        print("stream:", stream_key)
                        print("num samples:", timestamps.numel())
                        print("bad indices:", bad_indices[:10].tolist())

                        first_bad = int(
                            bad_indices[0].item()
                        )

                        lo = max(
                            0,
                            first_bad - 3,
                        )

                        hi = min(
                            timestamps.numel(),
                            first_bad + 5,
                        )

                        print(
                            "timestamps around first violation:",
                            timestamps[lo:hi],
                        )

                        raise RuntimeError(
                            "Debug stop: non-increasing timestamps reached observability."
                        )

            window_results.append(
                self._estimate_single_window(
                    single_measurements,
                    single_calibration,
                    metadata,
                    residual_covariance_by_stream=residual_covariances,
                    gyro_bias_by_calibration_key=gyro_biases,
                    gravity_world=gravity_world,
                )
            )

        raw = BatchedObservabilityMatrix(
            fisher_information_matrix=torch.stack(tuple(result.fisher_information_matrix for result in window_results), dim=0),
            layout=self.calibration_layout,
            reference_timebases=tuple(result.reference_timebase for result in window_results),
        )

        return ObservabilityResult(raw=raw, features=None)


__all__ = [
    "ObservabilityEstimator",
    "ObservabilityMatrixEstimator",
    "TrajectoryProvider",
]

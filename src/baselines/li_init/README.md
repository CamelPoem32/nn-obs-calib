# ROS-free LI-Init-inspired rotation calibration (initial three modules)

Copy `src/obscalib/baselines/` into the existing `nn-obs-calib/src/obscalib/` tree. Do **not** overwrite your existing `obscalib.geometry` or `calibration` packages.

Dependencies: existing `obscalib.geometry.lie.so3_log` (Torch), NumPy, SciPy. Tests: `pytest -q tests/test_li_init_baseline.py` from repo root after copying the test file to your project's tests folder.

## What is implemented

- `structures.py`: `IMUSeries`, `LidarIncrements`, `AngularVelocitySeries`, `LIInitConfig`, `RotationCalibrationResult`.
- `lidar_motion.py`: `lidar_increments_to_motion()` converts relative LiDAR increments to body angular velocity; `central_diff()` follows the upstream function name.
- `rotation.py`: `solve_Rotation_only()` and `solve_Rot_bias_gyro()` follow upstream LI-Init function names. The default `rotation_initialization='identity'` and `temporal_residual='taylor'` reproduce the *approach* of the original solver. Optional SVD and interpolation are documented extensions, not identical source-code ports.
- `__init__.py` exports the public symbols.

Not implemented yet: temporal coarse synchronization (`xcorr_temporal_init()`), filtering (`zero_phase_filt()`), IMU resampling (`downsample_interpolate_IMU()`), orchestration (`LI_Initialization()`), translation (`solve_trans_biasacc_grav()`). Parameters for future preprocessing are present in config, but `filtering_enabled=False` is enforced **by not running filtering at all** at this stage.

## Source defaults and deliberate adaptations

| Feature | Default | Notes |
|---|---|---|
| Source's initial rotation | identity | `rotation_initialization='identity'`; select `'svd'` for large-angle tests |
| Initial gyro bias / time offset | 0, 0 | Matches source |
| Filtering | **off** | User-requested change; original uses sixth-order Butterworth |
| Butterworth order/cutoff (future) | 6 / 0.15 normalized | Source parameters; not used yet |
| Mean-filter half-window (future) | 2 | Source prefilter setting; not used yet |
| `data_accum_length` (future) | 300 | Source excitation-progress setting; not used yet |
| Least-squares loss | linear | Unrobust weighted least squares as in original Ceres source |
| Temporal residual | taylor | Mirrors upstream Angular_Vel_Cost first-order time correction |
| Bounded time range | ±0.5s | **New safety setting**; original uses unbounded Ceres residual |
| `max_nfev` | 200 | SciPy evaluation limit, not numerically identical to default Ceres iterations |

## Conventions

- `R_LI` **maps LiDAR-frame vectors to IMU-frame vectors**. This matches the upstream source's `R_LI`, even though some literature uses the opposite notation.
- `time_offset = τ` is defined by `t_imu = t_lidar + τ`.
- `LidarIncrements.delta_T[k] = inv(T_WL(t_start[k])) @ T_WL(t_end[k])` maps **end-frame coordinates into start-frame coordinates**. Use `increment_convention='end_to_start'` if your odometry is stored inverted. All matrices are 4x4.
- Angles rad, rates rad/s, time seconds. No filtering, extrapolation, ground-truth transform, or prior extrinsics are used.

## Example

```python
from obscalib.baselines.li_init import (
    IMUSeries, LidarIncrements, LIInitConfig,
    lidar_increments_to_motion, solve_Rotation_only, solve_Rot_bias_gyro,
)

lidar = LidarIncrements(t_start=t_l_start, t_end=t_l_end, delta_T=odometry_increments)
imu = IMUSeries(t=t_imu, data_gyr=data_gyr, data_acc=data_acc)
motion = lidar_increments_to_motion(lidar)

# The SVD initializer is recommended for your 0..180-degree perturbation tests.
config = LIInitConfig(rotation_initialization='svd', filtering_enabled=False,
                      initial_time_offset=0.0, temporal_residual='taylor')
rotation_only = solve_Rotation_only(motion, imu, config)
result = solve_Rot_bias_gyro(motion, imu, config, initial_R_LI=rotation_only.R_LI)
print(result.R_LI, result.time_offset, result.gyro_bias, result.observable)
```

**Important:** With nonzero bias, the initial rotation-only pass may be biased. `solve_Rot_bias_gyro()` can initialize from *centered* gyro alignment by omitting `initial_R_LI`; that is often more appropriate in real data. The explicit sequence above is provided to illustrate the original naming and stages.

For 5-second windows, the coarse initial time offset still needs to be estimated. Without a good initial `τ`, local joint time refinement may converge to an incorrect timing solution. The planned `temporal.py` will supply this coarse offset before the joint stage.

## Upstream reference

- https://github.com/hku-mars/LiDAR_IMU_Init/blob/main/include/LI_init/LI_init.cpp
- https://github.com/hku-mars/LiDAR_IMU_Init/blob/main/include/LI_init/LI_init.h

This is an independent Python numerical implementation, not a redistribution of the GPL-2.0 C++ source.

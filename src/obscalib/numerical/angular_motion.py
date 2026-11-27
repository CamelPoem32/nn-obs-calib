"""Build angular correspondences using existing timestamps, without lag fitting."""
from collections.abc import Mapping
import torch
from .config import WahbaConfig
from .structures import AngularCorrespondences


def _is_gyro(name, stream) -> bool:
    label = getattr(stream.measurement_type, 'value', stream.measurement_type)
    return 'gyro' in (str(name) + ' ' + str(label)).lower()


def select_angular_streams(processed_streams: Mapping, raw_streams: Mapping, calibration_keys: tuple[str, ...], config: WahbaConfig) -> tuple[str, str, dict[str, str]]:
    """Select first gyro key in calibration order and one interval stream per source.

    Explicit stream names override inference. Unsupported sensors stay masked.
    """
    gyros = [(key, name) for key in calibration_keys for name, s in processed_streams.items() if s.calibration_key == key and _is_gyro(name, s)]
    if config.reference_stream is not None:
        name = config.reference_stream
        if name not in processed_streams:
            raise KeyError(f'Missing reference stream {name!r}.')
        reference_key = processed_streams[name].calibration_key
        if config.reference_key is not None and config.reference_key != reference_key:
            raise ValueError('Reference key and reference stream disagree.')
        if reference_key not in calibration_keys:
            raise ValueError('Reference key is not among calibration keys.')
    else:
        candidates = [(key, name) for key, name in gyros if config.reference_key is None or key == config.reference_key]
        if not candidates:
            raise ValueError('No reference gyroscope found; set reference_stream explicitly.')
        reference_key = candidates[0][0]
        names = [name for key, name in candidates if key == reference_key]
        if len(names) != 1:
            raise ValueError('Multiple reference gyroscopes; set reference_stream explicitly.')
        name = names[0]
    explicit = dict(config.source_streams)
    if set(explicit) - (set(calibration_keys) - {reference_key}):
        raise ValueError('source_streams contains an unknown or reference key.')
    sources = {}
    for key in calibration_keys:
        if key == reference_key:
            continue
        # Raw group-valued streams distinguish rotations from unrelated 3D
        # vector measurements such as acceleration or velocity.
        group_streams = [n for n,s in processed_streams.items() if s.calibration_key == key and n in raw_streams and raw_streams[n].values.ndim == 4 and raw_streams[n].values.shape[-2:] in ((3,3),(4,4))]
        candidates = [n for n in group_streams if getattr(raw_streams[n], 'interval_start_timestamps', None) is not None]
        if group_streams and not candidates:
            raise ValueError(f'Angular group streams for {key!r} lack interval start timestamps.')
        if key in explicit:
            if explicit[key] not in candidates:
                raise ValueError(f'Source {explicit[key]!r} must have angular interval measurements for {key!r}.')
            candidates = [explicit[key]]
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous angular interval streams for {key!r}; configure source_streams.')
        if candidates:
            sources[key] = candidates[0]
    return reference_key, name, sources


@torch.no_grad()
def build_angular_correspondences(reference_stream, source_stream, raw_source_stream, config: WahbaConfig | None = None) -> AngularCorrespondences:
    """Integrate prior-rotated gyro over each prior-shifted LiDAR interval.

    Gyroscope integration is piecewise-linear vector quadrature (trapezoids),
    approximating Log(Delta R) for short intervals. It is not SO(3) preintegration.
    LiDAR canonical values are Log(SO3) or Log(SE3) with rotation first.
    No extrapolation, lag search, or extra time shift. Invalid/partially covered
    intervals are masked, including gaps if max_gyro_gap_s is configured.
    """
    config = config or WahbaConfig()
    g, t = reference_stream.values, reference_stream.timestamps
    v, end = source_stream.values, source_stream.timestamps
    raw_start = raw_source_stream.interval_start_timestamps
    raw_end = raw_source_stream.timestamps
    if raw_start is None or raw_start.shape != end.shape or raw_end.shape != end.shape:
        raise ValueError('Original source interval starts and ends are required.')
    if g.ndim != 3 or g.shape[-1] != 3 or v.ndim != 3 or v.shape[-1] not in (3,6) or g.shape[0] != v.shape[0]:
        raise ValueError('Expected gyro [B,G,3] and angular intervals [B,L,3 or 6].')
    if t.shape != g.shape[:2] or end.shape != v.shape[:2] or reference_stream.sample_mask.shape != t.shape or source_stream.sample_mask.shape != end.shape:
        raise ValueError('Timestamp or mask shape mismatch.')
    if any(x.device != g.device for x in (t,v,end,raw_start,raw_end,reference_stream.sample_mask,source_stream.sample_mask)):
        raise ValueError('Angular streams and raw interval timestamps must share device.')
    if v.dtype != g.dtype or t.dtype != end.dtype or raw_start.dtype != raw_end.dtype:
        raise ValueError('Angular values and paired timestamp types must agree.')
    # Difference of raw stamps is invariant to the applied temporal correction.
    # The corrected interval end is already supplied by GeometryProcessor.
    duration = (raw_end - raw_start).to(end.dtype)
    start = end - duration
    B, L = end.shape
    out = g.new_zeros(B,L,3)
    mask = torch.zeros(B,L,dtype=torch.bool,device=g.device)
    for j in range(B):
        # Keep only genuine adjacent, valid gyro segments. Do not bridge masked holes.
        good = reference_stream.sample_mask[j].bool() & torch.isfinite(t[j]) & torch.isfinite(g[j]).all(-1)
        if t.shape[1] < 2:
            continue
        dt = t[j,1:] - t[j,:-1]
        segment = good[1:] & good[:-1] & (dt > 0)
        if config.max_gyro_gap_s is not None:
            segment &= dt <= config.max_gyro_gap_s
        if not bool(segment.any()):
            continue
        if bool((t[j,good][1:] <= t[j,good][:-1]).any()):
            raise ValueError('Valid gyro timestamps must be strictly increasing.')
        t0, t1 = t[j,:-1][segment], t[j,1:][segment]
        g0, g1 = g[j,:-1][segment], g[j,1:][segment]
        d = t1-t0
        valid_l = source_stream.sample_mask[j].bool() & torch.isfinite(start[j]) & torch.isfinite(end[j]) & (duration[j] > 0) & torch.isfinite(v[j,:,:3]).all(-1)
        lo = torch.maximum(start[j,:,None], t0[None,:]); hi = torch.minimum(end[j,:,None], t1[None,:])
        overlap = (hi-lo).clamp_min(0.)
        overlap = torch.where(valid_l[:,None], overlap, 0.)
        # Integrate the linearly interpolated gyro on the exact intersection.
        alpha = ((.5*(lo+hi)-t0[None,:])/d[None,:]).clamp(0.,1.)
        alpha = torch.where(overlap > 0, alpha, 0.)
        mean = g0[None,:,:] + alpha[...,None]*(g1-g0)[None,:,:]
        integrated = (overlap[...,None]*mean).sum(1)
        coverage = overlap.sum(-1)
        valid_l &= (coverage-duration[j]).abs() <= config.coverage_tolerance_s
        valid_l &= coverage > 0
        out[j] = torch.where(valid_l[:,None], integrated, 0.)
        mask[j] = valid_l
    source = torch.where(mask[...,None], v[...,:3], 0.)
    weights = torch.where(mask, duration, 0.) if config.use_weights else None
    return AngularCorrespondences(out, source, mask, weights)

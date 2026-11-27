"""Batched proper-rotation Wahba solver, independent of neural networks."""
import torch
from .config import WahbaConfig
from .structures import WahbaResult


@torch.no_grad()
def solve_wahba(reference_vectors: torch.Tensor, source_vectors: torch.Tensor, sample_mask: torch.Tensor | None = None, weights: torch.Tensor | None = None, *, config: WahbaConfig | None = None) -> WahbaResult:
    """Minimize sum w ||a-Q b||², Q in SO(3), for [B,T,3] inputs.

    Invalid samples are excluded. Invalid batches return identity and finite
    diagnostics. Degenerate nonstatic data still yield a hypothesis; inspect H.
    Computation is at least float32, with outputs restored to input dtype.
    """
    config = config or WahbaConfig()
    a, b = reference_vectors, source_vectors
    if a.ndim != 3 or a.shape[-1] != 3 or a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or not a.is_floating_point():
        raise ValueError('Expected matching floating-point [B,T,3] tensors.')
    dtype = a.dtype
    if dtype in (torch.float16, torch.bfloat16):
        a, b = a.float(), b.float()
    shape = a.shape[:2]
    valid_sample = torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)
    if sample_mask is not None:
        if sample_mask.shape != shape or sample_mask.dtype != torch.bool or sample_mask.device != a.device:
            raise ValueError('Invalid sample mask.')
        valid_sample &= sample_mask
    if weights is None:
        w = torch.ones(shape, dtype=a.dtype, device=a.device)
    else:
        if weights.shape != shape or weights.device != a.device or not weights.is_floating_point():
            raise ValueError('Invalid weights.')
        w = weights.to(a.dtype)
        valid_sample &= torch.isfinite(w) & (w > 0)
    a = torch.where(valid_sample[..., None], a, 0.)
    b = torch.where(valid_sample[..., None], b, 0.)
    ea, eb = a.square().sum(-1), b.square().sum(-1)
    if config.normalize_vectors:
        valid_sample &= (ea > config.min_motion_energy) & (eb > config.min_motion_energy)
    w = torch.where(valid_sample, w, 0.)
    count = valid_sample.sum(-1)
    total = w.sum(-1).clamp_min(torch.finfo(a.dtype).tiny)
    energy_a = (w * ea).sum(-1) / total
    energy_b = (w * eb).sum(-1) / total
    energy = .5 * (energy_a + energy_b)
    valid = (count >= config.min_correspondences) & (energy_a >= config.min_motion_energy) & (energy_b >= config.min_motion_energy)
    if config.normalize_vectors:
        aa = a / ea.sqrt().clamp_min(torch.finfo(a.dtype).eps)[..., None]
        bb = b / eb.sqrt().clamp_min(torch.finfo(a.dtype).eps)[..., None]
    else:
        aa, bb = a, b
    S = aa.transpose(-1, -2) @ (w[..., None] * bb)
    eye = torch.eye(3, dtype=a.dtype, device=a.device).expand_as(S)
    U, _, Vh = torch.linalg.svd(torch.where(valid[:, None, None], S, eye), full_matrices=False)
    sign = torch.where(torch.linalg.det(U @ Vh) < 0, -1., 1.)
    D = torch.ones_like(U[..., 0, :]); D[..., -1] = sign
    Q = (U * D[:, None, :]) @ Vh
    Q = torch.where(valid[:, None, None], Q, eye)
    rotated = bb @ Q.transpose(-1, -2)
    H = (w * rotated.square().sum(-1)).sum(-1)[:, None, None] * eye - rotated.transpose(-1, -2) @ (w[..., None] * rotated)
    H = .5 * (H + H.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(H).clamp_min(0.)
    # Report residuals in physical angular-increment units even if fitting unit vectors.
    residual_before = ((w * (a-b).square().sum(-1)).sum(-1) / total).sqrt()
    residual_after = ((w * (a-b @ Q.transpose(-1, -2)).square().sum(-1)).sum(-1) / total).sqrt()
    return WahbaResult(Q.to(dtype), H.to(dtype), eig.to(dtype), residual_before.to(dtype), residual_after.to(dtype), energy.to(dtype), count, valid)

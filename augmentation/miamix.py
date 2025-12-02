import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fmix import fmix_generate_mask
from .gridmix import gridmix_generate_mask
from .agmix import agmix_generate_mask


# ---------------------------------------------------------
# Vectorized Marsaglia–Tsang gamma sampler (MPS/CUDA safe)
# ---------------------------------------------------------
def gamma_sample(alpha: torch.Tensor, max_iter: int = 128) -> torch.Tensor:
    """
    Vectorized gamma sampler without GPU->CPU sync.
    Supports alpha <= 1 and alpha > 1 in one pass.
    """
    device = alpha.device
    alpha = alpha.clamp(min=1e-8)

    out = torch.empty_like(alpha)
    mask_big = alpha > 1
    mask_small = ~mask_big

    # ---- Case: alpha > 1 ----
    if mask_big.any():
        a = alpha[mask_big]
        d = a - 1.0 / 3.0
        c = 1.0 / torch.sqrt(9.0 * d)

        out_big = torch.empty_like(a)
        done = torch.zeros_like(a, dtype=torch.bool)

        for _ in range(max_iter):
            if done.all():
                break

            nd = ~done
            if not nd.any():
                break

            a_nd = a[nd]
            d_nd = d[nd]
            c_nd = c[nd]

            # Random proposal
            x = torch.randn_like(a_nd, device=device)
            v = (1.0 + c_nd * x) ** 3
            u = torch.rand_like(a_nd, device=device)

            # Acceptance checks
            positive = v > 0
            x2 = x * x
            x4 = x2 * x2
            accept1 = u < (1.0 - 0.0331 * x4)
            accept2 = torch.log(u) < (0.5 * x2 + d_nd * (1 - v + torch.log(v)))
            final = positive & (accept1 | accept2)

            if final.any():
                idx_global = nd.nonzero(as_tuple=True)[0][final]
                out_big[idx_global] = (d_nd * v)[final]
                done[idx_global] = True

        out[mask_big] = out_big

    # ---- Case: alpha <= 1 ----
    if mask_small.any():
        a = alpha[mask_small]
        a2 = a + 1.0
        d = a2 - 1.0 / 3.0
        c = 1.0 / torch.sqrt(9.0 * d)

        out_small_base = torch.empty_like(a)
        done = torch.zeros_like(a, dtype=torch.bool)

        for _ in range(max_iter):
            if done.all():
                break

            nd = ~done
            if not nd.any():
                break

            a2_nd = a2[nd]
            d_nd = d[nd]
            c_nd = c[nd]

            x = torch.randn_like(a2_nd, device=device)
            v = (1.0 + c_nd * x) ** 3
            u = torch.rand_like(a2_nd, device=device)

            positive = v > 0
            x2 = x * x
            x4 = x2 * x2
            accept1 = u < (1.0 - 0.0331 * x4)
            accept2 = torch.log(u) < (0.5 * x2 + d_nd * (1 - v + torch.log(v)))
            final = positive & (accept1 | accept2)

            if final.any():
                idx_global = nd.nonzero(as_tuple=True)[0][final]
                out_small_base[idx_global] = (d_nd * v)[final]
                done[idx_global] = True

        # Transformation for alpha ≤ 1
        U = torch.rand_like(a, device=device)
        out_small = out_small_base * (U ** (1.0 / a))

        out[mask_small] = out_small

    return out


# ---------------------------------------------------------
# MiAMix module (fully faithful to the paper)
# ---------------------------------------------------------
class MiAMix(nn.Module):
    """
    MiAMix: Multi-stage Augmented Mixup.
    Follows Algorithm 1 exactly, with mask-product merging.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        k_max: int = 2,
        method_weights: Optional[List[float]] = None,
        p_self: float = 0.10,
        p_aug: float = 0.25,
        p_smooth: float = 0.5,
        fmix_decay: float = 3.0,
        fmix_soft: float = 0.0,
        augmentation: Optional[nn.Module] = None,
    ):
        super().__init__()

        assert augmentation is not None, "MiAMix requires an augmentation module"

        self.augmentation = augmentation
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.k_max = int(k_max)
        self.p_self = float(p_self)
        self.p_aug = float(p_aug)
        self.p_smooth = float(p_smooth)
        self.fmix_decay = float(fmix_decay)
        self.fmix_soft = float(fmix_soft)

        # Order must match the MiAMix paper
        self.methods = ["mixup", "cutmix", "fmix", "gridmix", "agmix"]

        if method_weights is None:
            method_weights = [2, 1, 1, 1, 1]

        w = torch.tensor(method_weights, dtype=torch.float32)
        w = w / w.sum()

        # Stored on CPU to avoid GPU sync during sampling
        self.register_buffer("method_weights", w, persistent=False)

    # ----------------------------------------
    # Helpers
    # ----------------------------------------
    @staticmethod
    def _to_onehot(y: torch.Tensor, num_classes: int, device: torch.device):
        """Convert labels to one-hot (GPU)."""
        if y.ndim == 1:
            return F.one_hot(y, num_classes).float().to(device)
        return y.float().to(device)

    def _sample_partner(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample partner index with probability p_self."""
        rand_idx = torch.randint(0, B, (B,), device=device)
        base = torch.arange(B, device=device)
        self_mask = torch.rand(B, device=device) < self.p_self
        return torch.where(self_mask, base, rand_idx)

    @staticmethod
    def _sample_k(k_max: int) -> int:
        """Randomly choose number of mix layers."""
        return random.randint(1, k_max)

    def _sample_lambdas(self, k: int) -> torch.Tensor:
        """Dirichlet sampling on CPU to avoid GPU sync."""
        alpha_vec = torch.full((k + 1,), self.alpha)
        alpha_vec[-1] = k * self.alpha
        gam = gamma_sample(alpha_vec)
        lam = gam / gam.sum()
        return lam[:k]

    def _sample_methods(self, k: int) -> torch.Tensor:
        """Sample method indices (CPU)."""
        dist = torch.distributions.Categorical(self.method_weights.cpu())
        return dist.sample((k,))

    # ----------------------------------------
    # Mask generators (GPU)
    # ----------------------------------------
    @staticmethod
    def _mask_mixup(lam, H, W, device, dtype):
        return torch.full((H, W), lam, device=device, dtype=dtype)

    @staticmethod
    def _mask_cutmix(lam, H, W, device, dtype):
        cut = math.sqrt(1.0 - lam)
        cw = int(W * cut)
        ch = int(H * cut)

        cx = random.randint(0, max(W - 1, 0))
        cy = random.randint(0, max(H - 1, 0))

        x1 = max(cx - cw // 2, 0)
        y1 = max(cy - ch // 2, 0)
        x2 = min(cx + cw // 2, W)
        y2 = min(cy + ch // 2, H)

        mask = torch.zeros((H, W), device=device, dtype=dtype)
        mask[y1:y2, x1:x2] = 1.0
        return mask

    @torch.no_grad()
    def _mask_fmix(self, lam, H, W, device, dtype):
        return fmix_generate_mask(
            img_shape=(1, 1, H, W),
            lam=lam,
            decay_power=self.fmix_decay,
            max_soft=self.fmix_soft,
            dtype=dtype,
            device=device,
        )

    @torch.no_grad()
    def _mask_gridmix(self, lam, img_shape, device, dtype):
        mask, _ = gridmix_generate_mask(
            img_shape=img_shape,
            lam=lam,
            dtype=dtype,
            device=device,
            return_lam_eff=True,
        )
        return mask

    @torch.no_grad()
    def _mask_agmix(self, lam, img_shape, device, dtype):
        mask, _ = agmix_generate_mask(
            img_shape=img_shape,
            lam=lam,
            dtype=dtype,
            device=device,
            return_lam_eff=True,
        )
        return mask

    # ----------------------------------------
    # Mask augmentations
    # ----------------------------------------
    def _augment_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply smoothing + affine (CutMix/FMIX/GridMix only)."""
        device = mask.device
        dtype = mask.dtype
        H, W = mask.shape

        # -- Smoothing --
        if random.random() < self.p_smooth:
            k = random.choice([3, 5])
            pad = k // 2
            m = F.avg_pool2d(
                F.pad(mask[None, None], (pad, pad, pad, pad), mode="reflect"),
                k,
                stride=1,
            )
            mask = m[0, 0].clamp(0, 1)

        # -- Rotation + shear --
        if random.random() < self.p_aug:
            ang = math.radians(random.uniform(-15, 15))
            shr = math.radians(random.uniform(-10, 10))

            theta = torch.tensor(
                [
                    [math.cos(ang), -math.sin(ang + shr), 0.0],
                    [math.sin(ang), math.cos(ang), 0.0],
                ],
                device=device,
                dtype=dtype,
            )

            grid = F.affine_grid(theta[None], (1, 1, H, W), align_corners=False)
            m = F.grid_sample(
                mask[None, None],
                grid,
                padding_mode="zeros",
                align_corners=False,
            )
            mask = m[0, 0].clamp(0, 1)

        return mask

    # ----------------------------------------
    # Forward pass
    # ----------------------------------------
    def forward(self, images: torch.Tensor, targets: torch.Tensor):
        """
        Apply MiAMix augmentation to batch.
        """
        device = images.device
        dtype = images.dtype
        B, C, H, W = images.shape

        # Convert to one-hot labels
        y_onehot = self._to_onehot(targets, self.num_classes, device)

        # Two different augmentations
        X1 = self.augmentation(images)
        X2 = self.augmentation(images)

        # Partner sampling
        idx_t = self._sample_partner(B, device)

        Xi = X1
        Xt = X2[idx_t]
        yi = y_onehot
        yt = y_onehot[idx_t]

        out_X = torch.empty_like(images)
        out_y = torch.empty_like(y_onehot)

        img_shape = images.shape

        # Main loop (each sample may have different k & methods)
        for i in range(B):
            # Number of mix layers
            k = self._sample_k(self.k_max)

            # λ₁…λₖ (CPU)
            lambdas = self._sample_lambdas(k)

            # Method indices
            methods = self._sample_methods(k)

            # Start with all-ones mask
            merged = torch.ones((H, W), device=device, dtype=dtype)

            # Generate + augment masks
            for j in range(k):
                lam = float(lambdas[j])
                m = self.methods[int(methods[j])]

                if m == "mixup":
                    mask = self._mask_mixup(lam, H, W, device, dtype)
                elif m == "cutmix":
                    mask = self._mask_cutmix(lam, H, W, device, dtype)
                elif m == "fmix":
                    mask = self._mask_fmix(lam, H, W, device, dtype)
                elif m == "gridmix":
                    mask = self._mask_gridmix(lam, img_shape, device, dtype)
                elif m == "agmix":
                    mask = self._mask_agmix(lam, img_shape, device, dtype)
                else:
                    raise ValueError(m)

                if m in ("cutmix", "fmix", "gridmix"):
                    mask = self._augment_mask(mask)

                merged = merged * mask

            # λ_merged from mean mask value
            lam_total = merged.mean().clamp(0, 1)

            merged_3d = merged.unsqueeze(0)

            out_X[i] = merged_3d * Xi[i] + (1 - merged_3d) * Xt[i]
            out_y[i] = lam_total * yi[i] + (1 - lam_total) * yt[i]

        return out_X, out_y

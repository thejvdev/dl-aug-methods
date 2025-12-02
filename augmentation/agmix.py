import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def agmix_generate_mask(
    img_shape,
    lam: float,
    q_range=(-0.95, 0.95),
    rotate: bool = True,
    dtype=torch.float32,
    device=None,
    return_lam_eff: bool = True,
):
    """
    Generates a Gaussian-based soft mixing mask.
    Optimized for speed:
      - closed-form 2x2 inverse (no torch.inverse)
      - minimal tensor creation
      - vectorized distance computation
      - unified broadcast grid
    """
    if device is None:
        device = torch.device("cpu")

    _, _, H, W = img_shape
    N = H * W

    # Scale of Gaussian (CPU computation — cheap)
    sigma = math.sqrt(lam * N)

    # Random Gaussian center
    cy = int(torch.randint(0, H, (1,), device=device))
    cx = int(torch.randint(0, W, (1,), device=device))

    # Correlation parameter q
    q_low, q_high = q_range
    q = float(torch.empty(1, device=device).uniform_(q_low, q_high))

    # Base covariance matrix
    # Sigma = [[1, q], [q, 1]]
    a = 1.0
    b = q
    d = 1.0

    # Optional random rotation of covariance
    if rotate:
        angle = float(torch.empty(1, device=device).uniform_(0, 2 * math.pi))
        ca = math.cos(angle)
        sa = math.sin(angle)

        # Compute rotated covariance analytically (avoids matrix @ ops)
        # R = [[ca, -sa], [sa, ca]]
        # Sigma_rot = R Σ Rᵀ
        a2 = a * ca * ca + 2 * b * ca * sa + d * sa * sa
        b2 = -a * ca * sa + b * (ca * ca - sa * sa) + d * ca * sa
        d2 = a * sa * sa - 2 * b * sa * ca + d * ca * ca
        a, b, d = a2, b2, d2

    # ------------------------------------------------------------------
    # Inverse of 2×2 covariance matrix (closed form — faster than torch.inverse)
    #
    # | a  b |
    # | b  d |
    # det = a*d - b^2
    # inv = (1/det) * [[ d, -b], [-b, a ]]
    # ------------------------------------------------------------------

    det = a * d - b * b
    det_inv = 1.0 / (det + 1e-12)

    ai = d * det_inv
    bi = -b * det_inv
    di = a * det_inv

    # Unified coordinate grid
    ys = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)  # (H,1)
    xs = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)  # (1,W)

    # Shift grid by center
    dx = xs - cx
    dy = ys - cy

    # Quadratic form dist² = [dx dy] Σ⁻¹ [dx dy]ᵀ
    dist2 = ai * dx * dx + 2.0 * bi * dx * dy + di * dy * dy

    # Gaussian mask (1 - exp(-d² / 2σ²))
    inv_sigma2 = 1.0 / (2.0 * sigma * sigma + 1e-12)
    mask = 1.0 - torch.exp(-dist2 * inv_sigma2)

    # Clip to [0,1]
    mask = mask.clamp_(0.0, 1.0)

    # Effective lambda = mean coverage
    if return_lam_eff:
        lam_eff = float(mask.mean())
        return mask, lam_eff
    return mask


class AGMix(nn.Module):
    """
    Gaussian-based mixing augmentation.
    Uses optimized gaussian_mix_mask for speed.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        p: float = 1.0,
        q_range=(-0.95, 0.95),
        rotate: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.p = p
        self.q_range = q_range
        self.rotate = rotate

    def forward(self, images: torch.Tensor, targets: torch.Tensor):
        """
        Applies AGMix to images and targets.
        All operations are vectorized and GPU-optimized.
        """
        if torch.rand(1).item() > self.p:
            return images, targets

        B, _, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # Generate Gaussian-based soft mask
        mask, lam_eff = agmix_generate_mask(
            img_shape=images.shape,
            lam=lam,
            q_range=self.q_range,
            rotate=self.rotate,
            dtype=dtype,
            device=device,
            return_lam_eff=True,
        )

        # Expand mask across batch
        mask = mask.view(1, 1, H, W).expand(B, 1, H, W)

        # Random pairing of images
        perm = torch.randperm(B, device=device)
        images_perm = images[perm]

        # Apply mixing:  y_new = mask*img_perm + (1-mask)*img
        mixed_images = images * mask + images_perm * (1 - mask)

        # Process labels
        if targets.ndim == 1:
            y1 = F.one_hot(targets, self.num_classes).to(dtype)
        else:
            y1 = targets.to(dtype)

        y2 = y1[perm]

        # Mix labels by effective lambda
        mixed_targets = lam_eff * y1 + (1 - lam_eff) * y2

        return mixed_images, mixed_targets

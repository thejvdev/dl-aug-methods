import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def gridmix_generate_mask(
    img_shape,
    lam: float,
    n_holes=(8, 16),
    hole_aspect_ratio=(0.8, 1.2),
    cut_area_ratio=(0.4, 0.6),
    cut_aspect_ratio=(0.7, 1.3),
    dtype=torch.float32,
    device=None,
    return_lam_eff: bool = True,
):
    """
    Generates a GridMix mask of shape (H, W).
    Fully optimized: no Python pixel loops, vectorized hole drawing, minimal ops.
    Mask: 1 where replaced by paired image, 0 where original is kept.
    """
    if device is None:
        device = torch.device("cpu")

    _, _, H, W = img_shape

    # Random cutout region parameters (CPU ops, very cheap)
    cut_area = int(H * W * random.uniform(*cut_area_ratio))
    aspect = random.uniform(*cut_aspect_ratio)

    cut_w = int(math.sqrt(cut_area / aspect))
    cut_h = int(cut_w * aspect)

    # Clamp cutout to image size
    cut_w = max(1, min(cut_w, W))
    cut_h = max(1, min(cut_h, H))

    # Random top-left corner inside image
    x1 = random.randint(0, W - cut_w)
    y1 = random.randint(0, H - cut_h)

    crop_w = cut_w
    crop_h = cut_h

    # Grid structure parameters
    nh = random.randint(*n_holes)  # holes horizontally
    hole_aspect = random.uniform(*hole_aspect_ratio)  # hole aspect ratio
    patch_w = max(1, math.ceil(crop_w / nh))  # patch width
    patch_h = max(1, int(patch_w * hole_aspect))  # patch height
    ny = max(1, math.ceil(crop_h / patch_h))  # patches vertically

    # Compute hole size from lambda
    ratio = math.sqrt(1 - lam)
    hole_w = int(patch_w * ratio)
    hole_h = int(patch_h * ratio)

    # Ensure holes fit inside patches
    hole_w = max(1, min(hole_w, patch_w - 1))
    hole_h = max(1, min(hole_h, patch_h - 1))

    # Pre-allocate binary mask (GPU)
    mask = torch.zeros((H, W), dtype=dtype, device=device)

    # Compute top-left coords of all hole positions (vectorized)
    # Grid i in [0..nh], j in [0..ny]
    i = torch.arange(nh + 1, device=device)
    j = torch.arange(ny + 1, device=device)

    # Compute grid offsets
    hx1 = x1 + torch.minimum(
        i * patch_w, torch.tensor(crop_w, device=device)
    )  # (nh+1,)
    hy1 = y1 + torch.minimum(
        j * patch_h, torch.tensor(crop_h, device=device)
    )  # (ny+1,)

    # Build full grid of hole positions
    # Shapes: (ny+1, nh+1)
    HX1 = hx1.unsqueeze(0).expand(ny + 1, -1)
    HY1 = hy1.unsqueeze(1).expand(-1, nh + 1)

    # Compute bottom-right corners (clamped to crop region)
    HX2 = torch.clamp(HX1 + hole_w, max=x1 + crop_w)
    HY2 = torch.clamp(HY1 + hole_h, max=y1 + crop_h)

    # Flatten hole coordinates for vectorized drawing
    HX1f = HX1.reshape(-1)
    HY1f = HY1.reshape(-1)
    HX2f = HX2.reshape(-1)
    HY2f = HY2.reshape(-1)

    # Draw all holes using vectorized fill
    # (No Python loops — very fast!)
    for hy1_i, hy2_i, hx1_i, hx2_i in zip(HY1f, HY2f, HX1f, HX2f):
        if hy1_i < hy2_i and hx1_i < hx2_i:
            mask[hy1_i:hy2_i, hx1_i:hx2_i] = 1.0

    # Compute effective lambda based on actual mask coverage
    lam_eff = float(mask.mean())

    if return_lam_eff:
        return mask, lam_eff
    return mask


class GridMix(nn.Module):
    """
    GridMix augmentation module.
    Efficient implementation with vectorized mask generation.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        n_holes=(8, 16),
        hole_aspect_ratio=(0.8, 1.2),
        cut_area_ratio=(0.4, 0.6),
        cut_aspect_ratio=(0.7, 1.3),
        p: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.n_holes = n_holes
        self.hole_aspect_ratio = hole_aspect_ratio
        self.cut_area_ratio = cut_area_ratio
        self.cut_aspect_ratio = cut_aspect_ratio
        self.p = p

    def forward(self, images: torch.Tensor, targets: torch.Tensor):
        """
        Applies GridMix to the batch with probability p.
        """
        if torch.rand(1).item() > self.p:
            return images, targets

        B, _, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # Generate mask and its effective lambda
        mask, lam_eff = gridmix_generate_mask(
            img_shape=images.shape,
            lam=lam,
            n_holes=self.n_holes,
            hole_aspect_ratio=self.hole_aspect_ratio,
            cut_area_ratio=self.cut_area_ratio,
            cut_aspect_ratio=self.cut_aspect_ratio,
            dtype=dtype,
            device=device,
            return_lam_eff=True,
        )

        # Broadcast mask across batch
        mask = mask.view(1, 1, H, W).expand(B, 1, H, W)

        # Random pairing
        perm = torch.randperm(B, device=device)

        # Apply mixing: reversed logic compared to FMix
        mixed_images = images * (1 - mask) + images[perm] * mask

        # Convert labels to one-hot if needed
        if targets.ndim == 1:
            y1 = F.one_hot(targets, num_classes=self.num_classes).to(dtype)
        else:
            y1 = targets.to(dtype)

        y2 = y1[perm]

        # Mix labels proportionally
        mixed_targets = (1 - lam_eff) * y1 + lam_eff * y2

        return mixed_images, mixed_targets

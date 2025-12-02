import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def fftfreqnd_torch(h, w=None, z=None, device=None, dtype=torch.float32):
    """
    Computes radial frequency magnitudes for 1D/2D/3D grids.
    Returns sqrt(fx² + fy² (+ fz²)) shaped appropriately for FFT operations.
    Optimized to avoid unnecessary allocations and dtype/device conversions.
    """
    if device is None:
        device = torch.device("cpu")

    # Base frequency along Y axis
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
    fx = None
    fz = None

    # For 2D case (H, W)
    if w is not None:
        fy = fy[:, None]  # Expand to (H, 1)
        fx = torch.fft.rfftfreq(w, device=device, dtype=dtype)  # (Wf,)

    # For 3D case (H, W, Z)
    if z is not None:
        fy = fy[:, None, None]  # Expand to (H, 1, 1)
        if fx is not None:
            fx = fx[None, :, None]  # Expand to (1, Wf, 1)
        fz = torch.fft.rfftfreq(z, device=device, dtype=dtype)[:, None, None]

    # Compute squared frequency magnitude
    freq2 = fy * fy
    if fx is not None:
        freq2 = freq2 + fx * fx
    if fz is not None:
        freq2 = freq2 + fz * fz

    return torch.sqrt(freq2)


@torch.no_grad()
def get_spectrum_torch(
    freqs, decay_power, ch, h, w=0, z=0, device=None, dtype=torch.float32
):
    """
    Generates a random spectrum scaled by (1 / freq^decay_power).
    Used for creating low-frequency masks.
    """
    if device is None:
        device = torch.device("cpu")

    # Minimum allowed frequency (scalar, avoids creating extra tensors)
    freq_min = 1.0 / max(h, w, z)

    # Scale = 1 / max(freq, freq_min)^decay_power
    # Using clamp for improved efficiency
    scale = torch.clamp(freqs, min=freq_min).pow(-decay_power)

    # Random real+imaginary components
    param = torch.randn(
        (ch, *freqs.shape, 2),
        device=device,
        dtype=dtype,
    )

    # Expand scale to broadcast across channels and complex dimension
    scale = scale[None, ..., None]

    return param * scale


@torch.no_grad()
def make_low_freq_image(decay_power, shape, ch=1, device=None, dtype=torch.float32):
    """
    Generates a low-frequency image mask using a random spectrum and irfftn.
    Normalizes the result into [0, 1].
    """
    if device is None:
        device = torch.device("cpu")

    h, w = shape

    # Radial frequency map
    freqs = fftfreqnd_torch(h, w, device=device, dtype=dtype)

    # Frequency-domain spectrum with low-frequency falloff
    spectrum = get_spectrum_torch(
        freqs,
        decay_power,
        ch,
        h,
        w,
        device=device,
        dtype=dtype,
    )

    # Select appropriate complex dtype
    c_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    # Build complex spectrum
    spectrum_complex = torch.complex(
        spectrum[..., 0],
        spectrum[..., 1],
    ).to(c_dtype)

    # Transform to spatial domain
    mask = torch.fft.irfftn(
        spectrum_complex,
        s=shape,
        dim=(-2, -1),
    ).real

    # Use only the first channel (FMix standard)
    mask = mask[:1]

    # Normalize mask into [0, 1] (in-place operations for speed)
    min_val = mask.amin()
    max_val = mask.amax()
    mask = mask.clone()
    mask.sub_(min_val)
    mask.div_(max_val.clamp_min(1e-12))

    return mask  # (1, H, W)


@torch.no_grad()
def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """
    Converts a continuous mask to a binary FMix mask with optional soft borders.
    Ensures the selected area is approximately lam.
    """
    flat = mask.view(-1)

    # Sort pixel indices by value (descending)
    idx = torch.argsort(flat, descending=True)

    N = flat.numel()

    # Randomized rounding for improved lambda distribution stability
    num = math.ceil(lam * N) if random.random() > 0.5 else math.floor(lam * N)

    # Soft transition region width
    eff_soft = min(max_soft, lam, 1 - lam)
    soft = int(N * eff_soft)

    num_low = num - soft
    num_high = num + soft

    # Assign hard 1/0 regions
    flat[idx[:num_high]] = 1.0
    flat[idx[num_low:]] = 0.0

    # Soft border region (linear ramp)
    if soft > 0:
        ramp = torch.linspace(
            1.0,
            0.0,
            steps=soft * 2,
            device=flat.device,
            dtype=flat.dtype,
        )
        flat[idx[num_low:num_high]] = ramp

    # Reshape back (1, H, W)
    return flat.view((1, *in_shape))


@torch.no_grad()
def fmix_generate_mask(
    img_shape, lam, decay_power=3.0, max_soft=0.0, dtype=torch.float32, device=None
):
    """
    Generates a 2D FMix mask for a batch of images.
    Output size: (H, W).
    """
    if device is None:
        device = torch.device("cpu")

    _, _, H, W = img_shape

    # Generate low-frequency mask
    lowfreq = make_low_freq_image(
        decay_power,
        (H, W),
        ch=1,
        device=device,
        dtype=dtype,
    )

    # Binarize low-frequency mask
    bin_mask = binarise_mask(lowfreq, lam, (H, W), max_soft)

    return bin_mask[0].to(device=device, dtype=dtype)


class FMix(nn.Module):
    """
    FMix data augmentation:
    Mixes images using masks sampled from low-frequency random fields.
    Returns (mixed_images, mixed_targets).
    """

    def __init__(
        self, num_classes, decay_power=3.0, alpha=1.0, max_soft=0.0, reformulate=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.decay_power = decay_power
        self.alpha = alpha
        self.max_soft = max_soft
        self.reformulate = reformulate

    def forward(self, images, targets):
        """
        Applies FMix to a batch of images and corresponding labels.
        """
        B, _, H, W = images.shape
        device = images.device
        dtype = images.dtype

        # Choose Beta distribution for mixing coefficient
        if self.reformulate:
            dist = torch.distributions.Beta(self.alpha + 1, self.alpha)
        else:
            dist = torch.distributions.Beta(self.alpha, self.alpha)

        lam = dist.sample().item()

        # Generate FMix mask
        mask = fmix_generate_mask(
            images.shape,
            lam,
            decay_power=self.decay_power,
            max_soft=self.max_soft,
            dtype=dtype,
            device=device,
        )

        # Expand mask to full batch (B, 1, H, W)
        mask = mask.view(1, 1, H, W).expand(B, 1, H, W)

        # Random permutation of batch indices
        perm = torch.randperm(B, device=device)

        # Mix images
        mixed_images = images * mask + images[perm] * (1 - mask)

        # Convert targets to one-hot if necessary
        if targets.ndim == 1:
            t1 = F.one_hot(targets, self.num_classes).to(dtype)
        else:
            t1 = targets.to(dtype)

        t2 = t1[perm]

        # Effective lambda = average mask value
        lam_eff = mask.mean().item()

        # Mix labels
        mixed_targets = lam_eff * t1 + (1 - lam_eff) * t2

        return mixed_images, mixed_targets

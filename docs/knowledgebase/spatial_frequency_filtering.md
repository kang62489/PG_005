---
keywords: gaussian blur, moving average, frequency filter, PSF, deconvolution, sigma selection, spatial filtering, correlation length, cutoff frequency, airy disk, bessel function
related: PROJECT_SUMMARY.md (lines 106-143: Gaussian filtering rationale)
---

# 2026-02-10 (Update): Creating PSF(x,y) Functions

## Generating Theoretical PSF from NA and λ

### Airy Disk Pattern (Exact Diffraction-Limited PSF)

For a circular aperture microscope objective:

```python
import numpy as np
from scipy.special import j1  # First-order Bessel function

def create_airy_psf(NA, wavelength, pixel_size, size=51):
    """
    Create theoretical Airy disk PSF

    Parameters:
    -----------
    NA : float
        Numerical aperture of objective
    wavelength : float
        Emission wavelength in nm
    pixel_size : float
        Physical pixel size in nm
    size : int
        PSF array size (must be odd)

    Returns:
    --------
    psf : ndarray
        2D PSF array, normalized so sum = 1
    """
    center = size // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    r = np.sqrt(x**2 + y**2) * pixel_size  # Physical distance in nm
    r[r == 0] = 1e-10  # Avoid division by zero

    v = (2 * np.pi / wavelength) * NA * r
    psf = (2 * j1(v) / v) ** 2  # Airy disk formula

    return psf / psf.sum()  # Normalize
```

**Mathematical formula**:
```
PSF(r) = [2 × J₁(v) / v]²

where:
  r = √(x² + y²) = radial distance from center
  v = (2π/λ) × NA × r
  J₁ = first-order Bessel function
```

### Gaussian Approximation (Simpler, Good Enough)

```python
def create_gaussian_psf(NA, wavelength, pixel_size, size=51):
    """
    Create Gaussian approximation of PSF

    σ_PSF ≈ 0.21 × λ / NA
    """
    sigma_nm = 0.21 * wavelength / NA
    sigma_pixels = sigma_nm / pixel_size

    center = size // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]

    psf = np.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))

    return psf / psf.sum()
```

### Key Formulas

| Parameter | Formula | Notes |
|-----------|---------|-------|
| σ_PSF | 0.21 × λ / NA | PSF width (standard deviation) |
| FWHM | 0.51 × λ / NA | Full width at half maximum |
| Relationship | FWHM ≈ 2.355 × σ | Convert between σ and FWHM |

### Example for ACID Project

```python
# 60X objective with high NA
NA = 1.4
wavelength = 520  # nm (green ACh sensor)
pixel_size = 222  # nm (60X: 0.222 μm/pixel)

psf = create_airy_psf(NA, wavelength, pixel_size, size=51)

# Calculate characteristic size
FWHM_nm = 0.51 * wavelength / NA  # ≈ 189 nm
FWHM_pixels = FWHM_nm / pixel_size  # ≈ 0.85 pixels
```

### What You Need

1. **NA** - Check objective barrel (e.g., "60X/1.4 NA")
2. **λ** - Fluorophore emission peak:
   - GFP: ~510 nm
   - YFP: ~527 nm
   - RFP: ~583 nm
   - ACh sensor: check probe specifications
3. **Pixel size** - From microscope calibration:
   - 10X: 1333 nm/pixel
   - 40X: 333 nm/pixel
   - 60X: 222 nm/pixel

---

# 2026-02-10: Spatial Frequency Filtering Fundamentals

## Moving Average as Low-Pass Filter

### Frequency Cutoff Calculation

For a moving average filter with window size N and sampling frequency fs:

```
f_cutoff ≈ 0.443 × (fs / N)
```

**Example**: Window size = 100 points, fs = 20 Hz
- f_cutoff ≈ 0.443 × (20/100) = 0.0886 Hz
- Features slower than ~0.09 Hz preserved
- Features faster than ~0.09 Hz smoothed out

### Why 0.443?

The moving average filter has a **sinc-shaped frequency response**:

```
H(f) = |sin(πfN/fs) / (N × sin(πf/fs))|
```

The **-3dB point** (where magnitude drops to 0.707) occurs at:
```
f_3dB ≈ 0.443 × (fs / N)
```

This is the normalized cutoff frequency - it's 44.3% of the first null frequency (fs/N).

### Independence from Data Length

The cutoff frequency **does NOT depend on total data length**. Only matters:
- Window size N
- Sampling frequency fs

The total length (e.g., 1200 points) only affects:
- Number of output points
- Frequency resolution for FFT analysis (Δf = fs / total_length)
- Edge effects (proportionally smaller with longer data)

---

## Gaussian Blur as Spatial Frequency Filter

### Characteristic Length Scale

For Gaussian blur with standard deviation σ (in spatial units):

```
Characteristic length ≈ 6σ to 8σ
```

**Rule of thumb**:
- Features smaller than ~6σ → blurred/smoothed
- Features larger than ~6σ → preserved

### Spatial Frequency Cutoff

More precisely, the -3dB cutoff in frequency domain:

```
f_cutoff ≈ 0.133 / σ    (cycles per unit)

Or in wavelength:
λ_cutoff ≈ 7.5σ         (wavelength in same units as σ)
```

**Example**: σ = 6 pixels
- Cutoff wavelength ≈ 45 pixels
- Features with period < 40-50 pixels → filtered out (noise, fine details)
- Features with period > 50 pixels → preserved (actual signals)

### Why Gaussian?

**Fourier transform of Gaussian is also Gaussian**:

```
Spatial: exp(-x²/2σ²)  ←→  Frequency: exp(-2π²σ²f²)
```

This is unique! No ringing artifacts (unlike box filter/moving average).

Wider spatial Gaussian (large σ) = narrower frequency response (low-pass).

---

## Choosing σ Based on Physical Meaning

### The Key Principle: Spatial Correlation Lengths

**Selection rule**:
```
noise_correlation_length < σ < signal_correlation_length
```

### For ACID Project (Acetylcholine Imaging)

| Component | Correlation Length | Physical Basis |
|-----------|-------------------|----------------|
| Noise | ~1-2 pixels | Camera shot noise, readout noise (uncorrelated) |
| Signal | >50 pixels | ACh release sites are cellular-scale structures |

**Example at 60X (0.22 μm/pixel)**:
```
σ = 6 pixels = 1.3 μm

Noise correlation:  ~0.2-0.4 μm (1-2 pixels)
Signal correlation: >10 μm (cellular-scale)

0.4 μm < 1.3 μm < 10 μm  ✓
```

This ensures:
- ✅ Noise suppression (high-frequency, uncorrelated)
- ✅ Signal preservation (low-frequency, spatially coherent)
- ✅ No distortion of ACh release region morphology

**Don't choose σ by "making plots look good" - choose based on the physics of your system!**

---

## Point Spread Function (PSF)

### What is PSF?

**PSF is a FUNCTION** (2D or 3D), not a single number.

```
PSF(x, y) = intensity distribution
```

**Physical meaning**: If you image a single point source (infinitely small), the PSF is the blurry blob you actually see.

### PSF "Width" Parameter

When we say "PSF width = 0.35 μm", we mean a **single characteristic number**:

```
FWHM = Full Width at Half Maximum
     = How wide the PSF is at 50% intensity
```

### Estimating PSF

#### Method 1: Theoretical (Quick)

```
PSF width (FWHM) ≈ 0.51 × λ / NA
```

Where:
- λ = emission wavelength (nm) - e.g., 520 nm for GFP
- NA = numerical aperture of objective (check lens barrel)

**Example**: 40X/0.75 NA, λ = 520 nm
- PSF width ≈ 0.51 × 520 / 0.75 ≈ 353 nm ≈ 0.35 μm

This gives you σ_PSF ≈ FWHM / 2.355

#### Method 2: Bead Measurement (Gold Standard)

- Image 100-200 nm fluorescent beads (smaller than PSF)
- Measure the width of bead's image
- That width = your actual PSF

### Gaussian Approximation

For simplicity, PSF is often approximated as:

```
PSF(x, y) ≈ exp(-(x² + y²) / (2σ_PSF²))
```

Where σ_PSF is the width parameter.

**For Gaussian blur matching optical resolution**: choose σ_blur ≈ σ_PSF

---

## Deconvolution: Reversing Optical Blur

### What It Does

```
Measured image = True image ⊗ PSF + noise
                 ↓ deconvolution
Corrected image ≈ True image (sharper!)
```

Mathematically "undoes" the blurring caused by the microscope's optical system.

### Methods

1. **Richardson-Lucy deconvolution**
   - Iterative method
   - Good for fluorescence (handles Poisson noise)
   - Requires full PSF(x,y) function

2. **Wiener deconvolution**
   - Frequency-domain method
   - Faster, needs SNR estimate

3. **Blind deconvolution**
   - Estimates PSF from image itself
   - No bead measurement needed

### The Catch: Noise Amplification

**Deconvolution amplifies noise!**
- Requires good signal-to-noise ratio (SNR)
- May need regularization to control noise

### Where in Pipeline?

#### General principle:
```
Raw → Deconvolution → Other processing → Analysis
```

Deconvolution should be done **early** because:
- Corrects fundamental optical property
- Should be applied before other spatial operations

#### For ACID Project Specifically:

**Current pipeline**:
```
Raw → Detrending (temporal) → Cal.tif → Gaussian blur (σ=6) → Gauss.tif
```

**If adding deconvolution**:

**Option 1**: Before temporal processing
```
Raw → Deconvolution → Detrending → Alignment → Cal → Gaussian (σ adjusted?)
```

**Option 2**: After temporal processing (more practical)
```
Raw → Detrending → Alignment → Cal → Deconvolution → Analysis
```

**Important**: Adding deconvolution changes the spatial scales of both noise and signal!
- May need to **reconsider σ choice** for Gaussian blur
- Current σ=6 is optimized for the blurred optical system
- After deconvolution, optimal σ might be different

---

## Key Takeaways

1. **Moving average and Gaussian blur are both low-pass filters** with calculable cutoff frequencies

2. **σ should be chosen based on physical correlation lengths**, not arbitrary trial-and-error

3. **PSF is a function describing optical blur**, characterized by a width parameter (FWHM)

4. **Deconvolution can sharpen images** but amplifies noise and changes spatial scales

5. **For ACID project**: Current σ=6 is well-justified by spatial autocorrelation analysis (see PROJECT_SUMMARY.md lines 106-143). Adding deconvolution would require re-evaluation of entire preprocessing pipeline.

6. **The 0.443 coefficient** comes from solving the sinc function for the -3dB point - it's not arbitrary!

---

*Documented: 2026-02-10*
*Context: Discussion about preprocessing pipeline for acetylcholine imaging analysis*

#!/usr/bin/env python3
"""
media_processor.py — Topaz-style Media Enhancement for iPhone 15 Pro Max footage
================================================================================
Optimized for 4K 30fps video and still images → Instagram Stories (1080×1920,
H.265) or YouTube (native resolution preserved, H.264, with audio).
Still images (JPEG/PNG/HEIC/TIFF/WebP) are auto-detected and routed through a
crop + resize-only pipeline; everything else (incl. ProRes .MOV) is video.

Pipeline (all vectorized, GPU-ready):
  1. Apple Log → Rec.709  (fast 256-entry LUT via cv2.LUT)
  2. Adaptive denoising   (bilateral filter, edge-preserving)
  3. Detail sharpening    (Unsharp Mask + fine detail boost)
  4. Local contrast       (CLAHE on LAB luminance)
  5. Chroma enhancement   (vibrance + saturation in HSV)
  6. Temporal coherence   (motion-aware inter-frame blending)
  7. H.265/HEVC output    (Apple HVC1, IG-optimized bitrate)

Performance (CPU-only):
  - 4K  @ ~2-3 fps  (denoise ON)
  - 4K  @ ~5-6 fps  (denoise OFF)
  - 1080p @ ~15-20 fps

Usage:
  python3 media_processor.py input.mp4 output.mp4
  python3 media_processor.py input.mp4 output.mp4 --preset instagram_stories_log
  python3 media_processor.py input.mp4 output.mp4 --log --sharpen 0.8
  python3 media_processor.py --list-presets
"""

import cv2
import numpy as np
import subprocess
import os
import sys
import argparse
import time
import json
import shutil
import tempfile
import platform
from pathlib import Path

# ── Windows compatibility ────────────────────────────────────────
IS_WINDOWS = platform.system() == 'Windows'

# On Windows, subprocess pipes need specific flags to avoid hangs
_POPEN_KWARGS = {}
if IS_WINDOWS:
    import subprocess as _sp
    _POPEN_KWARGS = {'creationflags': _sp.CREATE_NO_WINDOW}

# Still-image input extensions — these route to process_image() instead of
# process_video() (auto-detected by extension in main()).
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff', '.webp'}


def find_tool(name: str) -> str:
    """
    Locate ffmpeg/ffprobe on PATH, including common Windows install locations.
    Returns the full path or just the name if found on PATH.
    """
    # Check PATH first
    found = shutil.which(name)
    if found:
        return found

    # Common Windows install locations
    win_paths = [
        rf"C:\ffmpeg\bin\{name}.exe",
        rf"C:\Program Files\ffmpeg\bin\{name}.exe",
        rf"C:\Program Files (x86)\ffmpeg\bin\{name}.exe",
        os.path.expanduser(rf"~\ffmpeg\bin\{name}.exe"),
        os.path.expanduser(rf"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg*\bin\{name}.exe"),
    ]
    for p in win_paths:
        # Support glob-style paths
        import glob
        matches = glob.glob(p)
        if matches:
            return matches[0]
        if os.path.exists(p):
            return p

    return name  # fall back — will produce a clear error if missing




# ══════════════════════════════════════════════════════════════════
#  CROP MATH  —  Landscape → 9:16 Vertical
# ══════════════════════════════════════════════════════════════════

def resolve_x(position, max_x: int) -> int:
    """
    Resolve a position value (string or int) to a pixel offset in [0, max_x].
    Horizontal aliases: 'left'→0, 'right'→max. Vertical aliases (used when the
    crop window spans the full width, e.g. tall/portrait images): 'top'→0,
    'bottom'→max. 'center' (or anything else) → midpoint.
    """
    if isinstance(position, int):
        return max(0, min(position, max_x))
    p = str(position).lower()
    if p in ('left', 'top'):
        return 0
    elif p in ('right', 'bottom'):
        return max_x
    else:  # 'center'
        return max_x // 2


def compute_crop(src_w: int, src_h: int,
                 target_w: int = 1080, target_h: int = 1920,
                 position='center',
                 position_end=None,
                 total_frames: int = 1,
                 easing: str = 'linear') -> dict:
    """
    Compute crop parameters for extracting a 9:16 vertical window.

    Supports animated crop (pan) when position_end is provided:
      - position      : start x  ('center'|'left'|'right'|int)
      - position_end  : end x    (same types) — if None, static crop
      - total_frames  : total frame count (needed for interpolation)
      - easing        : 'linear' | 'ease_in_out' (smooth start+end)

    Returns a dict including get_crop_x(frame_n) — a callable that
    returns the crop_x for any given frame number.
    """
    target_ratio = target_h / target_w  # 1920/1080 = 1.777...

    crop_h = src_h
    crop_w = int(round(crop_h / target_ratio))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w * target_ratio))

    max_x  = src_w - crop_w
    max_y  = src_h - crop_h

    # Orientation of the crop window:
    #  - Landscape-ish source (max_x > 0): the window is narrower than the source,
    #    so `position` pans horizontally and the vertical crop stays centred.
    #    (This is the only case for drone/landscape video — behaviour unchanged.)
    #  - Tall/portrait source (full width used, max_y > 0): the window is the full
    #    width but shorter than the source, so `position` selects the *vertical*
    #    window instead ('top'/'bottom'/'center'/int), and crop_x is pinned to 0.
    vertical_crop = (max_x <= 0 and max_y > 0)

    if vertical_crop:
        start_x = 0
        crop_y  = resolve_x(position, max_y)
    else:
        start_x = resolve_x(position, max_x)
        crop_y  = (src_h - crop_h) // 2

    # Static crop (original behaviour)
    if position_end is None:
        end_x = start_x
        animated = False
    else:
        end_x    = resolve_x(position_end, max_x)
        animated = (end_x != start_x)

    def get_crop_x(frame_n: int) -> int:
        """Return interpolated crop_x for frame_n."""
        if not animated or total_frames <= 1:
            return start_x
        t = frame_n / (total_frames - 1)          # 0.0 → 1.0
        if easing == 'ease_in_out':
            # Smooth cubic ease — slow start, fast middle, slow end
            t = t * t * (3 - 2 * t)
        return int(round(start_x + (end_x - start_x) * t))

    scale_x = target_w / crop_w
    scale_y = target_h / crop_h
    scale   = max(scale_x, scale_y)

    # Static FFmpeg filter (used only when crop is not animated)
    static_x     = get_crop_x(0)
    ff_filter    = f"crop={crop_w}:{crop_h}:{static_x}:{crop_y},scale={target_w}:{target_h}:flags=lanczos"

    pan_distance = abs(end_x - start_x)
    if animated:
        pan_label = f"pan {pan_distance}px  ({start_x} → {end_x}  {easing})"
    elif vertical_crop:
        pan_label = f"static  y={crop_y} (vertical)"
    else:
        pan_label = f"static  x={start_x}"

    return {
        'crop_x':       start_x,
        'crop_y':       crop_y,
        'crop_w':       crop_w,
        'crop_h':       crop_h,
        'crop_x_end':   end_x,
        'scale':        scale,
        'out_w':        target_w,
        'out_h':        target_h,
        'animated':     animated,
        'vertical_crop': vertical_crop,   # True → `position` selected the vertical window
        'get_crop_x':   get_crop_x,
        'ff_filter':    ff_filter,   # valid only for static crops
        'pan_label':    pan_label,
    }

# ══════════════════════════════════════════════════════════════════
#  COLOR SCIENCE  —  Apple Log → Rec.709
# ══════════════════════════════════════════════════════════════════

# Apple Log is Apple's log transfer curve on **Rec.2020 primaries** (D65).
# A correct conversion to Rec.709 is a colour-space transform, per Apple's spec:
#   1. decode Apple Log signal → linear scene light (Rec.2020)
#   2. gamut: linear Rec.2020 → linear Rec.709  (3×3 matrix, mixes channels)
#   3. soft highlight roll-off (log scene-linear reaches ~12× diffuse white)
#   4. Rec.709 OETF (gamma)
# The old approach was a 1-D per-channel LUT, which CANNOT do step 2 — that is
# why Apple Log footage came out pale/desaturated with lifted blacks.

# Official Apple Log decode constants (signal V∈[0,1] → linear L; L=1.0 ≈ white)
_AL_R0, _AL_RT = -0.05641088, 0.01
_AL_C, _AL_BETA, _AL_GAMMA, _AL_DELTA = 47.28711236, 0.00964052, 0.08550479, 0.69336945
_AL_PT = _AL_C * (_AL_RT - _AL_R0) ** 2


def _apple_log_decode(v: np.ndarray) -> np.ndarray:
    """Apple Log signal (0–1) → linear scene light in Rec.2020 (per Apple's spec)."""
    return np.where(
        v < _AL_PT,
        np.sqrt(np.maximum(v, 0.0) / _AL_C) + _AL_R0,
        np.power(2.0, (v - _AL_DELTA) / _AL_GAMMA) - _AL_BETA,
    )


# 8-bit decode table (frames arrive as 8-bit BGR): code value → linear Rec.2020
_APPLE_LOG_DECODE = _apple_log_decode(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)

# Linear Rec.2020 → linear Rec.709 gamut matrix, written for BGR channel order
# (the standard RGB matrix with rows & columns reversed) so it applies directly
# to our BGR frames via cv2.transform.
_REC2020_TO_709_BGR = np.array([
    [ 1.118730, -0.100579, -0.018151],   # → B
    [-0.008349,  1.132900, -0.124550],   # → G
    [-0.072850, -0.587641,  1.660491],   # → R
], dtype=np.float32)

_HL_KNEE = 0.80   # linear highlights above this are softly rolled off toward 1.0

# Fast Rec.709 OETF lookup (avoids a per-pixel pow() at 4K)
_OETF_N = 4096
_oetf_x = np.linspace(0.0, 1.0, _OETF_N)
_OETF_LUT = np.where(_oetf_x < 0.018, _oetf_x * 4.5,
                     1.099 * np.power(_oetf_x, 0.45) - 0.099)
_OETF_LUT = np.clip(_OETF_LUT * 255.0, 0, 255).astype(np.uint8)


def _apply_log_lut_cpu(frame_bgr: np.ndarray) -> np.ndarray:
    """CPU implementation of the Apple Log → Rec.709 transform (NumPy)."""
    lin = _APPLE_LOG_DECODE[frame_bgr]                 # (H,W,3) float32 linear, Rec.2020
    lin = cv2.transform(lin, _REC2020_TO_709_BGR)      # → linear Rec.709 (gamut)
    np.clip(lin, 0.0, None, out=lin)

    # Soft-knee highlight roll-off: identity below the knee, compress toward 1.0 above
    k = _HL_KNEE
    over = lin > k
    if over.any():
        lin[over] = k + (1.0 - k) * np.tanh((lin[over] - k) / (1.0 - k))

    # Rec.709 OETF via fast table
    idx = np.clip(lin, 0.0, 1.0) * (_OETF_N - 1)
    return _OETF_LUT[idx.astype(np.int32)]


# ── Optional GPU backend (CuPy) for the Apple Log transform ──────────
# Identical math to the CPU path, run on the GPU. The per-frame NumPy version is
# the bottleneck on long 4K Log clips; CuPy offloads it to the (NVIDIA) GPU.
_LOG_GPU = None            # None = untried; False = unavailable; else dict of GPU arrays
_FORCE_CPU_LOG = False     # set by --cpu-log


def set_force_cpu_log(v: bool = True):
    """Force the CPU Log path (disables the CuPy GPU backend)."""
    global _FORCE_CPU_LOG, _LOG_GPU
    _FORCE_CPU_LOG = v
    _LOG_GPU = None


def _log_gpu():
    """Lazily set up + cache the CuPy GPU backend. Returns a dict, or None."""
    global _LOG_GPU
    if _LOG_GPU is not None:
        return _LOG_GPU or None
    if _FORCE_CPU_LOG:
        _LOG_GPU = False
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            import cupy as cp
        g = {
            'cp':     cp,
            'decode': cp.asarray(_APPLE_LOG_DECODE),         # (256,) f32  code→linear
            'M':      cp.asarray(_REC2020_TO_709_BGR.T),     # (3,3) transposed for (H,W,3)@M
            'oetf':   cp.asarray(_OETF_LUT),                 # (N,) u8  linear→Rec.709
        }
        # Warm up / verify kernels actually compile and run on this machine
        w = cp.zeros((2, 2, 3), cp.uint8)
        _ = (g['decode'][w] @ g['M'])
        cp.cuda.runtime.deviceSynchronize()
        _LOG_GPU = g
    except Exception:
        _LOG_GPU = False
    return _LOG_GPU or None


def _apply_log_lut_gpu(frame_bgr: np.ndarray, g: dict) -> np.ndarray:
    cp = g['cp']
    x = cp.asarray(frame_bgr)                  # upload u8 (H,W,3)
    lin = g['decode'][x]                       # → f32 linear Rec.2020
    lin = lin @ g['M']                         # gamut → linear Rec.709
    cp.clip(lin, 0.0, None, out=lin)
    k = _HL_KNEE
    lin = cp.where(lin > k, k + (1.0 - k) * cp.tanh((lin - k) / (1.0 - k)), lin)
    idx = (cp.clip(lin, 0.0, 1.0) * (_OETF_N - 1)).astype(cp.int32)
    return cp.asnumpy(g['oetf'][idx])          # download u8


def apply_log_lut(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apple Log → Rec.709, colour-managed: decode → Rec.2020→709 gamut matrix →
    soft highlight roll-off → Rec.709 OETF. Runs on the GPU via CuPy when
    available (identical math), else on the CPU. Input/output: BGR uint8.
    """
    g = _log_gpu()
    if g is not None:
        try:
            return _apply_log_lut_gpu(frame_bgr, g)
        except Exception:
            pass    # any GPU hiccup → fall back to CPU for this frame
    return _apply_log_lut_cpu(frame_bgr)


def apply_chroma(frame: np.ndarray, sat: float, vib: float) -> np.ndarray:
    """
    Vibrance + saturation in HSV. Shared by the video Enhancer and the image
    pipeline so both use identical colour math.
      - sat : saturation multiplier (1.0 = unchanged)
      - vib : vibrance 0→1 — adaptive boost that affects dull colours more and
              protects already-saturated areas (e.g. skin tones).
    """
    if abs(sat - 1.0) < 0.01 and vib < 0.01:
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Vibrance boost: more effect on low-saturation pixels
    vib_boost = 1.0 + vib * (1.0 - s / 255.0)
    s = np.clip(s * sat * vib_boost, 0, 255)

    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_temperature(frame: np.ndarray, temp: float) -> np.ndarray:
    """
    White-balance temperature shift — the right tool for an overall colour cast
    (unlike saturation, which only changes how colourful things are).
      temp < 0 → cooler  (less red, more blue) — neutralises a warm/yellow cast
      temp > 0 → warmer  (more red, less blue)
    temp is roughly [-1, 1]; ±1 ≈ ±35% gain on the red/blue channels. Green is
    left as the luma anchor so brightness stays about the same.
    """
    if abs(temp) < 0.01:
        return frame
    t = max(-1.0, min(float(temp), 1.0))
    r_gain = 1.0 + 0.35 * t
    b_gain = 1.0 - 0.35 * t
    f = frame.astype(np.float32)
    f[..., 2] *= r_gain   # R  (OpenCV channel order is BGR)
    f[..., 0] *= b_gain   # B
    return np.clip(f, 0, 255).astype(np.uint8)


def fit_cover(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Scale `frame` to completely cover out_w×out_h while preserving aspect ratio,
    then centre-crop the overflow (the 'cover' fit — fills the frame, no bars).
    Used to honour force_resolution when the source isn't already that size and
    no explicit crop step produced it (e.g. a 16:9 clip → 9:16 IG Stories).
    """
    h, w = frame.shape[:2]
    if (w, h) == (out_w, out_h):
        return frame
    scale = max(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    x = (nw - out_w) // 2
    y = (nh - out_h) // 2
    return resized[y:y + out_h, x:x + out_w]


def fit_focus(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Fit the WHOLE frame inside out_w×out_h (contain — never cropped), so a
    landscape ends up width-aligned with letterbox space, a portrait height-
    aligned. The leftover space is filled with a soft, darkened, blurred 'cover'
    of the same image (the Instagram/YouTube look) rather than dead bars. When
    the frame already matches the target aspect this is just a plain fill.
    """
    h, w = frame.shape[:2]
    fasp = out_w / out_h
    asp = w / h
    # Sharp contain size
    if asp > fasp:                       # wider than frame → fit width
        cw, ch = out_w, max(1, int(round(out_w / asp)))
    else:                                # taller/equal → fit height
        cw, ch = max(1, int(round(out_h * asp))), out_h
    interp = cv2.INTER_AREA if cw < w else cv2.INTER_LANCZOS4
    sharp = cv2.resize(frame, (cw, ch), interpolation=interp)
    if cw >= out_w and ch >= out_h:
        return sharp                     # exact fill, no bars needed

    # Blurred cover fill (cheap: downscale → upscale → light blur), darkened
    small = cv2.resize(frame, (max(1, out_w // 12), max(1, out_h // 12)),
                       interpolation=cv2.INTER_AREA)
    bg = fit_cover(small, out_w, out_h)
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    out = (bg.astype(np.float32) * 0.55).astype(np.uint8)
    x = (out_w - cw) // 2
    y = (out_h - ch) // 2
    out[y:y + ch, x:x + cw] = sharp
    return out


def contain_rect(asp: float, W: int, H: int):
    """Centred rect (x, y, w, h floats) that fits aspect `asp` inside W×H (contain)."""
    if asp > (W / H):           # wider than frame → fit width
        cw, ch = float(W), W / asp
    else:                       # taller/equal → fit height
        cw, ch = H * asp, float(H)
    return ((W - cw) / 2.0, (H - ch) / 2.0, cw, ch)


def _render_dive_frame(canvas, tile_rect, src_native, c_rect, e, W, H):
    """
    One dive frame (smooth, monotonic camera push). The collage `canvas` is
    camera-zoomed toward the tile and darkened so the subject pops; the focused
    asset — re-rendered SHARP from its full-res source at its NATIVE aspect —
    grows from its tile rect (e=0) to its centred contain rect (e=1). Because
    both rects share the asset's aspect, it scales without distortion or cropping.
    """
    cx, cy, cw, ch = tile_rect
    ax, ay = e * cx, e * cy
    aw, ah = W + e * (cw - W), H + e * (ch - H)
    Mcam = np.array([[W / aw, 0, -ax * W / aw],
                     [0, H / ah, -ay * H / ah]], dtype=np.float32)
    out = cv2.warpAffine(canvas, Mcam, (W, H), flags=cv2.INTER_LINEAR)
    if e > 0.001:                                   # darken background as we close in
        out = cv2.convertScaleAbs(out, alpha=(1.0 - 0.45 * e))   # C-optimized (fast)

    tx, ty, tw, th = c_rect
    fx, fy = cx + e * (tx - cx), cy + e * (ty - cy)
    fw, fh = cw + e * (tw - cw), ch + e * (th - ch)
    iw, ih = max(1, int(round(fw))), max(1, int(round(fh)))
    interp = cv2.INTER_AREA if iw < src_native.shape[1] else cv2.INTER_LINEAR
    sharp = cv2.resize(src_native, (iw, ih), interpolation=interp)

    X, Y = int(round(fx)), int(round(fy))
    x0, y0 = max(0, X), max(0, Y)
    x1, y1 = min(W, X + iw), min(H, Y + ih)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = sharp[y0 - Y:y1 - Y, x0 - X:x1 - X]
    return out


# Hue-band centres in degrees (0–360), matching the Lightroom-style HSL model.
HUE_BANDS = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']
HUE_BAND_CENTERS = {
    'red': 0,   'orange': 30,  'yellow': 60,  'green': 120,
    'aqua': 180, 'blue': 240,  'purple': 270, 'magenta': 300,
}


def apply_selective_saturation(frame: np.ndarray, bands: dict, sigma: float = 20.0) -> np.ndarray:
    """
    Per-colour (hue-selective) saturation — like Lightroom's HSL Saturation.
    `bands` maps a colour name (see HUE_BANDS) to a multiplier (1.0 = unchanged).

    All eight hue bands take part in the blend; any band the caller doesn't set
    defaults to 1.0. Each band contributes a Gaussian weight around its hue-wheel
    centre and the per-pixel multiplier is the weighted average of the band
    multipliers. Because every hue is anchored to its own band (multiplier 1.0
    when unset), reducing e.g. red/orange/yellow leaves greens/blues unchanged
    instead of spilling onto them. An empty / all-1.0 `bands` is a true no-op.

    Typical use: tame warm tones after a global saturation boost, e.g.
    bands={'yellow':0.5,'orange':0.7,'red':0.8}.
    """
    if not bands or all(abs(m - 1.0) < 0.01 for m in bands.values()):
        return frame

    # Full 8-band model — unset bands = 1.0 so distant hues stay unchanged.
    centers = np.array([HUE_BAND_CENTERS[c] for c in HUE_BANDS], dtype=np.float32)
    mults   = np.array([bands.get(c, 1.0) for c in HUE_BANDS], dtype=np.float32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h_deg = h * 2.0  # OpenCV hue is 0–179 → 0–358

    diff = np.abs(h_deg[..., None] - centers[None, None, :])
    d = np.minimum(diff, 360.0 - diff)              # circular hue distance
    w = np.exp(-(d * d) / (2.0 * sigma * sigma))    # (H, W, 8)

    m = np.sum(w * mults, axis=-1) / np.sum(w, axis=-1)   # weighted avg (denom never 0)

    s = np.clip(s * m, 0, 255)
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


# ══════════════════════════════════════════════════════════════════
#  ENHANCEMENT ENGINE
# ══════════════════════════════════════════════════════════════════

class Enhancer:
    """
    Topaz Video AI-style frame enhancement pipeline.
    All operations vectorized — no Python loops over pixels.
    Input/output: uint8 BGR frames (avoids float32 conversion overhead in hot path).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg['clahe_clip'],
            tileGridSize=(8, 8)
        )

    # ── Step 1: Denoise ─────────────────────────────────────────

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Adaptive denoising.
        Light strengths use a fast edge-preserving bilateral filter;
        stronger strengths switch to Non-Local Means (Topaz-level quality,
        slower but far better noise reduction).
        """
        s = self.cfg['denoise_strength']
        if s < 0.05:
            return frame
        h = max(1, min(int(s * 10), 10))   # NLM filter strength 1–10
        if s < 0.4:
            # Fast bilateral for light denoising
            sigma = int(s * 80)
            return cv2.bilateralFilter(frame, 5, sigma, sigma)
        else:
            # NLM for stronger denoising — slower but much better quality
            return cv2.fastNlMeansDenoisingColored(frame, None,
                       h=h, hColor=h, templateWindowSize=7, searchWindowSize=21)

    # ── Step 2: Sharpen + Detail Recovery ───────────────────────

    def sharpen(self, frame: np.ndarray) -> np.ndarray:
        """
        Two-layer sharpening:
          a) Unsharp Mask  — broadband frequency detail (Topaz 'Sharpen')
          b) Fine detail   — sub-pixel texture injection (Topaz 'Recover Detail')
        """
        amount = self.cfg['sharpen_amount']
        detail = self.cfg['detail_amount']
        if amount < 0.01 and detail < 0.01:
            return frame

        f32 = frame.astype(np.float32)
        r = self.cfg['sharpen_radius']
        ksize = max(3, int(r * 6) | 1)  # ensure odd kernel

        # Unsharp Mask
        blurred = cv2.GaussianBlur(f32, (ksize, ksize), r)
        result = f32 + amount * (f32 - blurred)

        # Fine detail injection (sub-pixel high-freq)
        if detail > 0.01:
            fine_blur = cv2.GaussianBlur(f32, (3, 3), 0.5)
            result += detail * (f32 - fine_blur)

        return np.clip(result, 0, 255).astype(np.uint8)

    # ── Step 3: Local Contrast ───────────────────────────────────

    def local_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        CLAHE on LAB luminance channel only.
        Enhances micro-contrast (Topaz 'Micro Contrast' / tone mapping).
        Operating in LAB avoids color shifts — only luminance is affected.
        """
        if not self.cfg['local_contrast']:
            return frame
        blend = self.cfg['clahe_blend']

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l_enh = self._clahe.apply(l)
        l_out = cv2.addWeighted(l, 1.0 - blend, l_enh, blend, 0)
        return cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_Lab2BGR)

    # ── Step 4: Chroma Enhancement ───────────────────────────────

    def chroma(self, frame: np.ndarray) -> np.ndarray:
        """
        Vibrance + saturation in HSV.
        Vibrance = adaptive saturation: boosts dull colors more, protects
        already-saturated areas (like skin tones) from over-saturation.
        """
        return apply_chroma(frame, self.cfg['saturation'], self.cfg['vibrance'])

    # ── Step 5: Super Resolution / Upscale ──────────────────────

    def upscale_frame(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """
        Multi-pass upscaling with detail injection.
        Lanczos4 baseline → inject native-res detail map → result is sharper
        than pure interpolation without ringing artifacts.
        """
        if abs(scale - 1.0) < 0.01:
            return frame

        h, w = frame.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        # Primary: Lanczos4 (best quality standard interpolation)
        up = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Detail injection
        inject = self.cfg.get('detail_inject', 1.0)
        if inject > 0.01:
            f32 = frame.astype(np.float32)
            blurred = cv2.GaussianBlur(f32, (3, 3), 1.2)
            detail_map = f32 - blurred
            detail_up = cv2.resize(detail_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            up = np.clip(up.astype(np.float32) + inject * detail_up, 0, 255).astype(np.uint8)

        return up

    # ── Full pipeline ─────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply complete enhancement pipeline to one BGR uint8 frame.
        Order: color → denoise at native res → upscale → sharpen → contrast → chroma
        (denoising before upscale is faster and avoids amplifying noise).
        """
        if self.cfg['is_log']:
            frame = apply_log_lut(frame)

        frame = self.denoise(frame)

        scale = self.cfg.get('upscale', 1.0)
        if scale != 1.0:
            frame = self.upscale_frame(frame, scale)

        frame = self.sharpen(frame)
        frame = self.local_contrast(frame)
        frame = self.chroma(frame)

        return frame



# ══════════════════════════════════════════════════════════════════
#  GPU DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_gpu() -> dict:
    """
    Check CUDA availability via OpenCV and nvidia-smi.
    Returns capability dict used throughout the pipeline.
    """
    info = {
        'cuda_cv2':   False,   # OpenCV CUDA ops available
        'nvenc':      False,   # FFmpeg NVENC encoder available
        'nvdec':      False,   # FFmpeg NVDEC decoder available
        'device_name': None,
        'cuda_count':  0,
    }

    # ── OpenCV CUDA ──
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            info['cuda_cv2']  = True
            info['cuda_count'] = count
            cv2.cuda.setDevice(0)
            dev = cv2.cuda.DeviceInfo(0)
            info['device_name'] = dev.name()
    except Exception:
        pass

    # ── FFmpeg NVENC / NVDEC ──
    ffmpeg = find_tool('ffmpeg')
    try:
        r = subprocess.run(
            [ffmpeg, '-hide_banner', '-encoders'],
            capture_output=True, text=True, **_POPEN_KWARGS
        )
        if 'hevc_nvenc' in r.stdout:
            info['nvenc'] = True
        r2 = subprocess.run(
            [ffmpeg, '-hide_banner', '-decoders'],
            capture_output=True, text=True, **_POPEN_KWARGS
        )
        if 'h264_cuvid' in r2.stdout or 'hevc_cuvid' in r2.stdout:
            info['nvdec'] = True
    except Exception:
        pass

    return info


GPU = detect_gpu()   # detected once at import time


def print_gpu_status():
    if GPU['cuda_cv2']:
        print(f"  🟢  GPU   {GPU['device_name']}  (OpenCV CUDA ✅  NVENC {'✅' if GPU['nvenc'] else '❌'}  NVDEC {'✅' if GPU['nvdec'] else '❌'})")
    else:
        print("  🔴  GPU   CUDA not available — running on CPU")
        print("           (Install opencv-contrib-python with CUDA support for GPU acceleration)")


# ══════════════════════════════════════════════════════════════════
#  GPU ENHANCEMENT ENGINE
# ══════════════════════════════════════════════════════════════════

class GpuEnhancer:
    """
    GPU-accelerated enhancement pipeline using cv2.cuda.
    Mirrors the Enhancer API exactly — drop-in replacement.

    Key design: frames are uploaded to GPU memory once per frame,
    all ops run on GPU, then downloaded once for FFmpeg write.
    This minimises PCIe transfers (the main GPU bottleneck for video).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Pre-build GPU-resident objects (avoid per-frame construction)
        self._clahe = cv2.cuda.createCLAHE(
            clipLimit=cfg['clahe_clip'],
            tileGridSize=(8, 8)
        )
        # Gaussian filter objects (fixed kernel — build once)
        r = cfg['sharpen_radius']
        ksize = max(3, int(r * 6) | 1)
        self._gauss_sharp = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, (ksize, ksize), r
        )
        self._gauss_fine = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 0.5
        )
        self._gauss_temporal = cv2.cuda.createGaussianFilter(
            cv2.CV_32FC1, cv2.CV_32FC1, (7, 7), 2.0
        )

    # ── Denoise ──

    def denoise(self, gpu_frame: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        s = self.cfg['denoise_strength']
        if s < 0.05:
            return gpu_frame
        h = max(1, min(int(s * 10), 10))   # NLM filter strength 1–10
        if s < 0.4:
            # Fast bilateral for light denoising
            sigma = int(s * 80)
            return cv2.cuda.bilateralFilter(gpu_frame, 5, sigma, sigma)
        else:
            # NLM for stronger denoising — GPU NLM is grayscale only, so we
            # convert to LAB, denoise the L (luminance) channel only, and
            # reconstruct, leaving chroma untouched.
            lab = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2Lab)
            channels = cv2.cuda.split(lab)
            l_den = cv2.cuda.fastNlMeansDenoising(channels[0], h,
                        search_window=21, block_size=7)
            cv2.cuda.merge([l_den, channels[1], channels[2]], lab)
            return cv2.cuda.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # ── Sharpen ──

    def sharpen(self, gpu_frame: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        amount = self.cfg['sharpen_amount']
        detail = self.cfg['detail_amount']
        if amount < 0.01 and detail < 0.01:
            return gpu_frame

        # Convert to float for arithmetic
        gpu_f = gpu_frame.convertTo(cv2.CV_32FC3, alpha=1.0)

        # Unsharp mask
        blurred = self._gauss_sharp.apply(gpu_frame).convertTo(cv2.CV_32FC3)
        # USM: frame + amount*(frame - blurred)
        diff = cv2.cuda.subtract(gpu_f, blurred)
        sharpened = cv2.cuda.addWeighted(gpu_f, 1.0, diff, amount, 0)

        # Fine detail
        if detail > 0.01:
            fine_blur = self._gauss_fine.apply(gpu_frame).convertTo(cv2.CV_32FC3)
            fine = cv2.cuda.subtract(gpu_f, fine_blur)
            sharpened = cv2.cuda.addWeighted(sharpened, 1.0, fine, detail, 0)

        result = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)
        sharpened.convertTo(cv2.CV_8UC3, dst=result)
        return result

    # ── Local contrast (CLAHE) ──

    def local_contrast(self, gpu_frame: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        if not self.cfg['local_contrast']:
            return gpu_frame
        blend = self.cfg['clahe_blend']

        # Convert BGR → Lab on GPU
        lab = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2Lab)
        # Split channels
        channels = cv2.cuda.split(lab)
        l_ch = channels[0]

        # Apply CLAHE to L channel
        l_enh = self._clahe.apply(l_ch, cv2.cuda_Stream.Null())

        # Blend original L with enhanced L
        l_out = cv2.cuda.addWeighted(l_ch, 1.0 - blend, l_enh, blend, 0)

        # Merge and convert back
        cv2.cuda.merge([l_out, channels[1], channels[2]], lab)
        return cv2.cuda.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # ── Chroma ──

    def chroma(self, gpu_frame: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        sat = self.cfg['saturation']
        vib = self.cfg['vibrance']
        if abs(sat - 1.0) < 0.01 and vib < 0.01:
            return gpu_frame

        # BGR → HSV on GPU
        hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
        channels = cv2.cuda.split(hsv)
        h_ch, s_ch, v_ch = channels[0], channels[1], channels[2]

        # Download S channel for vibrance math (numpy), upload result
        # (vibrance requires per-pixel adaptive boost — easier in numpy,
        #  but the bulk cvtColor ops stay on GPU)
        s_cpu = s_ch.download().astype(np.float32)
        vib_boost = 1.0 + vib * (1.0 - s_cpu / 255.0)
        s_out = np.clip(s_cpu * sat * vib_boost, 0, 255).astype(np.uint8)
        s_ch_new = cv2.cuda_GpuMat()
        s_ch_new.upload(s_out)

        # Merge and convert back on GPU
        cv2.cuda.merge([h_ch, s_ch_new, v_ch], hsv)
        return cv2.cuda.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ── Upscale ──

    def upscale_frame(self, gpu_frame: cv2.cuda_GpuMat, scale: float) -> cv2.cuda_GpuMat:
        if abs(scale - 1.0) < 0.01:
            return gpu_frame
        h, w = gpu_frame.size()[1], gpu_frame.size()[0]
        new_w, new_h = int(w * scale), int(h * scale)

        up = cv2.cuda.resize(gpu_frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        inject = self.cfg.get('detail_inject', 1.0)
        if inject > 0.01:
            # Detail injection: compute on CPU (small), upload
            cpu_frame = gpu_frame.download()
            f32 = cpu_frame.astype(np.float32)
            blurred = cv2.GaussianBlur(f32, (3, 3), 1.2)
            detail_map = cv2.resize(f32 - blurred, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            detail_gpu = cv2.cuda_GpuMat()
            detail_gpu.upload(detail_map.astype(np.float32))
            up_f = up.convertTo(cv2.CV_32FC3)
            up_f = cv2.cuda.addWeighted(up_f, 1.0, detail_gpu, inject, 0)
            result = cv2.cuda_GpuMat()
            up_f.convertTo(cv2.CV_8UC3, dst=result)
            return result

        return up

    # ── Full pipeline ──

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Upload → GPU pipeline → download. One PCIe round-trip per frame."""
        # Apple Log → Rec.709 is a colour-managed transform (gamut matrix +
        # tone roll-off) done on the CPU before upload; a per-channel GPU LUT
        # can't do the Rec.2020→709 gamut step correctly.
        if self.cfg['is_log']:
            frame = apply_log_lut(frame)

        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)

        gpu = self.denoise(gpu)

        scale = self.cfg.get('upscale', 1.0)
        if scale != 1.0:
            gpu = self.upscale_frame(gpu, scale)

        gpu = self.sharpen(gpu)
        gpu = self.local_contrast(gpu)
        gpu = self.chroma(gpu)

        return gpu.download()

# ══════════════════════════════════════════════════════════════════
#  TEMPORAL COHERENCE
# ══════════════════════════════════════════════════════════════════

class TemporalBuffer:
    """
    Motion-aware inter-frame blending to reduce flicker/noise.
    Stationary regions are blended with previous frame(s).
    Moving regions use current frame directly (no ghosting on motion).
    """

    def __init__(self, blend: float = 0.12, size: int = 3):
        self.blend   = blend
        self.size    = size
        self.history = []

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Temporal blending disabled → pass through WITHOUT buffering. (Buffering
        # here would append a frame copy every iteration and never free it, so on
        # long/4K clips the history grows unbounded and exhausts memory.)
        if self.blend < 0.01:
            return frame
        if not self.history:
            self.history.append(frame.copy())
            return frame

        prev = self.history[-1]

        # Motion detection: pixels changed >12 counts → don't blend
        diff = cv2.absdiff(frame, prev)
        motion = (diff.mean(axis=2) > 12).astype(np.float32)
        motion = cv2.GaussianBlur(motion, (7, 7), 2.0)[..., np.newaxis]

        blended = cv2.addWeighted(prev, self.blend, frame, 1.0 - self.blend, 0)
        result = np.clip(
            frame.astype(np.float32) * motion +
            blended.astype(np.float32) * (1.0 - motion),
            0, 255
        ).astype(np.uint8)

        self.history.append(frame.copy())
        if len(self.history) > self.size:
            self.history.pop(0)

        return result


# ══════════════════════════════════════════════════════════════════
#  VIDEO I/O
# ══════════════════════════════════════════════════════════════════

def probe_video(path: str) -> dict:
    ffprobe = find_tool('ffprobe')
    try:
        r = subprocess.run(
            [ffprobe, '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', path],
            capture_output=True, text=True, check=True,
            **_POPEN_KWARGS
        )
    except FileNotFoundError:
        print("\n\u274c  ffprobe not found. Please install FFmpeg and add it to PATH.")
        print("    Download: https://www.gyan.dev/ffmpeg/builds/")
        print("    Extract to C:\\ffmpeg, add C:\\ffmpeg\\bin to system PATH, restart terminal.\n")
        sys.exit(1)

    info = json.loads(r.stdout)
    vs   = next(s for s in info['streams'] if s['codec_type'] == 'video')
    num, den = vs['r_frame_rate'].split('/')
    fps  = int(num) / int(den)
    nb   = int(vs.get('nb_frames', 0)) or int(float(info['format'].get('duration', 0)) * fps)

    # Rotation: ffmpeg auto-rotates on decode, so rawvideo frames come out at the
    # DISPLAY size. Report that (swap W/H for ±90°/270°) — otherwise frames get
    # reshaped with the wrong dimensions and look scrambled/sheared.
    rot = 0
    for sd in vs.get('side_data_list', []):
        if sd.get('rotation') is not None:
            try:
                rot = int(round(float(sd['rotation'])))
            except (TypeError, ValueError):
                pass
    if rot == 0:
        try:
            rot = int(round(float(vs.get('tags', {}).get('rotate', 0))))
        except (TypeError, ValueError):
            rot = 0
    w, h = int(vs['width']), int(vs['height'])
    if rot % 180 != 0:
        w, h = h, w

    return {
        'width':     w,
        'height':    h,
        'fps':       fps,
        'frames':    nb,
        'codec':     vs['codec_name'],
        'duration':  float(info['format'].get('duration', 0)),
        'has_audio': any(s['codec_type'] == 'audio' for s in info['streams']),
    }


def make_reader(path: str, vf: str = None,
                start_time: float = None, end_time: float = None,
                out_fps: float = None) -> subprocess.Popen:
    """
    Open an FFmpeg pipe reader with optional trim and crop filter.

    out_fps: if set, FFmpeg resamples (dup/drop) the decoded frames to this
    constant rate before piping. Used by the montage builder so every clip is
    delivered at one common fps — the resulting temp clips then concat losslessly.

    GPU note: We use -hwaccel cuda (assists decode) but NOT
    -hwaccel_output_format cuda. Keeping frames in CPU memory avoids
    three bugs: (1) hwdownload filter conflicts with animated crop which
    needs no vf at all, (2) -to trim interacts badly with cuda output
    format producing 0 frames, (3) the filter graph fails to initialise
    when no vf is present in cuda output mode.
    NVENC (encode) still runs fully on GPU — that's the bigger win anyway.

    start_time / end_time: seconds (float). Use None to mean start/end of file.
    """
    ffmpeg = find_tool('ffmpeg')
    cmd = [ffmpeg, '-v', 'quiet']

    # Fast keyframe seek (before -i) — only for offsets > 5s to avoid overhead.
    # Seeks to 2s before start so the subsequent frame-accurate -ss has
    # a short distance to decode accurately.
    # Bug fix: use `is not None` so start_time=0.0 is handled correctly.
    if start_time is not None and start_time > 5.0:
        cmd += ['-ss', f'{start_time - 2.0:.3f}']

    # GPU-assisted decode (no output_format=cuda — frames stay CPU-side)
    if GPU['nvdec']:
        cmd += ['-hwaccel', 'cuda']

    cmd += ['-i', path]

    # Frame-accurate trim after -i.
    # Bug fix: use `is not None` so 0.0 start is never skipped.
    if start_time is not None:
        cmd += ['-ss', f'{start_time:.3f}']

    # Bug fix: use -t (duration) instead of -to (absolute timestamp).
    # -to after -i is relative to the *decoded* position when a prior -ss
    # is present, which makes the duration unpredictable. -t is always a
    # simple duration from the current decode position — no ambiguity.
    if start_time is not None and end_time is not None:
        cmd += ['-t', f'{end_time - start_time:.3f}']
    elif end_time is not None:
        cmd += ['-t', f'{end_time:.3f}']

    if vf:
        cmd += ['-vf', vf]

    # Normalise output frame rate (dup/drop) so all montage clips share one fps
    if out_fps is not None:
        cmd += ['-r', f'{out_fps:.6f}']

    cmd += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        **_POPEN_KWARGS
    )


def make_writer(path: str, w: int, h: int, fps: float,
                audio_src: str, crf: int, bitrate: str,
                silent: bool = False, vcodec: str = 'h265',
                audio_ss: float = None, audio_t: float = None) -> subprocess.Popen:
    ffmpeg = find_tool('ffmpeg')
    buf = f"{int(bitrate.rstrip('M')) * 2}M"
    return subprocess.Popen(
        _build_writer_cmd(ffmpeg, w, h, fps, audio_src, crf, bitrate, buf, path,
                          silent, vcodec, audio_ss, audio_t),
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        **_POPEN_KWARGS
    )


def _build_writer_cmd(ffmpeg, w, h, fps, audio_src, crf, bitrate, buf, path,
                      silent=False, vcodec='h265', audio_ss=None, audio_t=None):
    """
    Build FFmpeg writer command — NVENC if GPU available, CPU library otherwise.
    vcodec='h265' (IG Stories, Apple HVC1) or 'h264' (most compatible — YouTube).
    silent=True drops audio entirely (used for montage temp clips).
    audio_ss/audio_t trim the muxed audio to the processed video's time window
    (so a trimmed clip's audio matches its video instead of running full length).
    """
    base = [ffmpeg, '-v', 'quiet', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0']
    if silent:
        maps = ['-map', '0:v']
    else:
        # Trim the audio input to the same window as the video (before -i = fast/accurate)
        if audio_ss is not None:
            base += ['-ss', f'{audio_ss:.3f}']
        if audio_t is not None:
            base += ['-t', f'{audio_t:.3f}']
        base += ['-i', audio_src]
        maps = ['-map', '0:v', '-map', '1:a?']

    maxrate2 = f'{int(bitrate.rstrip("M")) * 2}M'
    if vcodec == 'h264':
        # H.264 / AVC — maximum compatibility (YouTube's recommended upload codec)
        if GPU['nvenc']:
            enc = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', str(crf),
                   '-b:v', bitrate, '-maxrate', maxrate2, '-bufsize', buf]
        else:
            enc = ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
                   '-b:v', bitrate, '-maxrate', bitrate, '-bufsize', buf]
        vtag = []  # default avc1 tag is correct for H.264 in mp4
    else:
        # H.265 / HEVC — efficient, tagged hvc1 for Apple/IG compatibility
        if GPU['nvenc']:
            enc = ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', str(crf),
                   '-b:v', bitrate, '-maxrate', maxrate2, '-bufsize', buf]
        else:
            enc = ['-c:v', 'libx265', '-preset', 'medium', '-crf', str(crf),
                   '-b:v', bitrate, '-maxrate', bitrate, '-bufsize', buf]
        vtag = ['-tag:v', 'hvc1']

    tail = ['-pix_fmt', 'yuv420p',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709']
    tail += vtag
    if not silent:
        tail += ['-c:a', 'aac', '-b:a', '192k']
    tail += ['-movflags', '+faststart', path]

    return base + maps + enc + tail


# ══════════════════════════════════════════════════════════════════
#  MAIN PROCESSOR
# ══════════════════════════════════════════════════════════════════

def parse_time(t) -> float:
    """
    Parse a time value into seconds (float).
    Accepts: float/int seconds, or 'MM:SS' / 'HH:MM:SS' strings.
    Examples: 10.5  |  '1:30'  |  '0:10'  |  '1:02:30'
    """
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    parts = str(t).strip().split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(parts[0])


def fmt_time(s: float) -> str:
    """Format seconds as M:SS.f"""
    m = int(s // 60)
    return f"{m}:{s % 60:05.2f}"


def process_video(input_path: str, output_path: str, cfg: dict,
                  silent: bool = False, force_fps: float = None):
    """
    Process one video → enhanced 1080×1920 H.265.
    silent=True drops audio; force_fps resamples to a constant rate. Both are
    used by the montage builder (process_sequence) so temp clips concat cleanly;
    defaults preserve the original single-video behaviour exactly.
    """
    info = probe_video(input_path)
    fps, w, h = info['fps'], info['width'], info['height']
    out_fps = force_fps if force_fps else fps   # encode/resample rate
    total_duration = info['duration']

    # ── Time window ───────────────────────────────────────────────
    # Resolve start/end from cfg (set by CLI --start-time / --end-time)
    start_t = parse_time(cfg.get('start_time', None))
    end_t   = parse_time(cfg.get('end_time',   None))

    # Clamp to valid range
    if start_t is not None:
        start_t = max(0.0, min(start_t, total_duration - 0.1))
    if end_t is not None:
        end_t = max(0.1, min(end_t, total_duration))
    if start_t and end_t and end_t <= start_t:
        print(f"❌  --end-time ({end_t}s) must be greater than --start-time ({start_t}s)")
        sys.exit(1)

    # Segment duration and frame count (for easing calculation)
    seg_start    = start_t or 0.0
    seg_end      = end_t   or total_duration
    seg_duration = seg_end - seg_start
    seg_frames   = int(round(seg_duration * out_fps))   # frames actually delivered

    scale = cfg.get('upscale', 1.0)
    out_w, out_h = int(w * scale), int(h * scale)
    # force_resolution overrides scale-computed output size (e.g. IG Stories = 1080x1920)
    if cfg.get('force_resolution'):
        out_w, out_h = cfg['force_resolution']

    # ── Drone crop mode ──────────────────────────────────────────
    # Crop a 9:16 window from landscape source, then super-resolve to 1080x1920
    # Supports animated pan: crop_x interpolates from crop_position → crop_position_end
    crop_info = None
    vf_filter  = None
    if cfg.get('drone_crop'):
        position     = cfg.get('crop_position', 'center')
        position_end = cfg.get('crop_position_end', None)
        easing       = cfg.get('crop_easing', 'ease_in_out')
        target_w     = cfg.get('crop_target_w', 1080)
        target_h     = cfg.get('crop_target_h', 1920)
        crop_info    = compute_crop(w, h, target_w, target_h,
                                    position, position_end,
                                    total_frames=seg_frames,
                                    easing=easing)
        out_w, out_h = crop_info['out_w'], crop_info['out_h']

        if crop_info['animated']:
            # Animated crop: must be done per-frame in Python
            # FFmpeg reader delivers full-resolution frames; crop happens in the loop
            vf_filter = None
        else:
            # Static crop: let FFmpeg handle it (faster, can use NVDEC)
            vf_filter = crop_info['ff_filter']

    # Auto-select GPU or CPU enhancer (must happen before header prints)
    if GPU['cuda_cv2']:
        enhancer = GpuEnhancer(cfg)
        accel = f"GPU ({GPU['device_name']})"
    else:
        enhancer = Enhancer(cfg)
        accel = "CPU (install CUDA-enabled OpenCV for GPU)"

    print(f"\n{'═'*62}")
    print(f"  🎬  Video Upscaler  —  Topaz-style Enhancement POC")
    print(f"{'═'*62}")
    # Handle force_resolution (e.g. IG Stories 1080x1920)
    force_res = cfg.get('force_resolution')
    if force_res:
        out_w, out_h = force_res
        if (w, h) != (out_w, out_h):
            print(f"  ⚙️   Resampling {w}×{h} → {out_w}×{out_h} for IG Stories compliance")

    # Build time window label
    if start_t or end_t:
        tw_label = f"  [{fmt_time(seg_start)} → {fmt_time(seg_end)}  {seg_duration:.1f}s  {seg_frames} frames]"
    else:
        tw_label = f"  ({info['duration']:.1f}s  {info['frames']} frames)"

    vcodec = cfg.get('output_codec', 'h265')
    codec_label = "H.264/AVC" if vcodec == 'h264' else "H.265/HEVC  (Apple HVC1)"
    print(f"  Input   {w}×{h}  {fps:.2f}fps  {info['codec'].upper()}{tw_label}")
    spec_note = ("  ✅ IG Stories spec" if (out_w, out_h) == (1080, 1920)
                 else "  ✅ YouTube 1080p" if (out_w, out_h) == (1920, 1080)
                 else "  ✅ YouTube 4K" if (out_w, out_h) == (3840, 2160) else "")
    audio_note = "  (silent)" if silent else ""
    print(f"  Output  {out_w}×{out_h}  {out_fps:.2f}fps  {codec_label}{spec_note}{audio_note}")

    if crop_info:
        quality = "lossless" if crop_info['scale'] <= 1.0 else f"{crop_info['scale']:.2f}× upscale"
        print(f"  Crop    {crop_info['crop_w']}×{crop_info['crop_h']}  {crop_info['pan_label']}  {quality}")

    mode = "Apple Log → Rec.709 + Enhancement" if cfg['is_log'] else "Enhancement"
    if cfg.get('drone_crop'):
        mode = ("Apple Log → Rec.709 + " if cfg['is_log'] else "") + "Drone Crop → Super-Resolve"
    print(f"  Mode    {mode}")
    print(f"  Preset  {cfg.get('_name', 'custom')}")
    print(f"  Denoise {cfg['denoise_strength']:.2f}  "
          f"Sharpen {cfg['sharpen_amount']:.2f}  "
          f"Sat {cfg['saturation']:.2f}")
    print(f"  Bitrate {cfg['output_bitrate']}  CRF {cfg['output_crf']}")
    print(f"  Accel   {accel}")
    if cfg['is_log']:
        print(f"  LogConv {'GPU (CuPy)' if _log_gpu() is not None else 'CPU (pip install cupy-cuda12x for GPU)'}")
    if GPU['nvenc']:
        enc_label = f"{'h264_nvenc' if vcodec == 'h264' else 'hevc_nvenc'} (GPU)"
    else:
        enc_label = f"{'libx264' if vcodec == 'h264' else 'libx265'} (CPU)"
    dec_label = "NVDEC (GPU)" if GPU['nvdec'] else "CPU"
    print(f"  Encode  {enc_label}  |  Decode {dec_label}")
    print(f"  File    {output_path}")
    print(f"{'═'*62}\n")

    temporal  = TemporalBuffer(blend=cfg.get('temporal_blend', 0.12))

    # Frame size coming from FFmpeg reader:
    # - Static crop (vf_filter set): FFmpeg delivers already-cropped out_w × out_h
    # - Animated crop (vf_filter=None): FFmpeg delivers full source w × h
    # - No crop: full source w × h
    if crop_info and not crop_info['animated']:
        frame_w, frame_h = out_w, out_h   # static crop handled by FFmpeg
    else:
        frame_w, frame_h = w, h           # full frame (Python crops per-frame)
    frame_bytes = frame_w * frame_h * 3
    # Trim muxed audio to the processed window only when a time window is set
    if start_t is not None or end_t is not None:
        aud_ss, aud_t = seg_start, seg_duration
    else:
        aud_ss, aud_t = None, None

    reader = make_reader(input_path, vf=vf_filter, start_time=start_t, end_time=end_t,
                         out_fps=force_fps)
    writer = make_writer(output_path, out_w, out_h, out_fps,
                         input_path, cfg['output_crf'], cfg['output_bitrate'],
                         silent=silent, vcodec=cfg.get('output_codec', 'h265'),
                         audio_ss=aud_ss, audio_t=aud_t)

    n = 0
    total = seg_frames  # progress bar relative to segment, not full clip
    t0 = time.time()

    print("  Processing...\n")
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((frame_h, frame_w, 3))

            # Animated crop: apply per-frame interpolated crop window
            if crop_info and crop_info['animated']:
                cx    = crop_info['get_crop_x'](n)
                cy    = crop_info['crop_y']
                cw    = crop_info['crop_w']
                ch    = crop_info['crop_h']
                frame = frame[cy:cy+ch, cx:cx+cw]
                # Scale cropped region to target output size
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h),
                                       interpolation=cv2.INTER_LANCZOS4)

            out   = enhancer.process(frame)
            # Honour force_resolution when no crop/upscale already produced it
            # (e.g. landscape source → 9:16): cover-fit to the writer's size.
            if out.shape[1] != out_w or out.shape[0] != out_h:
                out = fit_cover(out, out_w, out_h)
            out   = temporal.process(out)
            writer.stdin.write(out.tobytes())

            n += 1
            elapsed  = time.time() - t0
            fps_proc = n / elapsed
            pct      = n / max(total, 1)
            eta      = max(0, (total - n) / max(fps_proc, 0.01))
            bar      = '█' * int(pct * 34) + '░' * (34 - int(pct * 34))

            sys.stdout.write(
                f"\r  [{bar}] {pct*100:5.1f}%  "
                f"frame {n}/{total}  "
                f"{fps_proc:4.1f} fps  "
                f"ETA {eta:4.0f}s"
            )
            sys.stdout.flush()

    except (BrokenPipeError, KeyboardInterrupt):
        print("\n  ⚠️  Interrupted.")
    finally:
        reader.stdout.close()
        writer.stdin.close()
        reader.wait()
        writer.wait()

    elapsed = time.time() - t0
    print(f"\n\n  ✅  {n} frames in {elapsed:.1f}s  ({n/max(elapsed,0.1):.1f} fps avg)\n")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        mb = os.path.getsize(output_path) / 1e6
        print(f"  📁  {output_path}  ({mb:.1f} MB)\n")
    else:
        print("  ⚠️  Output may be incomplete.\n")


# ══════════════════════════════════════════════════════════════════
#  IMAGE PROCESSOR
# ══════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """
    Load a still image as a BGR uint8 array.
    Standard formats (JPEG/PNG/TIFF/WebP) use cv2.imread(). HEIC/HEIF
    (the iPhone default) needs Pillow + the pillow-heif plugin — if that
    plugin is missing we exit with a clear pip install message.
    """
    ext = Path(path).suffix.lower()

    if ext in ('.heic', '.heif'):
        try:
            from PIL import Image
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            print("\n❌  HEIC/HEIF support requires the 'pillow-heif' plugin.")
            print("    Install it with:  pip install pillow-heif\n")
            sys.exit(1)
        pil = Image.open(path).convert('RGB')
        # PIL is RGB; OpenCV pipeline expects BGR
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"\n❌  Could not read image: {path}\n")
        sys.exit(1)
    return img


def sharpen_image(frame: np.ndarray, amount: float) -> np.ndarray:
    """
    Light unsharp mask for stills, applied after the resize.
    Big downscales (e.g. a 26MP pano → 1080px) soften fine detail; a touch of
    USM restores the 'crisp' look without the heavy multi-layer video sharpen.
    amount ≈ 0.4–0.8 is a good range; 0 disables.
    """
    if amount <= 0.01:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def _convert_to_srgb(out_bgr: np.ndarray, src_path: str):
    """
    Colour-manage `out_bgr` into sRGB using the source image's embedded ICC
    profile (via Pillow + littleCMS). Returns (rgb_uint8, srgb_icc_bytes) on
    success, or None if Pillow/ImageCms is unavailable (caller falls back).
    If the source has no profile we assume it is already sRGB and just tag it.
    """
    try:
        import io
        from PIL import Image, ImageCms
    except ImportError:
        print("  ⚠️  --srgb needs Pillow — skipping conversion (pip install pillow)")
        return None

    rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    srgb_profile = ImageCms.createProfile('sRGB')

    src_icc = None
    try:
        with Image.open(src_path) as s:
            src_icc = s.info.get('icc_profile')
    except Exception:
        pass

    if src_icc:
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(src_icc))
            pil = ImageCms.profileToProfile(pil, src_profile, srgb_profile, outputMode='RGB')
        except Exception as e:
            print(f"  ⚠️  ICC transform failed ({e}); tagging as sRGB without conversion")

    icc_bytes = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
    return np.asarray(pil), icc_bytes


def process_image(input_path: str, output_path: str, cfg: dict):
    """
    Still-image pipeline: crop a 9:16 window then resize to exactly 1080×1920.

    Optional, off by default:
      - sharpen    (cfg['sharpen_amount'] / --sharpen)     light USM after resize
      - temperature(cfg['temperature']    / --temperature) warm/cool white balance
      - saturation (cfg['saturation']     / --saturation)  HSV saturation multiplier
      - vibrance   (cfg['vibrance']       / --vibrance)     adaptive saturation boost
      - hue-sat    (cfg['sat_bands']      / --sat-COLOR)    per-colour saturation
      - quality    (cfg['out_quality']    / --quality)      JPEG quality (default 97)
      - sRGB       (cfg['srgb']           / --srgb)         colour-manage to sRGB

    Crop uses the same compute_crop() as video: for landscape sources
    crop_position pans horizontally; for tall/portrait sources (this DJI pano)
    it selects the vertical window ('top'/'bottom'/'center'/int).
    """
    img = load_image(input_path)
    src_h, src_w = img.shape[:2]
    in_ext = Path(input_path).suffix.lower()

    target_w = cfg.get('crop_target_w', 1080)
    target_h = cfg.get('crop_target_h', 1920)
    position = cfg.get('crop_position', 'center')
    amount   = float(cfg.get('sharpen_amount', 0.0))
    temp     = float(cfg.get('temperature', 0.0))
    sat      = float(cfg.get('saturation', 1.0))
    vib      = float(cfg.get('vibrance', 0.0))
    bands    = cfg.get('sat_bands', {}) or {}
    jpeg_q   = int(cfg.get('out_quality', 97))
    want_srgb = bool(cfg.get('srgb', False))

    # Static crop only — total_frames=1 means get_crop_x() returns start_x
    crop_info = compute_crop(src_w, src_h, target_w, target_h, position)
    cx, cy = crop_info['crop_x'], crop_info['crop_y']
    cw, ch = crop_info['crop_w'], crop_info['crop_h']

    cropped = img[cy:cy + ch, cx:cx + cw]
    out = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    out = apply_temperature(out, temp)             # white balance first (fixes casts)
    out = sharpen_image(out, amount)
    out = apply_chroma(out, sat, vib)
    out = apply_selective_saturation(out, bands)   # per-colour trim, after global sat

    out_ext = Path(output_path).suffix.lower()
    scale_label = "lossless" if crop_info['scale'] <= 1.0 else f"{crop_info['scale']:.2f}× upscale"
    axis = "vertical" if crop_info['vertical_crop'] else "horizontal"

    print(f"\n{'═'*62}")
    print(f"  🖼️   Image Processor  —  Crop → 9:16 Vertical")
    print(f"{'═'*62}")
    ig_note = "  ✅ IG Stories spec" if (target_w, target_h) == (1080, 1920) else ""
    print(f"  Input   {src_w}×{src_h}  {in_ext.lstrip('.').upper()}")
    print(f"  Output  {target_w}×{target_h}  {out_ext.lstrip('.').upper()}{ig_note}")
    print(f"  Crop    {cw}×{ch}  x={cx} y={cy}  ({position}, {axis})  {scale_label}")
    sharp_label = f"{amount:.2f}" if amount > 0.01 else "off"
    q_label = f"q{jpeg_q}" if out_ext != '.png' else "PNG (lossless)"
    temp_label = (f"{temp:+.2f} ({'cooler' if temp < 0 else 'warmer'})"
                  if abs(temp) > 0.01 else "0 (neutral)")
    print(f"  Resize  Lanczos4   Sharpen {sharp_label}   Temp {temp_label}")
    print(f"  Colour  Sat {sat:.2f}   Vib {vib:.2f}")
    if bands:
        band_str = "  ".join(f"{c}={bands[c]:.2f}" for c in HUE_BANDS if c in bands)
        print(f"  Hue-sat {band_str}")
    print(f"  Output  {q_label}   sRGB {'on' if want_srgb else 'off'}")
    print(f"  Preset  {cfg.get('_name', 'image')}")
    print(f"  File    {output_path}")
    print(f"{'═'*62}\n")

    # ── Save ──────────────────────────────────────────────────────
    # sRGB path goes through Pillow (so we can embed an ICC profile);
    # otherwise use cv2.imwrite. JPEG uses 4:4:4 chroma (subsampling=0) for
    # maximum crispness on fine coloured detail.
    icc_bytes = None
    if want_srgb:
        conv = _convert_to_srgb(out, input_path)
        if conv is not None:
            rgb, icc_bytes = conv  # rgb is RGB uint8
        else:
            want_srgb = False      # Pillow missing → fall back to cv2

    ok = False
    if want_srgb:
        from PIL import Image
        pil = Image.fromarray(rgb)
        if out_ext == '.png':
            pil.save(output_path, format='PNG', icc_profile=icc_bytes)
        else:
            pil.save(output_path, format='JPEG', quality=jpeg_q,
                     subsampling=0, icc_profile=icc_bytes)
        ok = True
    else:
        if out_ext == '.png':
            ok = cv2.imwrite(output_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            ok = cv2.imwrite(output_path, out,
                             [cv2.IMWRITE_JPEG_QUALITY, jpeg_q,
                              cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])

    if ok and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        kb = os.path.getsize(output_path) / 1e3
        print(f"  ✅  Saved {target_w}×{target_h}\n")
        print(f"  📁  {output_path}  ({kb:.0f} KB)\n")
    else:
        print("  ⚠️  Failed to write output image.\n")


# ══════════════════════════════════════════════════════════════════
#  MONTAGE / SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════

CLIP_MIN_SEC = 1.0   # ideal per-clip duration floor (snappy pacing)
CLIP_MAX_SEC = 6.0   # ideal per-clip duration ceiling


def sample_manifest_text() -> str:
    """A ready-to-edit JSON manifest for the montage builder (--sequence)."""
    sample = {
        "output": "montage.mp4",
        "fps": 30,
        "defaults": {"preset": "instagram_stories"},
        "clips": [
            {"input": "clip_a.mp4", "start": 0,  "duration": 3},
            {"input": "clip_b.mp4", "start": 12, "duration": 5,
             "preset": "drone_stories", "crop_position": "left"},
            {"input": "clip_c.mov", "start": 4,  "duration": 2,
             "log": True, "sharpen": 0.5, "saturation": 1.1}
        ]
    }
    return json.dumps(sample, indent=2)


def _load_json_file(path: str):
    """
    Read a JSON file tolerant of encoding: UTF-8, UTF-8-BOM, or UTF-16.
    PowerShell's `command > file.json` writes UTF-16, so manifests created that
    way must still load. Raises FileNotFoundError / json.JSONDecodeError as usual.
    """
    raw = open(path, 'rb').read()
    for enc in ('utf-8-sig', 'utf-16', 'utf-8'):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeError, json.JSONDecodeError):
            continue
    # Re-raise a clean error from the most likely (utf-8) decoding
    return json.loads(raw.decode('utf-8', 'replace'))


def load_manifest(path: str) -> dict:
    """Load and lightly validate a montage manifest (JSON)."""
    try:
        data = _load_json_file(path)
    except FileNotFoundError:
        print(f"\n❌  Manifest not found: {path}\n")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n❌  Manifest is not valid JSON: {e}\n")
        sys.exit(1)
    if not isinstance(data, dict) or not isinstance(data.get('clips'), list) or not data['clips']:
        print("\n❌  Manifest must be a JSON object with a non-empty 'clips' list.")
        print("    Run  --print-sample-manifest  to see the expected format.\n")
        sys.exit(1)
    return data


def build_clip_cfg(clip: dict, defaults: dict) -> dict:
    """
    Build a per-clip cfg from a preset plus manifest overrides (a clip's own
    keys win over the manifest 'defaults'). Mirrors the single-video CLI flags.
    """
    merged = {**defaults, **clip}
    preset = merged.get('preset', 'instagram_stories')
    if preset not in PRESETS:
        print(f"\n❌  Unknown preset '{preset}' for clip {clip.get('input')!r}.")
        print(f"    Valid presets: {', '.join(PRESETS)}\n")
        sys.exit(1)
    cfg = PRESETS[preset].copy()

    if merged.get('log'):                            cfg['is_log'] = True
    if merged.get('crop_position') is not None:      cfg['crop_position'] = merged['crop_position']
    if merged.get('crop_position_end') is not None:  cfg['crop_position_end'] = merged['crop_position_end']
    if merged.get('easing'):                         cfg['crop_easing'] = merged['easing']
    for mkey, cfgkey in (('sharpen', 'sharpen_amount'), ('denoise', 'denoise_strength'),
                         ('saturation', 'saturation'), ('vibrance', 'vibrance'),
                         ('upscale', 'upscale')):
        if merged.get(mkey) is not None:
            cfg[cfgkey] = merged[mkey]
    return cfg


def process_sequence(manifest_path: str, output_path: str = None):
    """
    Build a single silent montage from a JSON manifest of video clips.
    Each clip is trimmed to a 1–6s window, processed to 1080×1920 H.265 at one
    common fps (silent), then losslessly concatenated (stream copy).
    """
    manifest = load_manifest(manifest_path)
    out_path = output_path or manifest.get('output')
    if not out_path:
        print("\n❌  No output path — pass it on the CLI or set 'output' in the manifest.\n")
        sys.exit(1)

    fps      = float(manifest.get('fps', 30))
    defaults = manifest.get('defaults', {}) or {}
    clips    = manifest['clips']

    # ── Validate every clip up front (fail fast before any encoding) ──
    specs, total = [], 0.0
    for i, clip in enumerate(clips):
        src = clip.get('input')
        if not src:
            print(f"\n❌  Clip {i} has no 'input'.\n");                       sys.exit(1)
        if not os.path.exists(src):
            print(f"\n❌  Clip {i}: input not found: {src}\n");               sys.exit(1)
        if Path(src).suffix.lower() in IMAGE_EXTS:
            print(f"\n❌  Clip {i}: '{src}' is an image — the montage is video-only.\n"); sys.exit(1)
        if clip.get('duration') is None:
            print(f"\n❌  Clip {i} ('{src}') has no 'duration'.\n");          sys.exit(1)
        dur = float(clip['duration'])
        clamped = max(CLIP_MIN_SEC, min(dur, CLIP_MAX_SEC))
        if abs(clamped - dur) > 1e-6:
            print(f"  ⚠️  Clip {i}: duration {dur:g}s clamped to {clamped:g}s (1–6s range).")
        start = float(clip.get('start', 0) or 0)
        specs.append((clip, src, start, clamped))
        total += clamped

    print(f"\n{'═'*62}")
    print(f"  🎞️   Montage Builder  —  {len(specs)} clips → {total:.1f}s @ {fps:.0f}fps")
    print(f"{'═'*62}")
    for i, (_, src, start, dur) in enumerate(specs):
        print(f"  {i+1:2}. {Path(src).name:<30} {fmt_time(start)} +{dur:g}s")
    print(f"  Output  {out_path}   (silent)")
    print(f"{'═'*62}")

    # ── Render each clip to a silent, fps-normalised temp file ──
    tmpdir = tempfile.mkdtemp(prefix='mp_seq_')
    temp_files = []
    try:
        for i, (clip, src, start, dur) in enumerate(specs):
            cfg = build_clip_cfg(clip, defaults)
            cfg['start_time'] = start
            cfg['end_time']   = start + dur
            temp_out = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
            print(f"\n  ── Clip {i+1}/{len(specs)} ──────────────────────────────")
            process_video(src, temp_out, cfg, silent=True, force_fps=fps)
            if not (os.path.exists(temp_out) and os.path.getsize(temp_out) > 1000):
                print(f"\n❌  Clip {i+1} failed to render; aborting.\n")
                sys.exit(1)
            temp_files.append(temp_out)

        # ── Concatenate (stream copy — every temp shares codec/params/fps) ──
        list_path = os.path.join(tmpdir, 'concat.txt')
        with open(list_path, 'w', encoding='utf-8') as f:
            for tf in temp_files:
                f.write(f"file '{tf.replace(os.sep, '/')}'\n")

        ffmpeg = find_tool('ffmpeg')
        print(f"\n  Concatenating {len(temp_files)} clips…")
        r = subprocess.run(
            [ffmpeg, '-v', 'error', '-y', '-f', 'concat', '-safe', '0',
             '-i', list_path, '-c', 'copy', '-movflags', '+faststart', out_path],
            capture_output=True, text=True, **_POPEN_KWARGS
        )
        if r.returncode != 0:
            print(f"\n❌  Concatenation failed:\n{r.stderr.strip()}\n")
            sys.exit(1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        mb = os.path.getsize(out_path) / 1e6
        print(f"\n  ✅  Montage complete: {len(temp_files)} clips, {total:.1f}s")
        print(f"  📁  {out_path}  ({mb:.1f} MB)\n")
    else:
        print("\n  ⚠️  Montage output may be incomplete.\n")


# ══════════════════════════════════════════════════════════════════
#  GRID TOUR  —  mosaic → zoom into each tile (full-res) → zoom out
# ══════════════════════════════════════════════════════════════════
#
# Builds a single video that opens on a user-defined grid of photos/videos,
# then "flies" into each chosen tile one at a time. The focused tile is always
# re-rendered from its full-resolution source (sharp the whole zoom, full-res
# at the end — not an upscaled thumbnail). Videos play while focused and their
# audio is heard during that hold; the grid and transitions are silent.
#
# Output formats: 'story' (1080×1920) or 'post' (1080×1350, 4:5).

GRID_FORMATS = {
    'story': (1080, 1920),   # IG Story / Reel 9:16
    'post':  (1080, 1350),   # IG feed post 4:5 (portrait)
}


def sample_grid_text() -> str:
    """A ready-to-edit JSON manifest for the grid-tour builder (--grid)."""
    sample = {
        "output": "grid_tour.mp4",
        "format": "story",
        "fps": 30,
        "grid": {"rows": 2, "cols": 2, "gap": 8, "background": "#000000"},
        "grid_hold": 1.5,        # seconds on the full grid at the start
        "zoom_duration": 0.6,    # seconds for each zoom-in / zoom-out
        "grid_between": 0.4,     # seconds back on the grid between tiles
        "default_hold": 3.0,     # seconds a focused tile is held (images)
        "assets": [
            {"input": "photo1.jpg"},
            {"input": "clip1.mp4", "poster": 1.0, "start": 2, "duration": 4},
            {"input": "photo2.jpg"},
            {"input": "clip2.mov", "start": 0, "duration": 3}
        ],
        "zoom": [0, 1, 2, 3]     # tile indices (row-major) to tour, in order
    }
    return json.dumps(sample, indent=2)


def _hex_to_bgr(s: str):
    """'#RRGGBB' (or 'RRGGBB') → (B, G, R) ints. Defaults to black."""
    try:
        s = str(s).lstrip('#')
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return (b, g, r)
    except Exception:
        return (0, 0, 0)


def _read_one_frame(path: str, t: float = 0.0):
    """Decode a single BGR frame from a video at time t (seconds), full res."""
    ffmpeg = find_tool('ffmpeg')
    info = probe_video(path)
    w, h = info['width'], info['height']
    cmd = [ffmpeg, '-v', 'quiet']
    if t and t > 0:
        cmd += ['-ss', f'{t:.3f}']
    cmd += ['-i', path, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    p = subprocess.run(cmd, capture_output=True, **_POPEN_KWARGS)
    need = w * h * 3
    if len(p.stdout) < need:
        return None
    return np.frombuffer(p.stdout[:need], dtype=np.uint8).reshape((h, w, 3))


def _cell_rects(rows: int, cols: int, W: int, H: int, gap: int):
    """Row-major list of (x, y, w, h) tile rectangles, with gap around/between."""
    cw = (W - gap * (cols + 1)) / cols
    ch = (H - gap * (rows + 1)) / rows
    rects = []
    for r in range(rows):
        for c in range(cols):
            x = gap + c * (cw + gap)
            y = gap + r * (ch + gap)
            rects.append((x, y, cw, ch))
    return rects


def _ease(p: float) -> float:
    """Smoothstep ease-in-out."""
    p = max(0.0, min(1.0, p))
    return p * p * (3 - 2 * p)


def _render_zoom_frame(grid_img, cell, foc_full, p, W, H):
    """
    One zoom frame. cell=(x,y,w,h) tile rect on the grid; foc_full = the focused
    asset cover-fit to the full W×H frame (sharp). p in [0,1]: 0 = full grid,
    1 = focused asset fills the frame. The camera crops/zooms the grid toward the
    tile while the sharp focused render is composited exactly over the tile.
    """
    e = _ease(p)
    cx, cy, cw, ch = cell
    # Camera rect on the grid: lerp(full frame → tile)
    ax = e * cx
    ay = e * cy
    aw = W + e * (cw - W)
    ah = H + e * (ch - H)
    Mcam = np.array([[W / aw, 0, -ax * W / aw],
                     [0, H / ah, -ay * H / ah]], dtype=np.float32)
    bg = cv2.warpAffine(grid_img, Mcam, (W, H), flags=cv2.INTER_LINEAR)

    # Where the tile lands on screen under that camera
    fx = (cx - ax) * W / aw
    fy = (cy - ay) * H / ah
    fw = cw * W / aw
    fh = ch * H / ah
    Mfoc = np.array([[fw / W, 0, fx],
                     [0, fh / H, fy]], dtype=np.float32)
    foc = cv2.warpAffine(foc_full, Mfoc, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.full((H, W), 255, np.uint8), Mfoc, (W, H), flags=cv2.INTER_LINEAR)

    # Ramp the sharp overlay in over the first/last 20% so the framing swap from
    # the cell-cropped thumbnail to the full-frame crop isn't a hard pop.
    ramp = _ease(min(1.0, e / 0.2)) if e <= 0.5 else _ease(min(1.0, (1.0 - e) / 0.2))
    ramp = 1.0 if e >= 0.999 else ramp
    m = (mask.astype(np.float32) / 255.0) * ramp
    m = m[..., None]
    return (foc.astype(np.float32) * m + bg.astype(np.float32) * (1.0 - m)).astype(np.uint8)


def load_grid_manifest(path: str) -> dict:
    """Load + validate a grid-tour manifest (JSON)."""
    try:
        data = _load_json_file(path)
    except FileNotFoundError:
        print(f"\n❌  Grid manifest not found: {path}\n"); sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n❌  Grid manifest is not valid JSON: {e}\n"); sys.exit(1)
    if not isinstance(data, dict) or 'grid' not in data or not isinstance(data.get('assets'), list):
        print("\n❌  Manifest needs a 'grid' object and an 'assets' list.")
        print("    Run  --print-sample-grid  to see the expected format.\n")
        sys.exit(1)
    return data


def process_grid_tour(manifest_path: str, output_path: str = None, fmt_override: str = None):
    """
    Render a grid-tour video from a JSON manifest: open on the mosaic, then zoom
    into each chosen tile (full-res), holding/playing it, and zoom back out.
    """
    m = load_grid_manifest(manifest_path)
    out_path = output_path or m.get('output')
    if not out_path:
        print("\n❌  No output path — pass it on the CLI or set 'output' in the manifest.\n")
        sys.exit(1)

    fmt = (fmt_override or m.get('format', 'story')).lower()
    if fmt not in GRID_FORMATS:
        print(f"\n❌  Unknown format '{fmt}'. Choose 'story' (1080×1920) or 'post' (1080×1350).\n")
        sys.exit(1)
    W, H = GRID_FORMATS[fmt]

    grid = m['grid']
    rows, cols = int(grid['rows']), int(grid['cols'])
    gap = int(grid.get('gap', 8))
    bg_color = _hex_to_bgr(grid.get('background', '#000000'))
    fps = float(m.get('fps', 30))

    grid_hold    = float(m.get('grid_hold', 1.5))
    zoom_dur     = float(m.get('zoom_duration', 0.6))
    grid_between = float(m.get('grid_between', 0.4))
    default_hold = float(m.get('default_hold', 3.0))

    assets = m['assets']
    ncells = rows * cols
    if len(assets) != ncells:
        print(f"\n❌  Grid is {rows}×{cols} = {ncells} cells but {len(assets)} assets were given.")
        print("    Provide exactly one asset per cell.\n")
        sys.exit(1)

    zoom_order = m.get('zoom', list(range(ncells)))
    for z in zoom_order:
        if not isinstance(z, int) or not (0 <= z < ncells):
            print(f"\n❌  'zoom' entries must be tile indices 0..{ncells-1} (got {z!r}).\n")
            sys.exit(1)

    rects = _cell_rects(rows, cols, W, H, gap)

    # ── Resolve each asset: full-res poster (grid + transitions) + metadata ──
    resolved = []
    for i, a in enumerate(assets):
        src = a.get('input')
        if not src or not os.path.exists(src):
            print(f"\n❌  Asset {i}: input not found: {src}\n"); sys.exit(1)
        is_img = Path(src).suffix.lower() in IMAGE_EXTS
        if is_img:
            poster = load_image(src)
            has_audio = False
        else:
            poster = _read_one_frame(src, float(a.get('poster', 0.0)))
            if poster is None:
                print(f"\n❌  Asset {i}: could not read a frame from {src}\n"); sys.exit(1)
            has_audio = probe_video(src).get('has_audio', False)
        resolved.append({'src': src, 'is_img': is_img, 'poster': poster,
                         'start': float(a.get('start', 0.0)),
                         'duration': a.get('duration'), 'has_audio': has_audio})

    # ── Build the static grid composite (full-res posters cover-fit to cells) ──
    grid_img = np.full((H, W, 3), bg_color, dtype=np.uint8)
    foc_fulls = []   # each asset cover-fit to full W×H (for zoom/hold of stills)
    for i, r in enumerate(resolved):
        x, y, cw, ch = rects[i]
        ix, iy, iw, ih = int(round(x)), int(round(y)), int(round(cw)), int(round(ch))
        grid_img[iy:iy + ih, ix:ix + iw] = fit_cover(r['poster'], iw, ih)
        foc_fulls.append(fit_cover(r['poster'], W, H))

    # ── Timeline: durations → audio offsets (only video holds carry audio) ──
    def hold_seconds(r):
        d = r['duration']
        d = default_hold if d is None else float(d)
        return max(CLIP_MIN_SEC, min(d, CLIP_MAX_SEC))

    audio_segs = []   # (src, clip_start, dur, timeline_offset)
    t_cursor = grid_hold
    for z in zoom_order:
        r = resolved[z]
        hold = hold_seconds(r)
        hold_start = t_cursor + zoom_dur
        if (not r['is_img']) and r['has_audio']:
            audio_segs.append((r['src'], r['start'], hold, hold_start))
        t_cursor += zoom_dur + hold + zoom_dur + grid_between
    total_dur = t_cursor

    print(f"\n{'═'*62}")
    print(f"  🗺️   Grid Tour  —  {rows}×{cols} grid → {len(zoom_order)} zooms  ({fmt})")
    print(f"{'═'*62}")
    print(f"  Output  {W}×{H}  {fps:.0f}fps  {'IG Story 9:16' if fmt=='story' else 'IG Post 4:5'}")
    print(f"  Assets  {ncells} ({sum(1 for r in resolved if r['is_img'])} images, "
          f"{sum(1 for r in resolved if not r['is_img'])} videos)")
    print(f"  Tour    tiles {zoom_order}")
    print(f"  Length  ~{total_dur:.1f}s   Audio {len(audio_segs)} clip segment(s)")
    print(f"  File    {out_path}")
    print(f"{'═'*62}\n")

    # ── Render frames to a silent temp video ──
    tmpdir = tempfile.mkdtemp(prefix='mp_grid_')
    silent_path = os.path.join(tmpdir, 'silent.mp4')
    writer = make_writer(silent_path, W, H, fps, manifest_path,
                         crf=18, bitrate="20M", silent=True, vcodec='h264')

    def emit(frame):
        writer.stdin.write(frame.tobytes())

    def emit_n(frame, n):
        b = frame.tobytes()
        for _ in range(n):
            writer.stdin.write(b)

    n_total = 0
    t0 = time.time()
    try:
        # Opening grid hold
        gh = int(round(grid_hold * fps)); emit_n(grid_img, gh); n_total += gh

        for ti, z in enumerate(zoom_order):
            r = resolved[z]
            cell = rects[z]
            foc_full = foc_fulls[z]
            zn = max(1, int(round(zoom_dur * fps)))
            hold = hold_seconds(r)
            hn = max(1, int(round(hold * fps)))

            # Zoom in (focused content = poster still)
            for k in range(zn):
                emit(_render_zoom_frame(grid_img, cell, foc_full, (k + 1) / zn, W, H))
            n_total += zn

            # Hold: image = static full frame; video = play frames
            if r['is_img']:
                emit_n(foc_full, hn); n_total += hn
                last_full = foc_full
            else:
                reader = make_reader(r['src'], start_time=r['start'],
                                     end_time=r['start'] + hold, out_fps=fps)
                fw_, fh_ = probe_video(r['src'])['width'], probe_video(r['src'])['height']
                fb = fw_ * fh_ * 3
                last_full = foc_full
                got = 0
                while got < hn:
                    raw = reader.stdout.read(fb)
                    if len(raw) < fb:
                        break
                    fr = np.frombuffer(raw, dtype=np.uint8).reshape((fh_, fw_, 3))
                    last_full = fit_cover(fr, W, H)
                    emit(last_full); got += 1
                reader.stdout.close(); reader.wait()
                if got < hn:                       # pad if the clip ran short
                    emit_n(last_full, hn - got)
                n_total += hn

            # Zoom out (focused content = last shown frame)
            for k in range(zn):
                emit(_render_zoom_frame(grid_img, cell, last_full, 1.0 - (k + 1) / zn, W, H))
            n_total += zn

            # Brief grid pause between tiles
            gb = int(round(grid_between * fps))
            if gb:
                emit_n(grid_img, gb); n_total += gb

            sys.stdout.write(f"\r  Rendering…  tile {ti+1}/{len(zoom_order)}  "
                             f"{n_total} frames  {n_total/max(time.time()-t0,0.01):.0f} fps")
            sys.stdout.flush()
    finally:
        writer.stdin.close(); writer.wait()
    print(f"\n  ✅  {n_total} frames in {time.time()-t0:.1f}s\n")

    # ── Mux focused-clip audio onto the silent video (or just keep silent) ──
    try:
        if audio_segs:
            _mux_grid_audio(silent_path, audio_segs, total_dur, out_path)
        else:
            shutil.copyfile(silent_path, out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        mb = os.path.getsize(out_path) / 1e6
        print(f"  📁  {out_path}  ({mb:.1f} MB,  ~{total_dur:.1f}s,  {W}×{H})\n")
    else:
        print("  ⚠️  Grid-tour output may be incomplete.\n")


def _mux_grid_audio(silent_video, segments, total_dur, out_path):
    """Mux each focused clip's audio onto the silent video at its hold offset."""
    ffmpeg = find_tool('ffmpeg')
    cmd = [ffmpeg, '-v', 'error', '-y', '-i', silent_video]
    for (src, cs, dur, off) in segments:
        cmd += ['-ss', f'{cs:.3f}', '-t', f'{dur:.3f}', '-i', src]

    parts, labels = [], []
    for i, (src, cs, dur, off) in enumerate(segments):
        ms = int(round(off * 1000))
        parts.append(f"[{i+1}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    if len(segments) == 1:
        parts.append(f"{labels[0]}anull[mix]")
    else:
        parts.append("".join(labels) + f"amix=inputs={len(segments)}:normalize=0[mix]")
    filtergraph = ";".join(parts)

    cmd += ['-filter_complex', filtergraph, '-map', '0:v', '-map', '[mix]',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-t', f'{total_dur:.3f}', '-movflags', '+faststart', out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, **_POPEN_KWARGS)
    if r.returncode != 0:
        print(f"\n  ⚠️  Audio mux failed ({r.stderr.strip()[:200]}); writing silent video.")
        shutil.copyfile(silent_video, out_path)


# ══════════════════════════════════════════════════════════════════
#  CANVAS TOUR  —  Prezi/SCRL-style zooming collage (native aspect)
# ══════════════════════════════════════════════════════════════════
#
# Unlike the rigid grid, this lays out visuals at their NATIVE aspect ratio
# (a justified collage — landscape stays landscape, portrait stays portrait,
# nothing is cropped to a cell), then flies the camera from the full collage
# into each toured visual (full-res) and back, Prezi-style. Videos play live
# in the collage and loop; zooming into one continues from its current point.


def _justified_collage(aspects, W, H, gap, target_rows):
    """
    Justified-gallery layout: pack tiles (by native aspect) into rows that each
    fill the width; row heights vary so nothing is cropped. Returns float rects
    (x, y, w, h) fitted and centred within the W×H canvas.
    """
    trh = max(1.0, (H - gap * (target_rows + 1)) / target_rows)   # target row height
    rows, cur, s_asp = [], [], 0.0
    for i, a in enumerate(aspects):
        cur.append((i, a)); s_asp += a
        if s_asp * trh >= (W - gap * (len(cur) + 1)):
            rows.append(cur); cur, s_asp = [], 0.0
    if cur:
        rows.append(cur)

    rects = [None] * len(aspects)
    y = float(gap)
    for row in rows:
        asum = sum(a for _, a in row)
        avail = W - gap * (len(row) + 1)
        rh = avail / asum
        x = float(gap)
        for (i, a) in row:
            w = a * rh
            rects[i] = [x, y, w, rh]
            x += w + gap
        y += rh + gap
    total_h = y

    s = H / total_h if total_h > H else 1.0
    ox = (W - W * s) / 2.0
    oy = (H - total_h * s) / 2.0
    return [(x * s + ox, y * s + oy, w * s, h * s) for (x, y, w, h) in rects]


class _VideoLoop:
    """
    A live video frame source for a collage tile (decodes at output fps). Plays
    the source continuously from `start` to its END — the manifest `duration`
    only sets the zoom dwell time, NOT how much footage to use — so a long clip
    never repeats its first few seconds. Loops only when the source is exhausted.
    """
    def __init__(self, path, start, fps):
        self.path, self.start, self.fps = path, start, fps
        info = probe_video(path)
        self.w, self.h = info['width'], info['height']
        self.fb = self.w * self.h * 3
        self.reader = None
        self.current = None
        self._open()
        self.next()  # prime first frame

    def _open(self):
        if self.reader:
            try:
                self.reader.stdout.close(); self.reader.wait()
            except Exception:
                pass
        # No end_time → play through to the real end of the file; on EOF we
        # reopen from `start`, so we loop the WHOLE clip, not a short segment.
        self.reader = make_reader(self.path, start_time=self.start, end_time=None, out_fps=self.fps)

    def next(self):
        raw = self.reader.stdout.read(self.fb)
        if len(raw) < self.fb:        # end of segment → loop
            self._open()
            raw = self.reader.stdout.read(self.fb)
            if len(raw) < self.fb:
                return self.current   # give up; reuse last good frame
        self.current = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
        return self.current

    def close(self):
        try:
            self.reader.stdout.close(); self.reader.wait()
        except Exception:
            pass


def process_canvas_tour(manifest_path: str, output_path: str = None, fmt_override: str = None):
    """
    Prezi/SCRL-style canvas tour: native-aspect collage → fly into each tile
    (full-res) → back out. Videos play live and loop; zoom continues playback.
    """
    m = load_grid_manifest(manifest_path)
    out_path = output_path or m.get('output')
    if not out_path:
        print("\n❌  No output path — pass it on the CLI or set 'output' in the manifest.\n")
        sys.exit(1)

    fmt = (fmt_override or m.get('format', 'story')).lower()
    if fmt not in GRID_FORMATS:
        print(f"\n❌  Unknown format '{fmt}'. Choose 'story' or 'post'.\n"); sys.exit(1)
    W, H = GRID_FORMATS[fmt]

    grid = m.get('grid', {})
    gap = int(grid.get('gap', 12))
    bg = _hex_to_bgr(grid.get('background', '#000000'))
    fps = float(m.get('fps', 30))
    grid_hold    = float(m.get('grid_hold', 1.5))
    zoom_dur     = float(m.get('zoom_duration', 0.7))
    grid_between = float(m.get('grid_between', 0.5))
    default_hold = float(m.get('default_hold', 3.0))

    assets = m['assets']
    n = len(assets)
    if n == 0:
        print("\n❌  No assets in manifest.\n"); sys.exit(1)
    zoom_order = m.get('zoom', list(range(n)))
    for z in zoom_order:
        if not isinstance(z, int) or not (0 <= z < n):
            print(f"\n❌  'zoom' entries must be asset indices 0..{n-1} (got {z!r}).\n"); sys.exit(1)

    # ── Resolve assets (poster + native aspect) ──
    resolved, aspects = [], []
    for i, a in enumerate(assets):
        src = a.get('input')
        if not src or not os.path.exists(src):
            print(f"\n❌  Asset {i}: input not found: {src}\n"); sys.exit(1)
        is_img = Path(src).suffix.lower() in IMAGE_EXTS
        dur = a.get('duration')
        if dur is not None:
            dur = max(CLIP_MIN_SEC, min(float(dur), CLIP_MAX_SEC))
        if is_img:
            poster = load_image(src); has_audio = False
            start = 0.0
        else:
            start = float(a.get('start', 0.0))
            poster = _read_one_frame(src, float(a.get('poster', start)))
            if poster is None:
                print(f"\n❌  Asset {i}: could not read a frame from {src}\n"); sys.exit(1)
            has_audio = probe_video(src).get('has_audio', False)
        ph, pw = poster.shape[:2]
        aspects.append(pw / ph)
        resolved.append({'src': src, 'is_img': is_img, 'poster': poster,
                         'start': start, 'dur': dur, 'has_audio': has_audio})

    target_rows = int(grid.get('rows', max(1, round(n ** 0.5))))
    rects = _justified_collage(aspects, W, H, gap, target_rows)

    # ── Static base canvas (image tiles painted once; videos painted per frame) ──
    base = np.full((H, W, 3), bg, dtype=np.uint8)
    irects = []
    for i, r in enumerate(resolved):
        x, y, w, h = rects[i]
        ix, iy, iw, ih = int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))
        iw = min(iw, W - ix); ih = min(ih, H - iy)
        irects.append((ix, iy, iw, ih))
        if r['is_img']:
            base[iy:iy + ih, ix:ix + iw] = fit_cover(r['poster'], iw, ih)

    # Video loops — advanced only while watched (see render loop), so background
    # clips pause during a dive elsewhere and resume from that point on return.
    vloops = {}
    for i, r in enumerate(resolved):
        if not r['is_img']:
            vloops[i] = _VideoLoop(r['src'], r['start'], fps)

    # Centred contain rect per asset (zoom-in settles here — never cropped)
    c_rects = [contain_rect(aspects[i], W, H) for i in range(n)]

    # Pre-scale each image to its max on-screen (contain) size ONCE, so dive
    # frames resize from a ~1080px image instead of the full 4K source every
    # frame — the single biggest speed-up for the collage render.
    disp_src = {}
    for i, r in enumerate(resolved):
        if r['is_img']:
            _, _, cw, ch = c_rects[i]
            disp_src[i] = cv2.resize(r['poster'], (max(1, int(round(cw))), max(1, int(round(ch)))),
                                     interpolation=cv2.INTER_AREA)

    def hold_seconds(r):
        return r['dur'] if (not r['is_img'] and r['dur']) else (
            r['dur'] if r['dur'] else default_hold)

    # ── Build per-frame plan + audio offsets ──
    plan = []   # ('home',) | ('zoom', z, p) | ('hold', z)
    audio_segs = []
    gh = int(round(grid_hold * fps)); plan += [('home',)] * gh
    zn = max(1, int(round(zoom_dur * fps)))
    gb = int(round(grid_between * fps))
    for z in zoom_order:
        r = resolved[z]
        hn = max(1, int(round(hold_seconds(r) * fps)))
        for k in range(zn):
            plan.append(('zoom', z, _ease((k + 1) / zn)))
        hold_start_frame = len(plan)
        plan += [('hold', z)] * hn
        for k in range(zn):
            plan.append(('zoom', z, _ease(1.0 - (k + 1) / zn)))
        plan += [('home',)] * gb
        if (not r['is_img']) and r['has_audio']:
            audio_segs.append((r['src'], r['start'], hn / fps, hold_start_frame / fps))
    total_dur = len(plan) / fps

    print(f"\n{'═'*62}")
    print(f"  🎞️   Canvas Tour  —  {n} visuals, {len(zoom_order)} zooms  ({fmt}, collage)")
    print(f"{'═'*62}")
    print(f"  Output  {W}×{H}  {fps:.0f}fps  {'IG Story 9:16' if fmt=='story' else 'IG Post 4:5'}")
    print(f"  Layout  justified collage ({sum(1 for r in resolved if r['is_img'])} images, "
          f"{len(vloops)} videos live+looping)")
    print(f"  Tour    tiles {zoom_order}   ~{total_dur:.1f}s")
    print(f"  File    {out_path}")
    print(f"{'═'*62}\n")

    # ── Render ──
    tmpdir = tempfile.mkdtemp(prefix='mp_canvas_')
    silent_path = os.path.join(tmpdir, 'silent.mp4')
    writer = make_writer(silent_path, W, H, fps, manifest_path,
                         crf=18, bitrate="20M", silent=True, vcodec='h264')
    n_total = len(plan); t0 = time.time()
    last_canvas = None
    hold_cache = (None, None)   # (z, frame) — a static image hold is identical every frame
    try:
        for fi, entry in enumerate(plan):
            kind = entry[0]
            focus = entry[1] if kind in ('zoom', 'hold') else None

            # A video advances only while it's being watched: every video during
            # the collage (home) phases, and only the focused video during a dive.
            # Background videos therefore PAUSE while you're zoomed elsewhere and
            # RESUME from that point on the way back. Reuse the cached canvas on
            # frames where nothing advanced.
            if not vloops:
                canvas = base
            else:
                to_advance = [i for i in vloops if (focus is None or i == focus)]
                if to_advance or last_canvas is None:
                    canvas = base.copy()
                    for i, vl in vloops.items():
                        if i in to_advance:
                            vl.next()
                        ix, iy, iw, ih = irects[i]
                        canvas[iy:iy + ih, ix:ix + iw] = fit_cover(vl.current, iw, ih)
                    last_canvas = canvas
                else:
                    canvas = last_canvas

            if kind == 'home':
                frame = canvas
            elif kind == 'hold' and resolved[entry[1]]['is_img']:
                # Static image hold → identical every frame; render once, reuse
                z = entry[1]
                if hold_cache[0] != z:
                    hold_cache = (z, _render_dive_frame(canvas, rects[z], disp_src[z],
                                                        c_rects[z], 1.0, W, H))
                frame = hold_cache[1]
            else:
                hold_cache = (None, None)
                z = entry[1]
                src_native = disp_src[z] if resolved[z]['is_img'] else vloops[z].current
                e = 1.0 if kind == 'hold' else entry[2]   # entry[2] is already eased
                frame = _render_dive_frame(canvas, rects[z], src_native, c_rects[z], e, W, H)
            writer.stdin.write(frame.tobytes())

            if fi % 15 == 0:
                sys.stdout.write(f"\r  Rendering…  {fi+1}/{n_total} frames  "
                                 f"{(fi+1)/max(time.time()-t0,0.01):.0f} fps")
                sys.stdout.flush()
    finally:
        writer.stdin.close(); writer.wait()
        for vl in vloops.values():
            vl.close()
    print(f"\n  ✅  {n_total} frames in {time.time()-t0:.1f}s\n")

    try:
        if audio_segs:
            _mux_grid_audio(silent_path, audio_segs, total_dur, out_path)
        else:
            shutil.copyfile(silent_path, out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        mb = os.path.getsize(out_path) / 1e6
        print(f"  📁  {out_path}  ({mb:.1f} MB,  ~{total_dur:.1f}s,  {W}×{H})\n")
    else:
        print("  ⚠️  Canvas-tour output may be incomplete.\n")


# ══════════════════════════════════════════════════════════════════
#  PRESETS
# ══════════════════════════════════════════════════════════════════

PRESETS = {
    "instagram_stories": dict(
        _name="instagram_stories",
        description="Standard footage → IG Stories optimized (1080x1920, H.265, 30fps)",
        is_log=False, upscale=1.0,
        # Enhancement intentionally neutralised — add explicitly via CLI flags per clip
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=0.9, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=2.5, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
        # CRF 16 = very high quality; 20M bitrate = IG's recommended upload spec
        # H.265 at 20Mbps gives IG's encoder the best possible source to work from
        output_crf=16, output_bitrate="20M",
        # Force 1080x1920 output (IG Stories native res) regardless of input
        force_resolution=(1080, 1920),
    ),
    "instagram_stories_log": dict(
        _name="instagram_stories_log",
        description="Apple Log → Rec.709 + IG Stories (Log input needs this preset)",
        is_log=True, upscale=1.0,
        # Enhancement intentionally neutralised — add explicitly via CLI flags per clip
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=1.0, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=2.8, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0, output_crf=16, output_bitrate="20M",   # temporal neutralised
        force_resolution=(1080, 1920),
    ),
    "upscale_2x": dict(
        _name="upscale_2x",
        description="2× super-resolution (1080p → 4K, or 4K → 8K)",
        is_log=False, upscale=2.0,
        # Enhancement intentionally neutralised — add explicitly via CLI flags per clip
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=1.0, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=1.5, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0, output_crf=16, output_bitrate="40M",   # temporal neutralised
    ),
    "denoise_only": dict(
        _name="denoise_only",
        description="Aggressive denoising for low-light footage (no upscale)",
        is_log=False, upscale=1.0,
        denoise_strength=0.80,   # kept — strong denoise is this preset's entire purpose
        # All other enhancement intentionally neutralised — add explicitly via CLI flags per clip
        sharpen_amount=0.0, sharpen_radius=0.8, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=1.0, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0, output_crf=18, output_bitrate="15M",   # temporal neutralised
    ),
    "neutral_log": dict(
        _name="neutral_log",
        description="Log → Rec.709 only, minimal processing (preserve for grading)",
        is_log=True, upscale=1.0,
        # Enhancement intentionally neutralised — add explicitly via CLI flags per clip
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=0.8, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=1.0, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0, output_crf=16, output_bitrate="30M",   # temporal neutralised
    ),

    # ── Drone / Landscape → IG Stories ─────────────────────────────
    # Crops a 9:16 vertical window from landscape 4K/2.7K/1080p footage
    # then super-resolves to 1080×1920.
    # From 4K: crop is a slight downscale (lossless quality, zero generation loss)
    # From 2.7K: 1.26× mild upscale (excellent quality)
    # From 1080p: 1.78× upscale (good quality with detail injection)

    "drone_stories": dict(
        _name="drone_stories",
        description="4K/2.7K landscape drone → crop 9:16 → 1080×1920 IG Stories",
        drone_crop=True,
        crop_position='center',   # 'center' | 'left' | 'right' | int (pixel x offset)
        crop_target_w=1080, crop_target_h=1920,
        is_log=False, upscale=1.0,
        # Enhancement intentionally neutralised — drone footage often looks over-processed;
        # add sharpen/denoise/etc. explicitly via CLI flags per clip
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=0.9, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=2.5, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
        output_crf=16, output_bitrate="20M",
        force_resolution=(1080, 1920),
    ),

    "drone_stories_log": dict(
        _name="drone_stories_log",
        description="4K/2.7K landscape drone (Apple Log) → crop 9:16 → 1080×1920",
        drone_crop=True,
        crop_position='center',
        crop_target_w=1080, crop_target_h=1920,
        is_log=True, upscale=1.0,
        # Enhancement intentionally neutralised — drone footage often looks over-processed;
        # add sharpen/denoise/etc. explicitly via CLI flags per clip
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=1.0, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=2.8, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
        output_crf=16, output_bitrate="20M",
        force_resolution=(1080, 1920),
    ),

    # ── YouTube (keeps native resolution + aspect, H.264, keeps audio) ──
    # No forced resolution and no crop: the source resolution is preserved, so
    # 4K stays 4K (never downscaled). H.264 is YouTube's recommended upload
    # codec; pass --codec h265 for a smaller master. Quality is CRF-driven with
    # a generous bitrate cap (ample headroom for 4K).

    "youtube": dict(
        _name="youtube",
        description="→ YouTube, native resolution preserved (H.264, AAC). 4K stays 4K",
        is_log=False, upscale=1.0,
        # Enhancement off by default — add explicitly via CLI flags per clip
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=0.9, detail_amount=0.0, detail_inject=0.0,  # off; --sharpen to enable
        local_contrast=False, clahe_clip=2.5, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
        # CRF 18 ≈ visually lossless; 60 Mbps cap leaves headroom up to 4K
        output_crf=18, output_bitrate="60M",
        output_codec="h264",
        # No force_resolution → output matches the source resolution exactly
    ),

    "youtube_log": dict(
        _name="youtube_log",
        description="Apple Log (ProRes .MOV) → Rec.709 → YouTube, native res (H.264)",
        is_log=True, upscale=1.0,
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=1.0, detail_amount=0.0, detail_inject=0.0,  # off; --sharpen to enable
        local_contrast=False, clahe_clip=2.8, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
        output_crf=18, output_bitrate="60M",
        output_codec="h264",
        # No force_resolution → preserves the ProRes source resolution (e.g. 4K)
    ),

    "youtube_4k": dict(
        _name="youtube_4k",
        description="→ YouTube 4K, standardised to 3840×2160 (upscales smaller sources)",
        is_log=False, upscale=1.0,
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=1.0, detail_amount=0.0, detail_inject=0.0,  # off; --sharpen to enable
        local_contrast=False, clahe_clip=2.5, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
        # 45 Mbps ≈ YouTube's 2160p30 upload spec
        output_crf=18, output_bitrate="45M",
        output_codec="h264",
        force_resolution=(3840, 2160),   # 4K floor: cover-fit smaller sources up to 4K
    ),

    # ── Still image → IG Stories ───────────────────────────────────
    # Auto-selected for image inputs (.jpg/.png/.heic/.tif/.webp …).
    # Crop a 9:16 vertical window then resize to 1080×1920 — no enhancement.
    "image": dict(
        _name="image",
        description="Still image → crop 9:16 → 1080×1920 (crop+resize only, no enhancement)",
        drone_crop=True,
        crop_position='center',   # horizontal for landscape, vertical for tall/portrait
        crop_target_w=1080, crop_target_h=1920,
        is_log=False, upscale=1.0,
        # Enhancement off by default — opt in per image via CLI flags
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=0.9, detail_amount=0.0, detail_inject=0.0,  # off; --sharpen to enable
        local_contrast=False, clahe_clip=2.5, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
        out_quality=97,           # JPEG quality for image output (--quality)
        srgb=False,               # colour-manage output to sRGB (--srgb)
        output_crf=16, output_bitrate="20M",
        force_resolution=(1080, 1920),
    ),
}


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='media_processor.py',
        description='🎬 Topaz-style Video Enhancement — iPhone 15 Pro Max → Instagram Stories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRESETS
  instagram_stories       Standard footage → IG Stories (sharp, vivid, H.265)
  instagram_stories_log   Apple Log → Rec.709 + IG Stories  ← use for Log footage
  upscale_2x              2× super-resolution with detail injection
  denoise_only            Low-light denoising, no upscale
  neutral_log             Log → Rec.709 only, preserve for grading
  image                   Still image → crop 9:16 → 1080×1920 (auto for image inputs)

EXAMPLES
  # Still image (JPEG/PNG/HEIC/TIFF/WebP) → crop center → 1080x1920
  python3 media_processor.py photo.jpg ig_story.jpg

  # Crispest IG Stories still: light sharpen + max-quality JPEG + sRGB
  python3 media_processor.py photo.jpg ig_story.jpg --sharpen 0.6 --quality 100 --srgb

  # Punchier colours: saturation + vibrance
  python3 media_processor.py photo.jpg ig_story.jpg --saturation 1.2 --vibrance 0.3

  # Boost overall saturation but tame warm tones (per-colour, Lightroom-style)
  python3 media_processor.py photo.jpg ig_story.jpg --saturation 1.3 --sat-yellow 0.5 --sat-orange 0.7 --sat-red 0.8

  # Neutralise an overall warm/yellow cast (cooler white balance)
  python3 media_processor.py photo.jpg ig_story.jpg --temperature -0.3

  # PNG (lossless source for IG's re-encoder)
  python3 media_processor.py photo.jpg ig_story.png --sharpen 0.6

  # Tall/portrait image (pano): crop-position selects the VERTICAL window
  python3 media_processor.py pano.jpg ig_story.jpg --crop-position top
  python3 media_processor.py pano.jpg ig_story.jpg --crop-position bottom

  # Standard iPhone video → IG Stories
  python3 media_processor.py clip.mp4 ig_story.mp4

  # Apple Log footage → IG Stories
  python3 media_processor.py log_clip.mp4 ig_story.mp4 --preset instagram_stories_log

  # Drone 4K landscape → crop center → 1080x1920 IG Stories
  python3 media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories

  # Drone crop: static offset (subject at x=1500 throughout)
  python3 media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 1500

  # Drone crop: animated pan across the whole clip
  python3 media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 500 --crop-position-end 2200

  # Animated pan over a specific time window (easing relative to 10s–45s only)
  python3 media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 500 --crop-position-end 2200 --start-time 10 --end-time 45

  # Time window with M:SS format
  python3 media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 500 --crop-position-end 2200 --start-time 0:10 --end-time 0:45

  # Drone Apple Log + animated pan + time window
  python3 media_processor.py drone_log.mp4 ig_story.mp4 --preset drone_stories_log --crop-position 500 --crop-position-end 2200 --start-time 10 --end-time 45

  # Custom: Log + stronger sharpen
  python3 media_processor.py clip.mp4 out.mp4 --log --sharpen 0.9 --denoise 0.3

  # Upscale 1080p → 4K
  python3 media_processor.py 1080p.mp4 4k_out.mp4 --preset upscale_2x

  # YouTube, native resolution preserved (4K stays 4K), H.264, keeps audio
  python3 media_processor.py clip.mov yt.mp4 --preset youtube
  # iPhone ProRes Log 4K .MOV → Rec.709 → YouTube at native 4K
  python3 media_processor.py IMG_7971.MOV yt.mp4 --preset youtube_log
  # Smaller master: same but H.265
  python3 media_processor.py IMG_7971.MOV yt.mp4 --preset youtube_log --codec h265

  # MONTAGE: stitch 1–6s clips from a JSON manifest into one silent video
  python3 media_processor.py --print-sample-manifest > clips.json   # then edit
  python3 media_processor.py --sequence clips.json montage.mp4

  # GRID TOUR: mosaic of photos/videos → zoom into each tile (full-res) → out
  python3 media_processor.py --print-sample-grid > grid.json        # then edit
  python3 media_processor.py --grid grid.json tour.mp4              # IG Story
  python3 media_processor.py --grid grid.json tour.mp4 --format post # IG Post 4:5
        """
    )

    parser.add_argument('input',  nargs='?', help='Input video (4K 30fps)')
    parser.add_argument('output', nargs='?', help='Output video path')
    parser.add_argument('--preset', choices=PRESETS.keys(),
                        default='instagram_stories', metavar='PRESET')
    parser.add_argument('--log',        action='store_true',
                        help='Input is Apple Log → applies Rec.709 LUT')
    parser.add_argument('--codec', choices=['h264', 'h265'], default=None,
                        help='Output video codec: h264 (most compatible — YouTube) '
                             'or h265 (efficient — IG/Apple). Overrides the preset.')
    parser.add_argument('--cpu-log', action='store_true',
                        help='Force the Apple Log→Rec.709 conversion onto the CPU '
                             '(default uses the GPU via CuPy when available).')
    parser.add_argument('--crop-position', default=None,
                        metavar='center|left|right|INT',
                        help='Crop start position for drone mode (default: center). '
                             'Integer = pixel x offset from left edge of source frame.')
    parser.add_argument('--crop-position-end', default=None,
                        metavar='center|left|right|INT',
                        help='Crop end position for animated pan (drone mode). '
                             'When set, crop smoothly interpolates from --crop-position '
                             'to this value across the clip duration. '
                             'Example: --crop-position 500 --crop-position-end 2000')
    parser.add_argument('--easing', default='ease_in_out',
                        choices=['linear', 'ease_in_out'],
                        help='Pan animation curve (default: ease_in_out — slow start/end, '
                             'fast middle; linear = constant speed pan)')
    parser.add_argument('--start-time', default=None, metavar='SECONDS or M:SS',
                        help='Start time of the segment to process. '
                             'Easing and crop pan are calculated relative to this window. '
                             'Examples: 10  |  10.5  |  1:30')
    parser.add_argument('--end-time', default=None, metavar='SECONDS or M:SS',
                        help='End time of the segment to process. '
                             'Examples: 45  |  45.0  |  0:45')
    parser.add_argument('--upscale',    type=float, metavar='FACTOR',
                        help='Upscale factor (1.0=none, 2.0=double)')
    parser.add_argument('--denoise',    type=float, metavar='0-1',
                        help='Denoise strength')
    parser.add_argument('--sharpen',    type=float, metavar='0-1',
                        help='Sharpen amount')
    parser.add_argument('--temperature', '--temp', type=float, metavar='-1..1', dest='temperature',
                        help='White-balance temperature for images: negative = cooler/bluer '
                             '(neutralises a warm/yellow cast), positive = warmer. 0 = neutral.')
    parser.add_argument('--saturation', type=float, metavar='MULT',
                        help='Saturation multiplier (1.0=unchanged). Works for video and images.')
    parser.add_argument('--vibrance', type=float, metavar='0-1',
                        help='Vibrance — adaptive saturation that boosts dull colours more '
                             'and protects already-saturated areas (0=off). Video and images.')
    hue_group = parser.add_argument_group(
        'per-colour saturation (images only)',
        'Multiply the saturation of specific hue bands, like Lightroom HSL. '
        '1.0 = unchanged, <1 = less saturated, >1 = more. '
        'e.g. tame warm tones: --sat-yellow 0.5 --sat-orange 0.7 --sat-red 0.8')
    for _band in HUE_BANDS:
        hue_group.add_argument(f'--sat-{_band}', type=float, metavar='MULT',
                               help=f'Saturation multiplier for {_band} tones (1.0=unchanged)')
    parser.add_argument('--quality', type=int, metavar='1-100',
                        help='Image output JPEG quality (default 97). Images only; '
                             'ignored for PNG output and for video.')
    parser.add_argument('--srgb', action='store_true',
                        help='Colour-manage image output to the sRGB profile using the '
                             'source ICC profile (needs Pillow). Images only.')
    parser.add_argument('--sequence', metavar='MANIFEST.json',
                        help='Montage mode: stitch the 1–6s clips defined in a JSON '
                             'manifest into one silent video. Output path is the '
                             'positional arg or the manifest\'s "output" field.')
    parser.add_argument('--print-sample-manifest', action='store_true',
                        help='Print a sample --sequence JSON manifest and exit')
    parser.add_argument('--grid', metavar='MANIFEST.json',
                        help='Grid-tour mode: open on a mosaic then zoom into each '
                             'chosen tile (full-res) and back out. Output path is the '
                             'positional arg or the manifest\'s "output" field.')
    parser.add_argument('--format', choices=['story', 'post'], default=None,
                        help='Grid-tour output format: story (1080×1920) or '
                             'post (1080×1350, 4:5). Overrides the manifest.')
    parser.add_argument('--print-sample-grid', action='store_true',
                        help='Print a sample --grid JSON manifest and exit')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all presets and exit')

    args = parser.parse_args()

    if args.cpu_log:
        set_force_cpu_log(True)

    if args.print_sample_manifest:
        print(sample_manifest_text())
        return

    if args.print_sample_grid:
        print(sample_grid_text())
        return

    if args.list_presets:
        print("\n🎬  Media Processor — Presets\n")
        for name, cfg in PRESETS.items():
            tags = []
            if cfg['is_log']:          tags.append('Log')
            if cfg['upscale'] != 1.0:  tags.append(f"{cfg['upscale']}×")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            print(f"  {name:<28} {cfg['description']}{tag_str}")
        print()
        return

    # Montage mode — output is the positional arg (args.input) or manifest 'output'
    if args.sequence:
        process_sequence(args.sequence, args.input)
        return

    # Grid/canvas-tour mode — output is the positional arg or manifest 'output'.
    # Default is the Prezi/SCRL-style canvas (native aspect); "layout":"grid"
    # selects the legacy uniform grid.
    if args.grid:
        try:
            _layout = (_load_json_file(args.grid).get('layout') or 'canvas').lower()
        except Exception:
            _layout = 'canvas'
        if _layout == 'grid':
            process_grid_tour(args.grid, args.input, args.format)
        else:
            process_canvas_tour(args.grid, args.input, args.format)
        return

    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"\n❌  Input not found: {args.input}\n")
        sys.exit(1)

    # Auto-detect still images by extension → crop/resize-only image pipeline.
    # The 'image' preset has all enhancement off and drone_crop=True.
    is_image = Path(args.input).suffix.lower() in IMAGE_EXTS
    if is_image:
        cfg = PRESETS['image'].copy()
    else:
        cfg = PRESETS[args.preset].copy()

    # CLI overrides
    if args.log:                    cfg['is_log'] = True
    if args.codec is not None:      cfg['output_codec'] = args.codec
    if args.upscale is not None:    cfg['upscale'] = args.upscale
    if args.denoise is not None:    cfg['denoise_strength'] = args.denoise
    if args.sharpen is not None:    cfg['sharpen_amount'] = args.sharpen
    if args.temperature is not None: cfg['temperature'] = args.temperature
    if args.saturation is not None: cfg['saturation'] = args.saturation
    if args.vibrance is not None:   cfg['vibrance'] = args.vibrance
    # Per-colour saturation (--sat-red, --sat-yellow, …) → cfg['sat_bands']
    sat_bands = {b: getattr(args, f'sat_{b}') for b in HUE_BANDS
                 if getattr(args, f'sat_{b}') is not None}
    if sat_bands:                   cfg['sat_bands'] = sat_bands
    if args.quality is not None:    cfg['out_quality'] = max(1, min(args.quality, 100))
    if args.srgb:                   cfg['srgb'] = True
    if args.crop_position is not None:
        try:
            cfg['crop_position'] = int(args.crop_position)
        except ValueError:
            cfg['crop_position'] = args.crop_position
    if args.crop_position_end is not None:
        try:
            cfg['crop_position_end'] = int(args.crop_position_end)
        except ValueError:
            cfg['crop_position_end'] = args.crop_position_end
    if args.easing:
        cfg['crop_easing'] = args.easing
    if args.start_time is not None:
        cfg['start_time'] = args.start_time   # parse_time() handles str/float
    if args.end_time is not None:
        cfg['end_time'] = args.end_time

    # Route to image or video pipeline based on auto-detected input type
    if is_image:
        process_image(args.input, args.output, cfg)
    else:
        process_video(args.input, args.output, cfg)


if __name__ == '__main__':
    main()

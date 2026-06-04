#!/usr/bin/env python3
"""
media_processor.py — Topaz-style Media Enhancement for iPhone 15 Pro Max footage
================================================================================
Optimized for 4K 30fps video and still images → Instagram Stories output.
Still images (JPEG/PNG/HEIC/TIFF/WebP) are auto-detected and routed through a
crop + resize-only pipeline; everything else is treated as video.

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
    """Resolve a position value (string or int) to a pixel x offset."""
    if isinstance(position, int):
        return max(0, min(position, max_x))
    elif position == 'left':
        return 0
    elif position == 'right':
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

    crop_y = (src_h - crop_h) // 2
    max_x  = src_w - crop_w

    start_x = resolve_x(position, max_x)

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
    pan_label    = f"pan {pan_distance}px  ({start_x} → {end_x}  {easing})" if animated else f"static  x={start_x}"

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
        'get_crop_x':   get_crop_x,
        'ff_filter':    ff_filter,   # valid only for static crops
        'pan_label':    pan_label,
    }

# ══════════════════════════════════════════════════════════════════
#  COLOR SCIENCE  —  Apple Log → Rec.709
# ══════════════════════════════════════════════════════════════════

def build_apple_log_lut() -> np.ndarray:
    """
    Pre-compute a 256-entry Apple Log → Rec.709 lookup table.
    Applied per-channel via cv2.LUT() — very fast (~14ms at 4K).

    Apple Log spec: logarithmic encoding with ~17 stops dynamic range.
    Approximates the official Apple + Blackmagic Rec.709 transform.
    """
    x = np.arange(256, dtype=np.float64) / 255.0

    # Apple Log → linear light (inverse of Apple Log encoding)
    c1, c2, c3 = 0.212639, 0.215180, 0.532400
    linear = np.where(
        x > 0.092864,
        np.power(2.0, (x - c3) / c1) - c2,
        (x - 0.092864) / 5.367655
    )
    linear = np.clip(linear, 0.0, 1.0)

    # Linear → Rec.709 gamma (BT.709 OETF)
    rec709 = np.where(
        linear <= 0.018,
        linear * 4.5,
        1.099 * np.power(np.maximum(linear, 1e-10), 0.45) - 0.099
    )

    return np.clip(rec709 * 255, 0, 255).astype(np.uint8)


# Pre-compute at module load (shared across all frames)
_LOG_LUT = build_apple_log_lut()


def apply_log_lut(frame_bgr: np.ndarray) -> np.ndarray:
    """Apple Log → Rec.709 via pre-computed per-channel LUT. ~14ms at 4K."""
    return cv2.LUT(frame_bgr, _LOG_LUT)


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
        sat = self.cfg['saturation']
        vib = self.cfg['vibrance']
        if abs(sat - 1.0) < 0.01 and vib < 0.01:
            return frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # Vibrance boost: more effect on low-saturation pixels
        vib_boost = 1.0 + vib * (1.0 - s / 255.0)
        s = np.clip(s * sat * vib_boost, 0, 255)

        return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)

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

        # Pre-upload LUT to GPU
        self._lut_gpu = cv2.cuda_GpuMat()
        self._lut_gpu.upload(_LOG_LUT.reshape(1, 256, 1))

    # ── Log LUT ──

    def apply_log_lut_gpu(self, gpu_frame: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        """Apply Apple Log → Rec.709 LUT on GPU."""
        # cv2.cuda.LUT operates per channel
        return cv2.cuda.LUT(gpu_frame, self._lut_gpu)

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
        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)

        if self.cfg['is_log']:
            gpu = self.apply_log_lut_gpu(gpu)

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
        if not self.history or self.blend < 0.01:
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

    return {
        'width':     int(vs['width']),
        'height':    int(vs['height']),
        'fps':       fps,
        'frames':    nb,
        'codec':     vs['codec_name'],
        'duration':  float(info['format'].get('duration', 0)),
        'has_audio': any(s['codec_type'] == 'audio' for s in info['streams']),
    }


def make_reader(path: str, vf: str = None,
                start_time: float = None, end_time: float = None) -> subprocess.Popen:
    """
    Open an FFmpeg pipe reader with optional trim and crop filter.

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

    cmd += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        **_POPEN_KWARGS
    )


def make_writer(path: str, w: int, h: int, fps: float,
                audio_src: str, crf: int, bitrate: str) -> subprocess.Popen:
    ffmpeg = find_tool('ffmpeg')
    buf = f"{int(bitrate.rstrip('M')) * 2}M"
    return subprocess.Popen(
        _build_writer_cmd(ffmpeg, w, h, fps, audio_src, crf, bitrate, buf, path),
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        **_POPEN_KWARGS
    )


def _build_writer_cmd(ffmpeg, w, h, fps, audio_src, crf, bitrate, buf, path):
    """Build FFmpeg writer command — NVENC if GPU available, libx265 CPU fallback."""
    base = [ffmpeg, '-v', 'quiet', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0',
            '-i', audio_src,
            '-map', '0:v', '-map', '1:a?']

    if GPU['nvenc']:
        # NVENC: hardware H.265 encode — 5-10× faster than libx265 CPU
        # rc=vbr + cq maps to quality-equivalent of CPU CRF
        enc = ['-c:v', 'hevc_nvenc',
               '-preset', 'p4',           # NVENC quality preset (p1=fast, p7=best)
               '-rc', 'vbr',
               '-cq', str(crf),           # quality target (same scale as CRF)
               '-b:v', bitrate,
               '-maxrate', f'{int(bitrate.rstrip("M"))*2}M',
               '-bufsize', buf]
    else:
        # CPU fallback
        enc = ['-c:v', 'libx265', '-preset', 'medium',
               '-crf', str(crf),
               '-b:v', bitrate, '-maxrate', bitrate, '-bufsize', buf]

    tail = ['-pix_fmt', 'yuv420p',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            '-tag:v', 'hvc1',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            path]

    return base + enc + tail


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


def process_video(input_path: str, output_path: str, cfg: dict):
    info = probe_video(input_path)
    fps, w, h = info['fps'], info['width'], info['height']
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
    seg_frames   = int(round(seg_duration * fps))

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

    print(f"  Input   {w}×{h}  {fps:.2f}fps  {info['codec'].upper()}{tw_label}")
    ig_note = "  ✅ IG Stories spec" if (out_w, out_h) == (1080, 1920) else ""
    print(f"  Output  {out_w}×{out_h}  {fps:.2f}fps  H.265/HEVC  (Apple HVC1){ig_note}")

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
    enc_label = "NVENC (GPU)" if GPU['nvenc'] else "libx265 (CPU)"
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
    reader = make_reader(input_path, vf=vf_filter, start_time=start_t, end_time=end_t)
    writer = make_writer(output_path, out_w, out_h, fps,
                         input_path, cfg['output_crf'], cfg['output_bitrate'])

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


def process_image(input_path: str, output_path: str, cfg: dict):
    """
    Still-image pipeline: crop a 9:16 vertical window then resize to
    exactly 1080×1920. No color, sharpening, denoise, or enhancement is
    applied — the cropped/resized pixels are saved as-is.
    Crop uses the same compute_crop() as video (crop_position respected;
    crop_position_end is irrelevant for a single still).
    """
    img = load_image(input_path)
    src_h, src_w = img.shape[:2]
    in_ext = Path(input_path).suffix.lower()

    target_w = cfg.get('crop_target_w', 1080)
    target_h = cfg.get('crop_target_h', 1920)
    position = cfg.get('crop_position', 'center')

    # Static crop only — total_frames=1 means get_crop_x() returns start_x
    crop_info = compute_crop(src_w, src_h, target_w, target_h, position)
    cx, cy = crop_info['crop_x'], crop_info['crop_y']
    cw, ch = crop_info['crop_w'], crop_info['crop_h']

    cropped = img[cy:cy + ch, cx:cx + cw]
    out = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    out_ext = Path(output_path).suffix.lower()
    quality = "lossless" if crop_info['scale'] <= 1.0 else f"{crop_info['scale']:.2f}× upscale"

    print(f"\n{'═'*62}")
    print(f"  🖼️   Image Processor  —  Crop → 9:16 Vertical")
    print(f"{'═'*62}")
    ig_note = "  ✅ IG Stories spec" if (target_w, target_h) == (1080, 1920) else ""
    print(f"  Input   {src_w}×{src_h}  {in_ext.lstrip('.').upper()}")
    print(f"  Output  {target_w}×{target_h}  {out_ext.lstrip('.').upper()}{ig_note}")
    print(f"  Crop    {cw}×{ch}  x={cx} y={cy}  ({position})  {quality}")
    print(f"  Mode    Crop + Resize (Lanczos4) — no enhancement")
    print(f"  Preset  {cfg.get('_name', 'image')}")
    print(f"  File    {output_path}")
    print(f"{'═'*62}\n")

    # Save: high-quality JPEG (q=97) or PNG depending on output extension
    if out_ext == '.png':
        ok = cv2.imwrite(output_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok = cv2.imwrite(output_path, out, [cv2.IMWRITE_JPEG_QUALITY, 97])

    if ok and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        kb = os.path.getsize(output_path) / 1e3
        print(f"  ✅  Saved {target_w}×{target_h}\n")
        print(f"  📁  {output_path}  ({kb:.0f} KB)\n")
    else:
        print("  ⚠️  Failed to write output image.\n")


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

    # ── Still image → IG Stories ───────────────────────────────────
    # Auto-selected for image inputs (.jpg/.png/.heic/.tif/.webp …).
    # Crop a 9:16 vertical window then resize to 1080×1920 — no enhancement.
    "image": dict(
        _name="image",
        description="Still image → crop 9:16 → 1080×1920 (crop+resize only, no enhancement)",
        drone_crop=True,
        crop_position='center',   # 'center' | 'left' | 'right' | int (pixel x offset)
        crop_target_w=1080, crop_target_h=1920,
        is_log=False, upscale=1.0,
        # All enhancement off — image pipeline saves cropped/resized pixels as-is
        denoise_strength=0.0,                                      # neutralised
        sharpen_amount=0.0, sharpen_radius=0.9, detail_amount=0.0, detail_inject=0.0,  # neutralised
        local_contrast=False, clahe_clip=2.5, clahe_blend=0.0,     # neutralised
        saturation=1.0, vibrance=0.0,                              # neutralised
        temporal_blend=0.0,                                        # neutralised
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
  python3 media_processor.py photo.heic ig_story.jpg --crop-position right

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
        """
    )

    parser.add_argument('input',  nargs='?', help='Input video (4K 30fps)')
    parser.add_argument('output', nargs='?', help='Output video path')
    parser.add_argument('--preset', choices=PRESETS.keys(),
                        default='instagram_stories', metavar='PRESET')
    parser.add_argument('--log',        action='store_true',
                        help='Input is Apple Log → applies Rec.709 LUT')
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
    parser.add_argument('--saturation', type=float, metavar='MULT',
                        help='Saturation multiplier (1.0=unchanged)')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all presets and exit')

    args = parser.parse_args()

    if args.list_presets:
        print("\n🎬  Video Upscaler — Presets\n")
        for name, cfg in PRESETS.items():
            tags = []
            if cfg['is_log']:          tags.append('Log')
            if cfg['upscale'] != 1.0:  tags.append(f"{cfg['upscale']}×")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            print(f"  {name:<28} {cfg['description']}{tag_str}")
        print()
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
    if args.upscale is not None:    cfg['upscale'] = args.upscale
    if args.denoise is not None:    cfg['denoise_strength'] = args.denoise
    if args.sharpen is not None:    cfg['sharpen_amount'] = args.sharpen
    if args.saturation is not None: cfg['saturation'] = args.saturation
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

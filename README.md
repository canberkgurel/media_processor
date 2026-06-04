# media_processor

Topaz-style media enhancement for iPhone 15 Pro Max footage, optimized for **Instagram Stories** output (1080×1920, H.265/HEVC). Handles both **video** and **still images**.

- **Video** → Apple Log → Rec.709 LUT, adaptive denoise, detail sharpening, local contrast (CLAHE), chroma, temporal coherence, H.265 encode (NVENC/NVDEC GPU-accelerated when available, libx265/CPU fallback).
- **Still images** (JPEG/PNG/HEIC/TIFF/WebP) → auto-detected and routed through a **crop + resize-only** pipeline (no enhancement) to exactly 1080×1920.

All enhancement is **neutral by default** — presets ship with sharpening, saturation, denoise, CLAHE, etc. turned off. Add enhancement explicitly per clip via CLI flags (`--sharpen`, `--denoise`, `--saturation`, …).

## Requirements

- Python 3.8+
- [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) on `PATH` (or in a common Windows install location — see `find_tool()`)
- Python packages:

```bash
pip install opencv-python numpy
# Optional: HEIC/HEIF (iPhone default) image support
pip install pillow pillow-heif
# Optional: GPU acceleration (CUDA-enabled build)
pip install opencv-contrib-python
```

## Usage

```bash
python media_processor.py <input> <output> [options]
python media_processor.py --list-presets
```

The pipeline (video vs. image) is auto-selected from the input file extension.

## Examples

### Still images
```bash
# Still image (JPEG/PNG/HEIC/TIFF/WebP) → crop center → 1080×1920
python media_processor.py photo.jpg ig_story.jpg

# HEIC input (requires: pip install pillow-heif), crop to the right side
python media_processor.py photo.heic ig_story.jpg --crop-position right
```

### Video
```bash
# Standard iPhone video → IG Stories
python media_processor.py clip.mp4 ig_story.mp4

# Apple Log footage → Rec.709 + IG Stories
python media_processor.py log_clip.mp4 ig_story.mp4 --preset instagram_stories_log

# Drone 4K landscape → crop center → 1080×1920 IG Stories
python media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories

# Drone crop: static offset (subject at x=1500 throughout)
python media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 1500

# Drone crop: animated pan across the whole clip
python media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 500 --crop-position-end 2200

# Animated pan over a specific time window (easing relative to 10s–45s)
python media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 500 --crop-position-end 2200 --start-time 10 --end-time 45

# Time window with M:SS format
python media_processor.py drone_4k.mp4 ig_story.mp4 --preset drone_stories --crop-position 500 --crop-position-end 2200 --start-time 0:10 --end-time 0:45

# Drone Apple Log + animated pan + time window
python media_processor.py drone_log.mp4 ig_story.mp4 --preset drone_stories_log --crop-position 500 --crop-position-end 2200 --start-time 10 --end-time 45

# Custom enhancement: Log + stronger sharpen + denoise (layered on neutral preset)
python media_processor.py clip.mp4 out.mp4 --log --sharpen 0.9 --denoise 0.3

# Apply explicit sharpen + saturation on top of a neutral drone preset
python media_processor.py drone.mp4 out.mp4 --preset drone_stories --sharpen 0.7 --saturation 1.1

# Upscale 1080p → 4K
python media_processor.py 1080p.mp4 4k_out.mp4 --preset upscale_2x
```

## Presets

| Preset | Description |
|---|---|
| `instagram_stories` | Standard footage → IG Stories (1080×1920, H.265, 30fps) |
| `instagram_stories_log` | Apple Log → Rec.709 + IG Stories (use for Log footage) |
| `upscale_2x` | 2× super-resolution (1080p → 4K, or 4K → 8K) |
| `denoise_only` | Aggressive denoising for low-light footage (keeps `denoise_strength=0.80`) |
| `neutral_log` | Log → Rec.709 only, minimal processing (preserve for grading) |
| `drone_stories` | 4K/2.7K landscape drone → crop 9:16 → 1080×1920 |
| `drone_stories_log` | 4K/2.7K landscape drone (Apple Log) → crop 9:16 → 1080×1920 |
| `image` | Still image → crop 9:16 → 1080×1920 (crop+resize only; auto-selected for image inputs) |

Run `python media_processor.py --list-presets` for the live list.

## Key options

| Flag | Meaning |
|---|---|
| `--preset PRESET` | Preset to start from (default `instagram_stories`) |
| `--log` | Treat input as Apple Log → apply Rec.709 LUT |
| `--crop-position center\|left\|right\|INT` | Crop start position (drone/image mode) |
| `--crop-position-end center\|left\|right\|INT` | Crop end position → animated pan (video) |
| `--easing linear\|ease_in_out` | Pan animation curve |
| `--start-time` / `--end-time` | Process only a time window (`SECONDS` or `M:SS`) |
| `--upscale FACTOR` | Upscale factor (1.0 = none) |
| `--denoise 0-1` | Denoise strength |
| `--sharpen 0-1` | Sharpen amount |
| `--saturation MULT` | Saturation multiplier (1.0 = unchanged) |
| `--list-presets` | List all presets and exit |

## Notes

- **Denoising:** light strengths (`< 0.4`) use a fast bilateral filter; stronger strengths use Non-Local Means (NLM) for higher quality. On a CUDA build, NLM runs on the LAB luminance channel on the GPU.
- **HEIC:** if `pillow-heif` is not installed, HEIC/HEIF inputs fail with a clear `pip install pillow-heif` message.
- **GPU:** NVENC/NVDEC and OpenCV CUDA ops are used automatically when detected; otherwise the tool falls back to CPU.

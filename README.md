# media_processor

Topaz-style media enhancement for iPhone footage. Exports to **Instagram Stories** (1080×1920, H.265) **or YouTube** (native resolution preserved — 4K stays 4K — H.264 with audio). Handles **video** and **still images**.

- **Video** → Apple Log → Rec.709 LUT, adaptive denoise, detail sharpening, local contrast (CLAHE), chroma, temporal coherence, H.265/H.264 encode (NVENC/NVDEC GPU-accelerated when available, libx26x/CPU fallback). Any FFmpeg-readable container works, including iPhone **ProRes `.MOV`** (extension matching is case-insensitive — `.MOV`/`.mov`).
- **Still images** (JPEG/PNG/HEIC/TIFF/WebP) → auto-detected and routed through a **crop + resize-only** pipeline (no enhancement) to exactly 1080×1920.
- **Montage** → stitch 1–6 s clips from a JSON manifest into one silent video (`--sequence`).

All enhancement is **neutral by default** — presets ship with sharpening, saturation, denoise, CLAHE, etc. turned off. Add enhancement explicitly per clip via CLI flags (`--sharpen`, `--denoise`, `--saturation`, …).

## Requirements

- Python 3.8+
- [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) on `PATH` (or in a common Windows install location — see `find_tool()`)
- Python packages:

```bash
pip install opencv-python numpy
# Optional: sRGB colour management (--srgb) and HEIC/HEIF (iPhone default) support
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

# Crispest IG Stories still: light sharpen + max-quality JPEG + sRGB
python media_processor.py photo.jpg ig_story.jpg --sharpen 0.6 --quality 100 --srgb

# PNG output (lossless source for Instagram's re-encoder)
python media_processor.py photo.jpg ig_story.png --sharpen 0.6

# Punchier colours: saturation + vibrance
python media_processor.py photo.jpg ig_story.jpg --sharpen 0.5 --saturation 1.2 --vibrance 0.3

# Boost saturation but tame warm tones (per-colour, Lightroom-style HSL)
python media_processor.py photo.jpg ig_story.jpg --saturation 1.3 --sat-yellow 0.5 --sat-orange 0.7 --sat-red 0.8

# Neutralise an overall warm/yellow cast (cooler white balance)
python media_processor.py photo.jpg ig_story.jpg --temperature -0.3

# Tall/portrait image (e.g. a vertical pano): --crop-position selects the
# VERTICAL window (top | bottom | center | pixel offset) instead of horizontal
python media_processor.py pano.jpg ig_story.jpg --crop-position top
python media_processor.py pano.jpg ig_story.jpg --crop-position bottom

# HEIC input (requires: pip install pillow-heif), landscape crop to the right side
python media_processor.py photo.heic ig_story.jpg --crop-position right
```

> **Getting "crisp" IG Stories:** Instagram always re-encodes uploads to its own
> JPEG at 1080×1920, so the *format* you upload barely matters — what matters is
> feeding it a clean 1080-wide source so its (soft) downscaler does as little as
> possible. Export at exactly 1080×1920 (this tool does), add a light `--sharpen`
> to counter softening from large reductions, use `--srgb` so colours don't shift,
> and either PNG or a high `--quality` JPEG. See the "Image quality" options below.

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

### YouTube (native resolution preserved, H.264, keeps audio)
```bash
# Any landscape clip → YouTube, resolution unchanged (4K stays 4K)
python media_processor.py clip.mov yt.mp4 --preset youtube

# iPhone ProRes Log 4K .MOV → Rec.709 → YouTube at native 4K
python media_processor.py "H:\iPhone Media\IMG_7971.MOV" yt.mp4 --preset youtube_log

# Smaller master with the same quality target: switch to H.265
python media_processor.py IMG_7971.MOV yt.mp4 --preset youtube_log --codec h265

# Force/standardise to 4K (upscales smaller sources up to 3840×2160)
python media_processor.py clip1080.mp4 yt.mp4 --preset youtube_4k
```

> `youtube`/`youtube_log` **never downscale** — they keep the source's exact resolution and aspect. `youtube_4k` standardises to 3840×2160 (cover-fit). All YouTube presets keep audio.

### Montage — stitch multiple clips into one video

Define an ordered list of 1–6 second clips in a JSON manifest and export a single
**silent** 1080×1920 video. Each clip is trimmed to its window, processed with its
chosen preset, normalised to a common fps, and losslessly concatenated.

```bash
# Print a starter manifest, edit it, then build the montage
python media_processor.py --print-sample-manifest > clips.json
python media_processor.py --sequence clips.json montage.mp4
```

Manifest format:

```json
{
  "output": "montage.mp4",
  "fps": 30,
  "defaults": { "preset": "instagram_stories" },
  "clips": [
    { "input": "clip_a.mp4", "start": 0,  "duration": 3 },
    { "input": "clip_b.mp4", "start": 12, "duration": 5, "preset": "drone_stories", "crop_position": "left" },
    { "input": "clip_c.mov", "start": 4,  "duration": 2, "log": true, "sharpen": 0.5, "saturation": 1.1 }
  ]
}
```

- **`clips`** (required): ordered list. Each needs `input` and `duration`; `start` defaults to 0.
- **`duration`** is clamped to **1–6 s** (a warning prints if it was out of range).
- **`fps`** (default 30) and **`defaults`** (per-clip settings applied to every clip unless the clip overrides them) are optional. Per-clip keys mirror the CLI: `preset`, `log`, `crop_position`, `crop_position_end`, `easing`, `sharpen`, `denoise`, `saturation`, `vibrance`, `upscale`.
- **Output** comes from the CLI positional or the manifest's `output`.
- **Video only** — image inputs are rejected (use the still-image pipeline separately). Output is **silent** (add music in your editor / Instagram).

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
| `youtube` | → YouTube, **native resolution preserved** (H.264, AAC). 4K stays 4K |
| `youtube_log` | Apple Log (ProRes `.MOV`) → Rec.709 → YouTube, native res (H.264) |
| `youtube_4k` | → YouTube 4K, standardised to 3840×2160 (upscales smaller sources) |
| `image` | Still image → crop 9:16 → 1080×1920 (crop+resize only; auto-selected for image inputs) |

Run `python media_processor.py --list-presets` for the live list.

## Key options

| Flag | Meaning |
|---|---|
| `--preset PRESET` | Preset to start from (default `instagram_stories`) |
| `--log` | Treat input as Apple Log → apply Rec.709 LUT |
| `--codec h264\|h265` | Output video codec — overrides the preset (h264 = YouTube-friendly, h265 = efficient/IG) |
| `--crop-position center\|left\|right\|top\|bottom\|INT` | Crop position. Landscape sources pan **horizontally**; tall/portrait sources select the **vertical** window |
| `--crop-position-end center\|left\|right\|INT` | Crop end position → animated pan (video) |
| `--easing linear\|ease_in_out` | Pan animation curve |
| `--start-time` / `--end-time` | Process only a time window (`SECONDS` or `M:SS`) |
| `--upscale FACTOR` | Upscale factor (1.0 = none) |
| `--denoise 0-1` | Denoise strength |
| `--sharpen 0-1` | Sharpen amount (video: multi-layer USM; image: light USM after resize) |
| `--saturation MULT` | Saturation multiplier (1.0 = unchanged). Works for video **and** images |
| `--vibrance 0-1` | Adaptive saturation — boosts dull colours more, protects saturated areas. Video **and** images |
| `--list-presets` | List all presets and exit |

### Image quality (still images only)

| Flag | Meaning |
|---|---|
| `--sharpen 0-1` | Light unsharp mask applied **after** the downscale (default off). ~0.4–0.8 restores crispness lost in large reductions |
| `--temperature -1..1` | White balance: negative = cooler/bluer (neutralises a warm/yellow **cast**), positive = warmer. 0 = neutral. Use this — not saturation — to fix an overall tint |
| `--saturation MULT` | HSV saturation multiplier (1.0 = unchanged) |
| `--vibrance 0-1` | Adaptive saturation boost (dull colours affected more; default off) |
| `--sat-COLOR MULT` | Per-colour (hue-selective) saturation, like Lightroom HSL. `COLOR` ∈ `red, orange, yellow, green, aqua, blue, purple, magenta`. 1.0 = unchanged, `<1` less, `>1` more. e.g. `--sat-yellow 0.5 --sat-orange 0.7 --sat-red 0.8` to tame warm tones after a global boost |
| `--quality 1-100` | Output JPEG quality (default 97; 4:4:4 chroma). Ignored for PNG |
| `--srgb` | Colour-manage the output to sRGB using the source's ICC profile, embedding an sRGB profile. Needs Pillow |

## Notes

- **YouTube output:** `youtube`/`youtube_log` preserve the source's native resolution and aspect (no crop, no downscale), so 4K ProRes exports at 4K. H.264 is the default (YouTube's recommended codec); `--codec h265` makes a smaller master. Quality is CRF-driven with a generous bitrate cap.
- **Audio + trimming:** when you trim with `--start-time`/`--end-time`, the muxed audio is now trimmed to the same window (previously the full-length audio was kept, leaving video and audio mismatched).
- **Fitting to 1080×1920:** presets that force the IG-Stories resolution now *actually* resize non-matching sources with a **cover fit** (scale to fill 9:16, centre-crop the overflow) rather than only relabelling the size. A 16:9 clip therefore fills the vertical frame with the sides cropped. For explicit framing control on landscape footage, use `--preset drone_stories` with `--crop-position`.
- **Montage:** `--sequence manifest.json` builds one silent 1080×1920 video from ordered 1–6 s clips (durations auto-clamped to 1–6 s). All clips are normalised to one fps and concatenated losslessly; `--print-sample-manifest` prints a starter file.
- **Crop orientation is automatic:** if the source is wider than 9:16 the crop window is narrower than the frame and `--crop-position` pans horizontally (`left`/`right`/`center`/pixel-x). If the source is taller than 9:16 (e.g. a vertical pano) the window spans the full width and `--crop-position` selects the vertical slice (`top`/`bottom`/`center`/pixel-y).
- **Denoising:** light strengths (`< 0.4`) use a fast bilateral filter; stronger strengths use Non-Local Means (NLM) for higher quality. On a CUDA build, NLM runs on the LAB luminance channel on the GPU.
- **HEIC:** if `pillow-heif` is not installed, HEIC/HEIF inputs fail with a clear `pip install pillow-heif` message.
- **sRGB:** `--srgb` needs Pillow (`pip install pillow`); without it the flag is skipped with a clear message and the image is still written.
- **GPU:** NVENC/NVDEC and OpenCV CUDA ops are used automatically when detected; otherwise the tool falls back to CPU.

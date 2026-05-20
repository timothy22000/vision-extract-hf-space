"""Vision Extract Pipeline — HuggingFace Gradio Space.

Extract structured tabular data from images and videos using Claude's vision AI.
"""

import base64
import hashlib
import json
import os
import random
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import cv2
import gradio as gr
import imagehash
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
}

MODEL_CHOICES = ["claude-sonnet-4-6", "claude-haiku-3-5", "claude-opus-4-6"]

DEFAULT_MODEL = "claude-sonnet-4-6"

DEFAULT_PROMPT = (
    "Extract all data from this spreadsheet image as CSV. "
    "Include column headers if visible. Output only the CSV : "
    "no explanation, no markdown fences."
)

MAX_TOKENS = 6000
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0

# Rate limiting — configurable via env vars
RATE_LIMIT_IMAGES_PER_HOUR = int(os.environ.get("RATE_LIMIT_IMAGES", "10"))
RATE_LIMIT_VIDEOS_PER_HOUR = int(os.environ.get("RATE_LIMIT_VIDEOS", "3"))

# Owner API key is set as HF secret; users can optionally bring their own
OWNER_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Rate limiter + budget tracker
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe per-session rate limiter for shared Max plan key."""

    def __init__(self):
        self._lock = threading.Lock()
        self._image_requests: Dict[str, List[float]] = defaultdict(list)
        self._video_requests: Dict[str, List[float]] = defaultdict(list)

    def _prune(self, timestamps: List[float], window: float = 3600.0) -> List[float]:
        cutoff = time.time() - window
        return [t for t in timestamps if t > cutoff]

    def check_image(self, session_id: str):
        with self._lock:
            self._image_requests[session_id] = self._prune(
                self._image_requests[session_id]
            )
            if len(self._image_requests[session_id]) >= RATE_LIMIT_IMAGES_PER_HOUR:
                raise gr.Error(
                    f"Rate limit: max {RATE_LIMIT_IMAGES_PER_HOUR} image extractions "
                    "per hour. Please wait or use your own API key."
                )
            self._image_requests[session_id].append(time.time())

    def check_video(self, session_id: str):
        with self._lock:
            self._video_requests[session_id] = self._prune(
                self._video_requests[session_id]
            )
            if len(self._video_requests[session_id]) >= RATE_LIMIT_VIDEOS_PER_HOUR:
                raise gr.Error(
                    f"Rate limit: max {RATE_LIMIT_VIDEOS_PER_HOUR} video extractions "
                    "per hour. Please wait or use your own API key."
                )
            self._video_requests[session_id].append(time.time())


rate_limiter = RateLimiter()


def get_session_id(request: gr.Request) -> str:
    """Derive a stable session identifier from the request."""
    # Use IP + User-Agent hash as a fingerprint
    ip = ""
    ua = ""
    if request:
        ip = request.client.host or ""
        ua = dict(request.headers).get("user-agent", "")
    raw = f"{ip}:{ua}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_api_key(user_key: str) -> Tuple[str, bool]:
    """Return (api_key, is_owner_key). User key takes priority."""
    user_key = (user_key or "").strip()
    if user_key and user_key.startswith("sk-ant-"):
        return user_key, False
    if OWNER_API_KEY:
        return OWNER_API_KEY, True
    raise gr.Error(
        "No API key available. Set ANTHROPIC_API_KEY as a Space secret "
        "or enter your own key below."
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def encode_image_to_base64(image_path: Path) -> Tuple[str, str]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    suffix = str(image_path).lower()
    media_type = "image/jpeg" if suffix.endswith((".jpg", ".jpeg")) else "image/png"
    return b64, media_type


def clean_csv_response(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            block = parts[1]
            first_nl = block.find("\n")
            if first_nl != -1:
                block = block[first_nl:].strip()
            return block
    return text


def compute_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    return (input_tokens / 1_000_000) * pricing["input"] + (
        output_tokens / 1_000_000
    ) * pricing["output"]


def call_claude_vision(
    image_path: Path, model: str, prompt: str, api_key: str = ""
) -> Dict[str, Any]:
    key = api_key or OWNER_API_KEY
    if not key:
        raise EnvironmentError("No API key available.")
    client = anthropic.Anthropic(api_key=key)
    b64, media_type = encode_image_to_base64(image_path)
    start = time.time()
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return {
        "extracted_data": message.content[0].text,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "api_time": time.time() - start,
        "model": message.model,
    }


def call_with_retry(
    image_path: Path, model: str, prompt: str, api_key: str = "",
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            return call_claude_vision(image_path, model, prompt, api_key)
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
        ):
            if attempt == max_retries - 1:
                raise
            delay = (2**attempt) * BASE_RETRY_DELAY + random.uniform(0, 1)
            time.sleep(delay)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries - 1:
                delay = (2**attempt) * BASE_RETRY_DELAY + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Retry loop exhausted")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def extract_video_frames(
    video_path: str, output_dir: str, frame_interval: int = 30
) -> List[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = saved = 0
    paths: List[Path] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            p = out / f"frame_{saved:04d}.png"
            cv2.imwrite(str(p), frame)
            paths.append(p)
            saved += 1
        frame_count += 1
    cap.release()
    return sorted(paths)


def dedupe_frames(
    frames_dir: str,
    output_dir: str,
    hash_size: int = 8,
    threshold: int = 0,
) -> List[Path]:
    src = Path(frames_dir)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    frames = sorted(src.glob("frame_*.png"))
    hashes: Dict[Any, Path] = {}
    kept: List[Path] = []
    for frame in frames:
        h = imagehash.phash(Image.open(frame), hash_size=hash_size)
        if not any(h - existing <= threshold for existing in hashes):
            hashes[h] = frame
            dest_path = dest / frame.name
            shutil.copy2(frame, dest_path)
            kept.append(dest_path)
    return sorted(kept)


# ---------------------------------------------------------------------------
# Stats HTML
# ---------------------------------------------------------------------------


def format_stats_html(
    input_tokens: int,
    output_tokens: int,
    cost: float,
    api_time: Optional[float] = None,
    extra_stats: Optional[Dict[str, str]] = None,
) -> str:
    items = []
    if extra_stats:
        for label, value in extra_stats.items():
            items.append(
                f'<div style="text-align:center;">'
                f'<div style="font-size:22px;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{value}</div>'
                f'<div style="font-size:11px;font-weight:600;color:#86868b;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">{label}</div>'
                f"</div>"
            )
    items.append(
        f'<div style="text-align:center;">'
        f'<div style="font-size:22px;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{input_tokens:,}</div>'
        f'<div style="font-size:11px;font-weight:600;color:#86868b;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">Input Tokens</div>'
        f"</div>"
    )
    items.append(
        f'<div style="text-align:center;">'
        f'<div style="font-size:22px;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{output_tokens:,}</div>'
        f'<div style="font-size:11px;font-weight:600;color:#86868b;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">Output Tokens</div>'
        f"</div>"
    )
    items.append(
        f'<div style="text-align:center;">'
        f'<div style="font-size:22px;font-weight:600;color:#0071e3;font-variant-numeric:tabular-nums;">${cost:.4f}</div>'
        f'<div style="font-size:11px;font-weight:600;color:#86868b;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">Est. Cost</div>'
        f"</div>"
    )
    if api_time is not None:
        items.append(
            f'<div style="text-align:center;">'
            f'<div style="font-size:22px;font-weight:600;color:#1d1d1f;font-variant-numeric:tabular-nums;">{api_time:.1f}s</div>'
            f'<div style="font-size:11px;font-weight:600;color:#86868b;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">Time</div>'
            f"</div>"
        )
    inner = "\n".join(items)
    return (
        f'<div style="display:flex;justify-content:center;gap:36px;padding:20px 0;'
        f'flex-wrap:wrap;">\n{inner}\n</div>'
    )


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------


def process_image(
    image_path: Optional[str],
    model: str,
    prompt: str,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    output_format: str,
    user_api_key: str = "",
    request: gr.Request = None,
):
    if image_path is None:
        raise gr.Error("Please upload an image.")

    api_key, is_owner = get_api_key(user_api_key)

    # Enforce rate limits when using the shared key
    if is_owner:
        session_id = get_session_id(request)
        rate_limiter.check_image(session_id)

    img_path = Path(image_path)
    tmp_dir = None

    try:
        # Crop if specified
        has_crop = any(v > 0 for v in [crop_left, crop_top, crop_right, crop_bottom])
        if has_crop:
            tmp_dir = tempfile.mkdtemp()
            img = Image.open(img_path)
            cropped = img.crop(
                (int(crop_left), int(crop_top), int(crop_right), int(crop_bottom))
            )
            crop_path = Path(tmp_dir) / "cropped.png"
            cropped.save(crop_path)
            img_path = crop_path

        result = call_with_retry(img_path, model, prompt, api_key)
        raw_text = result["extracted_data"]
        clean = clean_csv_response(raw_text)

        input_t = result["input_tokens"]
        output_t = result["output_tokens"]
        cost = compute_cost(input_t, output_t, model)
        stats_html = format_stats_html(input_t, output_t, cost, result["api_time"])

        # Parse to DataFrame
        try:
            df = pd.read_csv(StringIO(clean))
            df = df.dropna(how="all")
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        except Exception:
            df = pd.DataFrame({"raw_output": [clean]})

        if output_format == "Table":
            return df, "", stats_html
        elif output_format == "CSV":
            return None, df.to_csv(index=False), stats_html
        elif output_format == "JSON":
            return None, df.to_json(orient="records", indent=2), stats_html
        else:  # Markdown
            return None, df.to_markdown(index=False), stats_html

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def process_video(
    video_path: Optional[str],
    model: str,
    prompt: str,
    frame_interval: int,
    max_frames: int,
    do_dedup: bool,
    hash_size: int,
    threshold: int,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    output_format: str,
    user_api_key: str = "",
    request: gr.Request = None,
    progress=gr.Progress(track_tqdm=False),
):
    if video_path is None:
        raise gr.Error("Please upload a video.")

    api_key, is_owner = get_api_key(user_api_key)

    if is_owner:
        session_id = get_session_id(request)
        rate_limiter.check_video(session_id)
        max_frames = min(int(max_frames), 10)  # Cap for shared key

    tmp_root = tempfile.mkdtemp()
    frames_dir = os.path.join(tmp_root, "frames")
    dedup_dir = os.path.join(tmp_root, "deduped")
    crop_dir = os.path.join(tmp_root, "cropped")

    try:
        # 1. Extract frames
        progress(0.0, desc="Extracting frames...")
        all_frames = extract_video_frames(video_path, frames_dir, int(frame_interval))
        total_extracted = len(all_frames)
        if total_extracted == 0:
            raise gr.Error("No frames could be extracted from this video.")

        # 2. Dedup
        if do_dedup:
            progress(0.1, desc="Deduplicating frames...")
            working_frames = dedupe_frames(
                frames_dir, dedup_dir, int(hash_size), int(threshold)
            )
        else:
            working_frames = all_frames

        after_dedup = len(working_frames)

        # 3. Crop
        has_crop = any(v > 0 for v in [crop_left, crop_top, crop_right, crop_bottom])
        if has_crop:
            progress(0.15, desc="Cropping frames...")
            Path(crop_dir).mkdir(parents=True, exist_ok=True)
            cropped_frames = []
            box = (int(crop_left), int(crop_top), int(crop_right), int(crop_bottom))
            for fp in working_frames:
                img = Image.open(fp)
                out_path = Path(crop_dir) / fp.name
                img.crop(box).save(out_path)
                cropped_frames.append(out_path)
            working_frames = cropped_frames

        # 4. Cap at max_frames
        working_frames = working_frames[: int(max_frames)]
        to_process = len(working_frames)

        # 5. Process each frame
        all_dfs = []
        total_input = 0
        total_output = 0
        total_time = 0.0
        gallery_images = []

        for i, fp in enumerate(working_frames):
            frac = 0.2 + 0.75 * (i / max(to_process, 1))
            progress(frac, desc=f"Processing frame {i + 1}/{to_process}...")
            try:
                result = call_with_retry(fp, model, prompt, api_key)
                clean = clean_csv_response(result["extracted_data"])
                total_input += result["input_tokens"]
                total_output += result["output_tokens"]
                total_time += result["api_time"]

                try:
                    df = pd.read_csv(StringIO(clean))
                    df = df.dropna(how="all")
                    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                    df["frame_index"] = i
                    df["frame_name"] = fp.name
                    all_dfs.append(df)
                except Exception:
                    pass

                gallery_images.append(str(fp))
            except Exception:
                gallery_images.append(str(fp))

        progress(0.95, desc="Combining results...")

        if not all_dfs:
            raise gr.Error("No data could be extracted from any frame.")

        combined = pd.concat(all_dfs, ignore_index=True)
        cost = compute_cost(total_input, total_output, model)

        extra = {
            "Extracted": str(total_extracted),
            "After Dedup": str(after_dedup),
            "Processed": str(to_process),
        }
        stats_html = format_stats_html(
            total_input, total_output, cost, total_time, extra_stats=extra
        )

        progress(1.0, desc="Done")

        if output_format == "Table":
            return combined, "", stats_html, gallery_images
        elif output_format == "CSV":
            return None, combined.to_csv(index=False), stats_html, gallery_images
        elif output_format == "JSON":
            return (
                None,
                combined.to_json(orient="records", indent=2),
                stats_html,
                gallery_images,
            )
        else:
            return (
                None,
                combined.to_markdown(index=False),
                stats_html,
                gallery_images,
            )

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Theme + CSS
# ---------------------------------------------------------------------------

theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)

css = """
:root, .dark, .gradio-container {
    --body-background-fill: #000000 !important;
    --background-fill-primary: #161617 !important;
    --background-fill-secondary: #1c1c1e !important;
    --block-background-fill: #1c1c1e !important;
    --block-border-color: #2c2c2e !important;
    --block-label-text-color: #8e8e93 !important;
    --body-text-color: #f5f5f7 !important;
    --body-text-color-subdued: #8e8e93 !important;
    --input-background-fill: #2c2c2e !important;
    --input-border-color: #3a3a3c !important;
    --border-color-primary: #2c2c2e !important;
    --color-accent: #0a84ff !important;
    --color-accent-soft: rgba(10,132,255,0.15) !important;
    --table-even-background-fill: #1c1c1e !important;
    --table-odd-background-fill: #232324 !important;
    --table-border-color: #2c2c2e !important;
    --checkbox-border-color: #48484a !important;
    --shadow-drop: none !important;
    --shadow-drop-lg: none !important;
    color-scheme: dark !important;
}

/* Base */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background: #000000 !important;
    color: #f5f5f7 !important;
}
.main-surface {
    background: #000000 !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 24px 32px !important;
}
.main-surface > div {
    border: none !important;
    background: transparent !important;
}

/* Hero */
.hero-title { text-align: center; margin: 0 0 2px 0 !important; padding: 16px 0 0 0 !important; }
.hero-title p { font-size: 28px !important; font-weight: 700 !important; letter-spacing: -0.02em !important; color: #f5f5f7 !important; margin: 0 !important; }
.hero-subtitle { text-align: center; margin: 0 0 20px 0 !important; }
.hero-subtitle p { font-size: 14px !important; color: #8e8e93 !important; margin: 0 !important; }

/* Section labels */
.section-label p {
    font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; color: #636366 !important; margin: 0 !important;
}

/* Banner */
.api-warning {
    background: #1c1c1e !important; border: 1px solid #2c2c2e !important;
    border-radius: 8px !important;
}
.api-warning p { font-size: 13px !important; color: #aeaeb2 !important; margin: 0 !important; }

/* Button */
.primary-btn {
    background: #0a84ff !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; font-size: 14px !important; font-weight: 500 !important;
    padding: 10px 24px !important; width: 100% !important; margin-top: 12px !important;
}
.primary-btn:hover { background: #0071e3 !important; }

/* Column divider */
.column-separator { border-left: 1px solid #2c2c2e !important; padding-left: 24px !important; }

/* Tables */
table { font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }
th {
    background: #1c1c1e !important; color: #8e8e93 !important;
    font-size: 11px !important; text-transform: uppercase !important;
    letter-spacing: 0.04em !important; font-weight: 600 !important;
    border-bottom: 1px solid #2c2c2e !important;
}
td { color: #f5f5f7 !important; border-bottom: 1px solid #232324 !important; }

/* Inputs — all uniform dark */
input, textarea, select, .wrap input, .wrap textarea {
    background: #2c2c2e !important; color: #f5f5f7 !important;
    border: 1px solid #3a3a3c !important; border-radius: 8px !important;
}
input:focus, textarea:focus { border-color: #0a84ff !important; outline: none !important; }
input::placeholder, textarea::placeholder { color: #636366 !important; }

/* Labels */
label, .label-wrap, span.svelte-1gfkn6j {
    color: #aeaeb2 !important;
}

/* Dropdowns */
.secondary-wrap, .wrap .secondary-wrap { background: #2c2c2e !important; border: 1px solid #3a3a3c !important; border-radius: 8px !important; }
ul[role="listbox"] { background: #2c2c2e !important; border: 1px solid #3a3a3c !important; }
ul[role="listbox"] li { color: #f5f5f7 !important; }
ul[role="listbox"] li:hover, ul[role="listbox"] li.selected { background: #3a3a3c !important; }

/* Upload areas */
.upload-text { color: #636366 !important; }

/* Accordion */
.label-wrap { color: #aeaeb2 !important; }

/* Slider track */
input[type="range"] { accent-color: #0a84ff !important; }

/* Footer */
.footer-text p { font-size: 12px !important; color: #48484a !important; text-align: center !important; margin: 0 !important; }
.cost-note p { font-size: 12px !important; color: #636366 !important; margin: 0 !important; }

/* Setup */
.setup-section { border: none !important; box-shadow: none !important; background: transparent !important; }

/* Gallery */
.gallery { background: #1c1c1e !important; border-radius: 8px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1c1c1e; }
::-webkit-scrollbar-thumb { background: #3a3a3c; border-radius: 3px; }

/* Examples */
.gallery-item, .gallery-item img { border-radius: 6px !important; }
"""

# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------

SETUP_MD = """
### Try it free

This Space is powered by a shared key with fair-use limits. \
You get **{img_limit} image extractions** and **{vid_limit} video extractions** per hour \
with all models available.

To remove rate limits, enter your own API key above.

### Get your own API key

1. Go to [console.anthropic.com](https://console.anthropic.com/) and sign up or log in
2. Navigate to **Settings > API Keys**
3. Click **Create Key**, give it a name, and copy the key (starts with `sk-ant-`)
4. Paste it into the **Your API Key** field above — it stays in your browser session only

### Running locally

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
pip install -r requirements.txt
python app.py
```

### Duplicating this Space

1. Click **Duplicate this Space** in the top right
2. In your copy, go to **Settings > Repository secrets**
3. Add a secret named `ANTHROPIC_API_KEY` with your key
4. Your copy will run with full access and no shared limits
""".format(img_limit=RATE_LIMIT_IMAGES_PER_HOUR, vid_limit=RATE_LIMIT_VIDEOS_PER_HOUR)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

# Gradio 6 moved theme/css to launch(); Gradio 5 keeps them on Blocks.
_gradio_major = int(gr.__version__.split(".")[0])
_blocks_kwargs = {"title": "Vision Extract Pipeline"}
_launch_kwargs: Dict[str, Any] = {}
if _gradio_major >= 6:
    _launch_kwargs["theme"] = theme
    _launch_kwargs["css"] = css
else:
    _blocks_kwargs["theme"] = theme
    _blocks_kwargs["css"] = css

with gr.Blocks(**_blocks_kwargs) as demo:
    gr.Markdown("Vision Extract Pipeline", elem_classes=["hero-title"])
    gr.Markdown(
        "Extract structured tabular data from images and videos using Claude's vision AI",
        elem_classes=["hero-subtitle"],
    )

    with gr.Group(elem_classes=["main-surface"]):
        # Status banner
        if OWNER_API_KEY:
            gr.Markdown(
                f"**Free to try** — {RATE_LIMIT_IMAGES_PER_HOUR} image / "
                f"{RATE_LIMIT_VIDEOS_PER_HOUR} video extractions per hour. "
                "Bring your own API key to remove limits.",
                elem_classes=["api-warning"],
            )
        else:
            gr.Markdown(
                "**No shared key configured.** Enter your own key below or "
                "see the *Setup* section at the bottom.",
                elem_classes=["api-warning"],
            )

        with gr.Accordion("Your API Key (optional)", open=True):
            user_key_input = gr.Textbox(
                value="",
                label="Anthropic API Key",
                placeholder="sk-ant-... (leave blank to use free tier)",
                type="password",
                lines=1,
            )
            gr.Markdown(
                "Sent directly to Anthropic, never stored. "
                "Removes rate limits and frame caps.",
                elem_classes=["cost-note"],
            )

        with gr.Tabs(elem_classes=["tabs"]):
            # =============================================================
            # IMAGE TAB
            # =============================================================
            with gr.Tab("Image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("INPUT", elem_classes=["section-label"])
                        img_input = gr.Image(
                            type="filepath", label="Upload Image", height=220
                        )
                        img_model = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value=DEFAULT_MODEL,
                            label="Model",
                        )
                        img_prompt = gr.Textbox(
                            value=DEFAULT_PROMPT, label="Extraction Prompt", lines=2
                        )
                        with gr.Accordion("Crop Region", open=False):
                            with gr.Row():
                                img_cl = gr.Number(value=0, label="Left", precision=0)
                                img_ct = gr.Number(value=0, label="Top", precision=0)
                                img_cr = gr.Number(value=0, label="Right", precision=0)
                                img_cb = gr.Number(value=0, label="Bottom", precision=0)
                        img_format = gr.Radio(
                            choices=["Table", "CSV", "JSON", "Markdown"],
                            value="Table",
                            label="Output Format",
                        )
                        img_btn = gr.Button(
                            "Extract Data", elem_classes=["primary-btn"], variant="primary"
                        )

                    with gr.Column(scale=1, elem_classes=["column-separator"]):
                        gr.Markdown("RESULTS", elem_classes=["section-label"])
                        img_stats = gr.HTML(value="")
                        img_df = gr.Dataframe(label="Extracted Table", wrap=True)
                        img_text = gr.Textbox(
                            label="Extracted Data", lines=12, visible=False,
                        )

                gr.Examples(
                    examples=[["sample_table.png"], ["sample_receipt.png"]],
                    inputs=[img_input],
                    label="Try an example",
                )

                def on_image_format_change(fmt):
                    if fmt == "Table":
                        return gr.update(visible=True), gr.update(visible=False)
                    return gr.update(visible=False), gr.update(visible=True)

                img_format.change(
                    on_image_format_change, [img_format], [img_df, img_text]
                )

                def on_image_submit(image, model, prompt, cl, ct, cr, cb, fmt, ukey, request: gr.Request):
                    df, text, stats = process_image(
                        image, model, prompt, cl, ct, cr, cb, fmt,
                        user_api_key=ukey, request=request,
                    )
                    if fmt == "Table":
                        return df, "", stats, gr.update(visible=True), gr.update(visible=False)
                    return None, text, stats, gr.update(visible=False), gr.update(visible=True)

                img_btn.click(
                    on_image_submit,
                    [img_input, img_model, img_prompt, img_cl, img_ct, img_cr, img_cb, img_format, user_key_input],
                    [img_df, img_text, img_stats, img_df, img_text],
                )

            # =============================================================
            # VIDEO TAB
            # =============================================================
            with gr.Tab("Video"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("INPUT", elem_classes=["section-label"])
                        vid_input = gr.Video(label="Upload Video")
                        vid_model = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value=DEFAULT_MODEL,
                            label="Model",
                        )
                        vid_prompt = gr.Textbox(
                            value=DEFAULT_PROMPT, label="Extraction Prompt", lines=2
                        )

                        gr.Markdown("FRAME EXTRACTION", elem_classes=["section-label"])
                        vid_interval = gr.Slider(
                            minimum=1, maximum=120, value=30, step=1,
                            label="Extract every Nth frame",
                        )
                        vid_maxframes = gr.Slider(
                            minimum=1, maximum=50, value=20, step=1,
                            label="Max frames to process",
                            info="Free tier capped at 10",
                        )
                        vid_dedup = gr.Checkbox(
                            value=True, label="Deduplicate frames"
                        )
                        vid_hashsize = gr.Slider(
                            minimum=4, maximum=16, value=8, step=1,
                            label="Hash size",
                        )
                        vid_threshold = gr.Slider(
                            minimum=0, maximum=10, value=0, step=1,
                            label="Similarity threshold",
                        )
                        with gr.Accordion("Crop Region", open=False):
                            with gr.Row():
                                vid_cl = gr.Number(value=0, label="Left", precision=0)
                                vid_ct = gr.Number(value=0, label="Top", precision=0)
                                vid_cr = gr.Number(value=0, label="Right", precision=0)
                                vid_cb = gr.Number(value=0, label="Bottom", precision=0)
                        vid_format = gr.Radio(
                            choices=["Table", "CSV", "JSON", "Markdown"],
                            value="Table",
                            label="Output Format",
                        )
                        gr.Markdown(
                            "Free tier: max 10 frames per video. "
                            "Use your own key for up to 50.",
                            elem_classes=["cost-note"],
                        )
                        vid_btn = gr.Button(
                            "Extract from Video",
                            elem_classes=["primary-btn"],
                            variant="primary",
                        )

                    with gr.Column(scale=1, elem_classes=["column-separator"]):
                        gr.Markdown("RESULTS", elem_classes=["section-label"])
                        vid_stats = gr.HTML(value="")
                        vid_df = gr.Dataframe(label="Extracted Table", wrap=True)
                        vid_text = gr.Textbox(
                            label="Extracted Data", lines=12, visible=False,
                        )
                        vid_gallery = gr.Gallery(
                            label="Processed Frames", columns=4, height=200
                        )

                def on_dedup_toggle(checked):
                    return gr.update(visible=checked), gr.update(visible=checked)

                vid_dedup.change(
                    on_dedup_toggle, [vid_dedup], [vid_hashsize, vid_threshold]
                )

                def on_video_format_change(fmt):
                    if fmt == "Table":
                        return gr.update(visible=True), gr.update(visible=False)
                    return gr.update(visible=False), gr.update(visible=True)

                vid_format.change(
                    on_video_format_change, [vid_format], [vid_df, vid_text]
                )

                def on_video_submit(
                    video, model, prompt, interval, maxf, dedup, hs, th,
                    cl, ct, cr, cb, fmt, ukey, request: gr.Request
                ):
                    df, text, stats, gallery = process_video(
                        video, model, prompt, interval, maxf, dedup, hs, th,
                        cl, ct, cr, cb, fmt,
                        user_api_key=ukey, request=request,
                    )
                    if fmt == "Table":
                        return (
                            df, "", stats, gallery,
                            gr.update(visible=True), gr.update(visible=False),
                        )
                    return (
                        None, text, stats, gallery,
                        gr.update(visible=False), gr.update(visible=True),
                    )

                vid_btn.click(
                    on_video_submit,
                    [
                        vid_input, vid_model, vid_prompt, vid_interval, vid_maxframes,
                        vid_dedup, vid_hashsize, vid_threshold,
                        vid_cl, vid_ct, vid_cr, vid_cb, vid_format, user_key_input,
                    ],
                    [vid_df, vid_text, vid_stats, vid_gallery, vid_df, vid_text],
                )

    # Setup section — outside the main surface
    with gr.Accordion("Setup: How to get an API key", open=False, elem_classes=["setup-section"]):
        gr.Markdown(SETUP_MD)

    gr.Markdown(
        "Built with Claude Vision API",
        elem_classes=["footer-text"],
    )

if __name__ == "__main__":
    demo.launch(**_launch_kwargs)

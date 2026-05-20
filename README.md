---
title: Vision Extract Pipeline
emoji: "\U0001F441\uFE0F"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
license: mit
short_description: Extract tabular data from images/videos via Claude
---

# Vision Extract Pipeline

Extract structured tabular data from images and videos using Claude's vision AI.

## Features

- **Image extraction** — Upload a screenshot of a table, receipt, spreadsheet, or any structured data and get back a clean CSV/JSON/Markdown table
- **Video extraction** — Upload a video, extract frames at configurable intervals, deduplicate with perceptual hashing, and batch-process through Claude
- **Multiple output formats** — Table view, CSV, JSON, or Markdown
- **Crop support** — Isolate a region of interest before extraction
- **Cost tracking** — See token usage and estimated API cost for every extraction

## Supported formats

- **Images:** PNG, JPG, WEBP
- **Videos:** MP4, MOV, AVI, MKV, WEBM

## Models

- `claude-sonnet-4-6` (default) — Best balance of speed and quality
- `claude-haiku-3-5` — Fastest and cheapest
- `claude-opus-4-6` — Highest quality

## Setup

This Space requires an Anthropic API key.

### HuggingFace Spaces

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)
2. Go to your Space's **Settings** tab
3. Under **Repository secrets**, add a secret named `ANTHROPIC_API_KEY` with your key

### Local

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
pip install -r requirements.txt
python app.py
```

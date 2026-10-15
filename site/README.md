# Website and demo video

The website in `docs/` (https://malevrigns.github.io/agent-jev/), the demo video `docs/media/agentjev-demo.mp4` and the README preview `assets/agentjev_demo.webp` are build output. Everything is built from the files here. Every number on the page and in the video is read from recorded runs, none typed by hand. Edit the sources here and rebuild. Don't edit `docs/index.html` by hand.

## What is here

| Path | What it is |
| --- | --- |
| `src/` | Page template `body.html` (`{{name}}` placeholders filled by `build_site.py`), `style.css`, `app1.js`–`app4.js` |
| `data/capture.json` | The recorded runs: the server started the quickstart's way with each release, every request and response, the quickstart's curl call and the Claude Code hook's stdout (one RTX 5060, 2026-10-14) |
| `data/fetch_gates.log`, `data/fetch_v1.log` | Real `fetch_v1_weights.py` runs (only the per-file lines of the v1 run are kept; those are all the page shows) |
| `data/example_requests.json` | The two hand-written requests `capture.py` sends: the README client example and the agent trace |
| `data/site_data.json` | Every number the page and the video show, written by `build_data.py`. Committed because rebuilding it needs the 1.2 GB release |
| `video/stage.html`, `video/stage.js` | The video as a web page: `render(t)` draws the frame at t seconds |
| `video/timeline.json` | Duration and chapters, written by `render.js`, read by `build_site.py` |
| `capture.py` | Records `data/capture.json` again (GPU, see below) |
| `check.js` | `node site/check.js [desktop\|tablet\|mobile]` loads `docs/` in headless Chrome at 1440, 820 and 390 px wide, reports script errors and horizontal overflow, and screenshots each section into `site/shots/` |

The build also reads files elsewhere in the repo: `coding_gates/results` and `coding_gates/runs.jsonl.gz`, `jev_service/serving_benchmark.json`, the Typed Decisions table in `README.md`, and `docs/index.html` as of commit `90b37d7`. That last file is the earlier single-page manual, which becomes `docs/guide.html`.

## Rebuild

Run from the repository root, in this order:

| Step | Command | Writes | Needs |
| --- | --- | --- | --- |
| 0 | `npm ci --prefix site` | `site/node_modules` | Node 18+ |
| 1 | `python site/build_data.py` | `site/data/site_data.json` | the coding-gates release and `coding_gates/data` (below), no GPU |
| 2 | `python site/video/build_stage.py` | `site/video/stage_data.js` | Python 3.10+ |
| 3 | `node site/video/render.js video` | `docs/media/agentjev-demo.mp4`, `docs/media/agentjev-demo-poster.jpg`, `site/video/timeline.json` | Chrome and ffmpeg, about 3 minutes, no GPU |
| 3b | `node site/video/render.js preview` | `assets/agentjev_demo.webp` | ffmpeg |
| 4 | `python site/build_site.py` | `docs/index.html`, `docs/guide.html`, `docs/media/fonts/` | full git history (not a shallow clone) |

With unchanged sources, steps 0, 2 and 4 alone reproduce `docs/` exactly: `docs/index.html` and `docs/guide.html` come out byte for byte as committed. Only the capture in the next section needs a GPU.

**Step 1** is needed only when an input changes. `python site/build_data.py --check` rebuilds in memory and exits 1 if the committed `site_data.json` no longer matches. It reads the tokenizer and only the header of `model.safetensors` (for the parameter count):

```bash
pip install -r requirements.txt huggingface_hub safetensors
python fetch_v1_weights.py --variant coding-gates
python coding_gates/build_dataset.py --model-path models/agentjev-coding-gates
python site/build_data.py            # --model-dir if the release is elsewhere
```

**Step 3** is needed only when the video changes. `build_site.py` takes the duration and chapters from `video/timeline.json` and the size it prints from the MP4. A new render stays 82 s with 8 chapters, but its bytes can differ with the Chrome version and the system's Chinese font.

- `node site/video/render.js stills 3,9,20` writes single frames to `site/video/out/` for checking a layout change.
- `node site/video/render.js video 23`: the argument is the x264 CRF, 23 by default, which is what the committed video used.
- `VIDEO_OUT=<folder>` makes `video` and `preview` write everything there, leaving the committed files alone.

The preview keeps the README example (27.5–38.6 s) and the Claude Code hook (49.5–59.6 s) at 12 fps, 1280 px wide. `render.js preview` runs:

```bash
ffmpeg -i docs/media/agentjev-demo.mp4 -filter_complex "[0:v]trim=27.5:38.6,setpts=PTS-STARTPTS[a];[0:v]trim=49.5:59.6,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[c];[c]fps=12,scale=1280:-1:flags=lanczos[o]" -map "[o]" -c:v libwebp_anim -loop 0 -quality 82 -compression_level 6 -preset drawing assets/agentjev_demo.webp
```

## Tools

- **Node packages** (`package.json`, versions pinned): `puppeteer-core` drives Chrome, and `@fontsource/inter` and `@fontsource/jetbrains-mono` (OFL) supply the fonts. The page self-hosts their latin subsets in `docs/media/fonts`; the video stage loads them from `site/node_modules`.
- **Chrome**: by default puppeteer finds the installed Google Chrome. Set `CHROME_PATH` to use another Chrome or Chromium binary. Chinese text uses the system font: Noto Sans SC if it is installed, otherwise Microsoft YaHei. The committed video was rendered on Windows 11 with Noto Sans SC installed.
- **ffmpeg** with libx264 and libwebp: `ffmpeg` on the PATH, or set `FFMPEG`. The committed video and preview were encoded with the ffmpeg 7.1 build that `imageio-ffmpeg` 0.6.0 ships:

  ```bash
  pip install imageio-ffmpeg==0.6.0
  export FFMPEG="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  ```

  In PowerShell: `$env:FFMPEG = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`.

## Recording the runs again (GPU)

`python site/capture.py` rewrites `data/capture.json`. It starts `jev_service.server` from the repository root with the coding-gates release, then with v1. For each, it sends every example request once and 15 more times for the median timings, runs the quickstart's curl call, and pipes six tool calls through a temporary copy of `agentjev_hook.py`, so your own hook log is left alone. It needs both releases (`python fetch_v1_weights.py` and `python fetch_v1_weights.py --variant coding-gates`), a CUDA GPU and curl. Then rebuild from step 1. The outputs and timings, and the GPU name in the video, then come from your machine. The page's sentence about where the numbers come from, near the end of `src/body.html`, names the RTX 5060 in plain text, so edit it too.

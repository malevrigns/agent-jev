// Render stage.html frame by frame in headless Chrome and encode with ffmpeg. Run build_stage.py first.
//   node site/video/render.js stills 3,9,20  -> site/video/out/still_<t>.png for checking layout
//   node site/video/render.js video [crf]    -> docs/media/agentjev-demo.mp4 + agentjev-demo-poster.jpg, and
//                                               site/video/timeline.json (duration and chapters for build_site.py)
//   node site/video/render.js preview        -> assets/agentjev_demo.webp, the README's looping preview, cut from the MP4
// CHROME_PATH picks the browser (default: the installed Google Chrome), FFMPEG the encoder (default: ffmpeg on PATH).
// VIDEO_OUT=<folder> writes every output there instead, for a trial render that leaves the committed files alone.
const { launch, serve } = require("../browser");
const { spawn, spawnSync } = require("child_process");
const fs = require("fs"), path = require("path");
const SITE = path.join(__dirname, "..");
const REPO = path.join(SITE, "..");
const FFMPEG = process.env.FFMPEG || "ffmpeg";
const TRIAL = process.env.VIDEO_OUT && path.resolve(process.env.VIDEO_OUT);
const MEDIA = TRIAL || path.join(REPO, "docs", "media");
const TIMELINE = path.join(TRIAL || __dirname, "timeline.json");
const PREVIEW = TRIAL ? path.join(TRIAL, "agentjev_demo.webp") : path.join(REPO, "assets", "agentjev_demo.webp");
const STILLS = TRIAL || path.join(__dirname, "out");
const FPS = 30, POSTER_T = 4.2;
// The README preview: the README example (27.5-38.6 s) and the Claude Code hook (49.5-59.6 s), 12 fps, 1280 wide.
const PREVIEW_FILTER = "[0:v]trim=27.5:38.6,setpts=PTS-STARTPTS[a];[0:v]trim=49.5:59.6,setpts=PTS-STARTPTS[b];"
  + "[a][b]concat=n=2:v=1[c];[c]fps=12,scale=1280:-1:flags=lanczos[o]";

function ffmpeg(args, stdin = "ignore") {
  if (spawnSync(FFMPEG, ["-version"]).error) throw new Error(`cannot run ${FFMPEG}; set FFMPEG to an ffmpeg binary`);
  const ff = spawn(FFMPEG, ["-hide_banner", "-loglevel", "error", "-y", ...args], { stdio: [stdin, "inherit", "inherit"] });
  return { ff, done: new Promise(resolve => ff.on("close", resolve)) };
}

async function preview() {
  const src = path.join(MEDIA, "agentjev-demo.mp4");
  fs.mkdirSync(path.dirname(PREVIEW), { recursive: true });
  const code = await ffmpeg(["-i", src, "-filter_complex", PREVIEW_FILTER, "-map", "[o]", "-c:v", "libwebp_anim", "-loop", "0",
    "-quality", "82", "-compression_level", "6", "-preset", "drawing", PREVIEW]).done;
  console.log("ffmpeg exit", code, PREVIEW, code === 0 ? (fs.statSync(PREVIEW).size / 1e6).toFixed(2) + " MB" : "");
  if (code !== 0) process.exit(1);
}

(async () => {
  const [mode = "stills", arg] = process.argv.slice(2);
  if (!["stills", "video", "preview"].includes(mode)) throw new Error(`unknown mode ${mode}: stills, video or preview`);
  if (mode === "preview") return preview();
  if (!fs.existsSync(path.join(__dirname, "stage_data.js"))) throw new Error("run python site/video/build_stage.py first");
  const server = await serve(SITE);
  const browser = await launch(["--hide-scrollbars", "--force-color-profile=srgb", "--disable-lcd-text"]);
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  page.on("console", m => { if (m.type() === "error") errors.push(m.text()) });
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  await page.goto(server.url + "video/stage.html", { waitUntil: "load" });
  await page.evaluate(async () => {  // load every face now, so no frame is drawn with a fallback font
    await Promise.all(["400 20px Inter", "500 20px Inter", "600 20px Inter", "700 20px Inter", "800 20px Inter", "400 20px JBM",
      "600 20px JBM", "900 20px 'Noto Sans SC'", "600 20px 'Noto Sans SC'"].map(f => document.fonts.load(f, "AgentJev 测试")));
    await document.fonts.ready;
  });
  const fonts = await page.evaluate(() => [...document.fonts].filter(f => f.status === "loaded").map(f => f.family + " " + f.weight));
  const { duration, chapters } = await page.evaluate(() => ({ duration: window.DURATION, chapters: window.CHAPTERS }));
  console.log("duration", duration, "chapters", chapters.length, "fonts", fonts.join(", "));
  if (mode === "stills") {
    fs.mkdirSync(STILLS, { recursive: true });
    for (const t of (arg || "3").split(",").map(Number)) {
      await page.evaluate(t => window.render(t), t);
      await page.screenshot({ path: path.join(STILLS, `still_${String(t).replace(".", "_")}.png`) });
    }
  } else {
    // crf 23 is what the committed video was encoded with
    const crf = arg || "23", n = Math.round(duration * FPS), file = path.join(MEDIA, "agentjev-demo.mp4");
    fs.mkdirSync(MEDIA, { recursive: true });
    const { ff, done } = ffmpeg(["-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-",
      "-c:v", "libx264", "-preset", "slow", "-crf", crf, "-pix_fmt", "yuv420p", "-movflags", "+faststart", file], "pipe");
    const t0 = Date.now();
    for (let i = 0; i < n; i++) {
      await page.evaluate(t => window.render(t), i / FPS);
      const buf = await page.screenshot({ type: "png", optimizeForSpeed: true });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
      if (i % 300 === 0) console.log(`frame ${i}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    }
    ff.stdin.end();
    const code = await done;
    await page.evaluate(t => window.render(t), POSTER_T);
    await page.screenshot({ path: path.join(MEDIA, "agentjev-demo-poster.jpg"), type: "jpeg", quality: 88 });
    const timeline = JSON.stringify({ duration, fps: FPS, size: [1920, 1080], chapters }, null, 1);
    const before = fs.existsSync(TIMELINE) ? fs.readFileSync(TIMELINE, "utf8").replace(/\r\n/g, "\n") : null;
    if (before !== timeline) fs.writeFileSync(TIMELINE, timeline);  // unchanged (perhaps CRLF in a Windows checkout): leave it
    console.log("ffmpeg exit", code, "size", (fs.statSync(file).size / 1e6).toFixed(2), "MB", "in", ((Date.now() - t0) / 1000).toFixed(0), "s");
    if (code !== 0) process.exitCode = 1;
  }
  console.log("page errors:", errors.length ? errors : "none");
  await browser.close();
  server.close();
})().catch(e => { console.error(e); process.exit(1) });

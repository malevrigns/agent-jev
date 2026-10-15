// Load the built site (docs/) in headless Chrome at three widths, report console errors and layout overflow, and
// screenshot each section into site/shots/.   node site/check.js [desktop|tablet|mobile]
const { launch, serve } = require("./browser");
const path = require("path"), fs = require("fs");
const OUT = path.join(__dirname, "shots");
const only = process.argv[2];
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const server = await serve(path.join(__dirname, "..", "docs"));
  const url = process.env.SITE_URL || server.url + "index.html";
  const browser = await launch(["--hide-scrollbars", "--force-color-profile=srgb"]);
  for (const [name, w, h, mobile] of [["desktop", 1440, 900, false], ["tablet", 820, 1180, true], ["mobile", 390, 844, true]]) {
    if (only && only !== name) continue;
    const page = await browser.newPage();
    const errors = [];
    page.on("console", m => { if (["error", "warning"].includes(m.type())) errors.push(m.type() + ": " + m.text()) });
    page.on("pageerror", e => errors.push("pageerror: " + e.message));
    page.on("requestfailed", r => {
      const reason = (r.failure() || {}).errorText;
      // Chrome cancels the video's preload="metadata" request once it has the metadata
      if (!(reason === "net::ERR_ABORTED" && r.resourceType() === "media")) errors.push("requestfailed: " + r.url().slice(0, 120) + " " + reason);
    });
    await page.setViewport({ width: w, height: h, deviceScaleFactor: 1, isMobile: mobile, hasTouch: mobile });
    await page.goto(url, { waitUntil: "load" });
    await new Promise(r => setTimeout(r, 2600));
    await page.screenshot({ path: path.join(OUT, `${name}-00-hero.png`) });
    const ids = await page.$$eval("main > section", s => s.map(x => x.id));
    let i = 1;
    for (const id of ids.slice(1)) {
      await page.evaluate(id => { const el = document.getElementById(id); window.scrollTo(0, el.getBoundingClientRect().top + scrollY - 66) }, id);
      await new Promise(r => setTimeout(r, 2300));
      await page.screenshot({ path: path.join(OUT, `${name}-${String(i++).padStart(2, "0")}-${id}.png`) });
      const sh = await page.evaluate(id => document.getElementById(id).getBoundingClientRect().height, id);
      if (sh > h * 1.15) { // tall section: one more shot further down
        await page.evaluate(() => window.scrollBy(0, innerHeight - 120));
        await new Promise(r => setTimeout(r, 1800));
        await page.screenshot({ path: path.join(OUT, `${name}-${String(i++).padStart(2, "0")}-${id}-b.png`) });
        if (sh > h * 2.1) {
          await page.evaluate(() => window.scrollBy(0, innerHeight - 120));
          await new Promise(r => setTimeout(r, 1500));
          await page.screenshot({ path: path.join(OUT, `${name}-${String(i++).padStart(2, "0")}-${id}-c.png`) });
        }
      }
    }
    const overflow = await page.evaluate(() => {
      const W = document.documentElement.clientWidth, bad = [];
      document.querySelectorAll("body *").forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.width && r.right > W + 1 && !el.closest(".hero-bg,.code pre,.table-wrap,.orb,.tip")) bad.push((el.id ? "#" + el.id : el.className || el.tagName) + " right=" + Math.round(r.right));
      });
      return { scrollWidth: document.documentElement.scrollWidth, W, bad: bad.slice(0, 15) };
    });
    console.log(name, "errors:", errors.length ? errors : "none");
    console.log(name, "overflow:", JSON.stringify(overflow));
    await page.close();
  }
  await browser.close();
  server.close();
})().catch(e => { console.error(e); process.exit(1) });

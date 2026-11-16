// Shared by video/render.js and check.js: headless Chrome, and a folder served over http (web fonts do not load
// from file:// pages). CHROME_PATH picks the browser; by default puppeteer finds the installed Google Chrome.
const puppeteer = require("puppeteer-core");
const http = require("http"), fs = require("fs"), path = require("path");

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css",
  ".json": "application/json", ".woff2": "font/woff2", ".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png" };

exports.launch = args => puppeteer.launch({ headless: true, args,
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: "chrome" }) });

exports.serve = root => new Promise(resolve => {
  root = path.resolve(root);
  const server = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(new URL(req.url, "http://127.0.0.1").pathname));
    if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  server.listen(0, "127.0.0.1", () => resolve({ url: `http://127.0.0.1:${server.address().port}/`, close: () => server.close() }));
});

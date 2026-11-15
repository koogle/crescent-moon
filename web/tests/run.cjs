// Runs a browser test suite of web/tests/ in headless Chromium with WebGPU:
//
//   node web/tests/run.cjs kernels
//   node web/tests/run.cjs reference --export artifacts/web/NAME [--image STEM] [--timeout MINUTES]
//
// Serves the repository root on a local port, opens web/tests/index.html and prints the
// page's log. This session's Chromium has no GPU, so WebGPU runs on SwiftShader (CPU):
// correctness only.

const http = require("http");
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..", "..");
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };

function serve(request, response) {
  const file = path.join(root, decodeURIComponent(new URL(request.url, "http://x").pathname));
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, {
    "Content-Type": types[path.extname(file)] || "application/octet-stream",
    "Content-Length": fs.statSync(file).size,
  });
  fs.createReadStream(file).pipe(response);
}

(async () => {
  const [suite, ...rest] = process.argv.slice(2);
  const options = {};
  for (let i = 0; i < rest.length; i += 2) options[rest[i].replace(/^--/, "")] = rest[i + 1];
  const server = http.createServer(serve).listen(0, "127.0.0.1");
  await new Promise((resolve) => server.once("listening", resolve));
  const query = new URLSearchParams({ suite });
  if (options.export) query.set("export", `/${options.export.replace(/\/$/, "")}/`);
  if (options.image) query.set("image", options.image);
  const browser = await chromium.launch({
    args: ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan",
      "--use-angle=swiftshader"],
  });
  const page = await browser.newPage();
  page.on("console", (message) => console.log(message.text()));
  page.on("pageerror", (error) => console.log(`PAGE ERROR ${error}`));
  await page.goto(`http://127.0.0.1:${server.address().port}/web/tests/index.html?${query}`);
  const minutes = Number(options.timeout || 30);
  await page.waitForFunction("window.done === true", null, { timeout: minutes * 60000 });
  const results = await page.evaluate("window.results");
  if (options.out) fs.writeFileSync(options.out, JSON.stringify(results, null, 1) + "\n");
  await browser.close();
  server.close();
  process.exit(results.failed ? 1 : 0);
})();

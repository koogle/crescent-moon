// Runs a browser test suite of web/tests/ in headless Chromium with WebGPU:
//
//   node web/tests/run.cjs kernels
//   node web/tests/run.cjs reference --export artifacts/web/NAME [--image STEM] [--timeout MINUTES]
//   node web/tests/run.cjs bench --export artifacts/web/NAME [--frames N] --gpu vulkan
//       [--budget SECONDS] [--slow SECONDS]   (the limits of the kernel timing, see tune)
//
// Serves the repository root on a local port, opens web/tests/index.html and prints the
// page's log. `--gpu swiftshader` (the default) runs WebGPU on the CPU, for correctness
// on a machine without a GPU; `--gpu vulkan` uses the machine's GPU through Vulkan
// (`modal_app.py::browser` runs that on a Modal GPU); `--gpu metal` uses a Mac's GPU, and
// with `--browser webkit` the suite runs in Playwright's WebKit, Safari's engine, on the
// system's GPU (`.github/workflows/browser-apple.yml` runs both on a macOS runner).

const http = require("http");
const fs = require("fs");
const path = require("path");
const { chromium, webkit } = require("playwright");

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
  for (const name of ["frames", "budget", "slow"]) if (options[name]) query.set(name, options[name]);
  const adapters = {
    swiftshader: { args: ["--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-angle=swiftshader"] },
    // The full browser in its new headless mode; the headless shell has no GPU process.
    vulkan: {
      channel: "chromium",
      args: ["--enable-features=Vulkan", "--use-angle=vulkan", "--ignore-gpu-blocklist",
        "--disable-vulkan-surface", "--enable-gpu"],
    },
    metal: { channel: "chromium", args: ["--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu"] },
  };
  const chosen = adapters[options.gpu || "swiftshader"];
  if (!chosen) throw new Error(`Unknown --gpu ${options.gpu}`);
  const browser = options.browser === "webkit"
    ? await webkit.launch()
    : await chromium.launch({ channel: chosen.channel, args: ["--enable-unsafe-webgpu", ...chosen.args] });
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

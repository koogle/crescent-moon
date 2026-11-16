"""Detection demo as a Hugging Face Space (Gradio on ZeroGPU).

Three tabs: compare Moondream 2's `detect` with the slot heads on an image, run one model
over a video, and detect live on a camera or a video file. The camera tab embeds
`camera.html`; its default model runs in the visitor's browser on WebGPU (`web/`, weights
from the HF model repository), and the Space's models answer its frames through the
`frame` API.

One Moondream 2 is loaded when the Space starts and shared between its own `detect`
skill and the slot heads listed in `demo.json`. On ZeroGPU a GPU is attached only while
a function decorated with `spaces.GPU` runs. `scripts/push_space.py` assembles and
pushes the Space; run locally with `python space/app.py` from a folder it assembled.
"""

import spaces  # before torch: ZeroGPU patches CUDA start-up  # isort: skip

import base64
import hashlib
import io
import json
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import gradio as gr
import torch
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from huggingface_hub import hf_hub_download
from PIL import Image

from crescent_moon.demo import colour, detections, draw, phrases_of, stream_video
from crescent_moon.moondream import MoondreamDetector
from crescent_moon.tasks import COCO_NAMES
from crescent_moon.tracking import Tracker
from crescent_moon.train_slots import SlotDetector, load_backbone

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "demo.json").read_text())
WEIGHTS = CONFIG["weights_repo"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MOONDREAM = "Moondream 2 detect"
CAMERA_SIDE = 640
CAMERA_SECONDS = 10 * 60
CAMERA_SESSIONS = 64
WEB = HERE / "web"

backbone = load_backbone(CONFIG["moondream"]["repo"], CONFIG["moondream"]["revision"], DEVICE)
MODELS = {
    MOONDREAM: MoondreamDetector(
        CONFIG["moondream"]["repo"], CONFIG["moondream"]["revision"], 150, DEVICE, model=backbone[0]
    )
}
for _name, _head in CONFIG["slot_heads"].items():
    MODELS[_name] = SlotDetector(
        hf_hub_download(WEIGHTS, f"{_head['run']}/head.pt"),
        DEVICE,
        threshold=_head["threshold"],
        backbone=backbone,
        **_head.get("options", {}),
    )
NAMES = list(MODELS)


def detect(name, image, phrases):
    """[(label, box, score)] of one model on one image, and the seconds it took."""
    if name != MOONDREAM:
        phrases = [p for p in phrases if p in COCO_NAMES]  # the heads know these only
    start = time.perf_counter()
    result = MODELS[name].detect(image, phrases)
    return detections(result, phrases), time.perf_counter() - start


@spaces.GPU(duration=60)
def detect_all(image, phrases, names):
    detect(names[0], image, phrases)  # warm the first model's kernels
    return {name: detect(name, image, phrases) for name in names}


@spaces.GPU(duration=10)
def detect_frame(image, name, phrases):
    return detect(name, image, phrases)


@spaces.GPU(duration=120)
def video_chunks(data, name, phrases, every, max_frames, track):
    yield from stream_video(
        data, lambda image: detect(name, image, phrases)[0], every, max_frames, track=track
    )


def compare(image, text, chosen):
    if image is None or not chosen:
        return [], "Choose an image and at least one model."
    image = image.convert("RGB")
    results = detect_all(image, phrases_of(text), chosen)
    gallery, rows = [], []
    for name in chosen:
        found, seconds = results[name]
        caption = f"{name}: {len(found)} boxes, {seconds:.3f} s"
        gallery.append((draw(image, found, caption), caption))
        rows.append([name, len(found), round(seconds, 3)])
    return gallery, rows


def video(path, text, name, every, max_frames, track):
    """Stream the annotated video chunk by chunk while the GPU works through it."""
    if not path:
        yield None, None
        return
    chunks = video_chunks(
        Path(path).read_bytes(), name, phrases_of(text), int(every), int(max_frames), track
    )
    folder = Path(tempfile.mkdtemp(prefix="stream-"))
    for index, (chunk, stats) in enumerate(chunks):
        target = folder / f"chunk-{index:05d}.ts"
        target.write_bytes(chunk)
        yield str(target), stats


sessions = OrderedDict()  # camera session id -> {"model", "tracker", "started"}


def frame(jpeg: str, model: str, categories: str, track: bool, session: str) -> dict:
    """Detections on one camera frame (base64 JPEG), in the frame's pixels; with `track`,
    through this session's tracker. {"stop": reason} ends the stream."""
    if model not in MODELS or not session:
        return {"stop": "Unknown model or missing session."}
    state = sessions.get(session)
    if state is None:
        state = sessions[session] = {"model": model, "tracker": None, "started": time.time()}
        while len(sessions) > CAMERA_SESSIONS:
            sessions.popitem(last=False)
    if time.time() - state["started"] > CAMERA_SECONDS:
        return {"stop": "This stream reached its time limit; start it again."}
    began = time.perf_counter()
    image = Image.open(io.BytesIO(base64.b64decode(jpeg))).convert("RGB")
    found, seconds = detect_frame(image, model, phrases_of(categories))
    if track:
        if state["tracker"] is None or state["model"] != model:
            state["model"], state["tracker"] = model, Tracker(smoothing=0.5, coast=1)
        tracks = state["tracker"].update(
            [b for _, b, _ in found], [s for *_, s in found], [n for n, *_ in found]
        )
        found = [(f"{t['label']} #{t['id']}", t["box"], t["score"]) for t in tracks]
    else:
        state["tracker"] = None
    return {
        "boxes": [
            {"label": label, "box": box, "score": score, "colour": colour(label.split(" #")[0])}
            for label, box, score in found
        ],
        "seconds": seconds,
        "server_seconds": time.perf_counter() - began,
    }


# The page imports the browser runtime from a path named after its content, so a browser
# never pairs the page with modules it cached from an earlier version of the Space.
FRESH = {"Cache-Control": "no-cache"}
WEB_FILES = ("kernels.js", "model.js", "select.js")
RUNTIME = (
    "web/"
    + hashlib.sha256(b"".join((WEB / n).read_bytes() for n in WEB_FILES)).hexdigest()[:12]
    + "/"
)


def camera_page():
    return HTMLResponse((HERE / "camera.html").read_text(), headers=FRESH)


def camera_config():
    export = CONFIG["device_export"]
    return {
        "models": NAMES,
        "default": CONFIG["camera_default"],
        "side": CAMERA_SIDE,
        "seconds": CAMERA_SECONDS,
        "device": {
            "export": export,
            "weights": f"https://huggingface.co/{WEIGHTS}/resolve/main/web/{export}/",
            "runtime": RUNTIME,
        },
    }


def web_file(version: str, name: str):
    """The browser runtime's modules, imported by the camera page."""
    if name not in WEB_FILES or f"web/{version}/" != RUNTIME:
        raise HTTPException(404)
    return FileResponse(WEB / name, media_type="text/javascript", headers=FRESH)


with gr.Blocks(title="Crescent Moon") as page:
    gr.Markdown((HERE / "intro.md").read_text())
    with gr.Tab("Camera"):
        gr.HTML(
            '<iframe src="camera" allow="camera" title="Live detection" '
            'style="width:100%;height:900px;border:0"></iframe>'
        )
    with gr.Tab("Images"):
        with gr.Row():
            picture = gr.Image(type="pil", label="Image")
            with gr.Column():
                text = gr.Textbox(label="Categories", value="person, car, dog, cup, chair")
                chosen = gr.CheckboxGroup(NAMES, value=NAMES[:2], label="Models")
                run = gr.Button("Detect", variant="primary")
        gallery = gr.Gallery(label="Results", columns=2, height="auto")
        summary = gr.Dataframe(headers=["Model", "Boxes", "Seconds"], label="Per model")
        run.click(compare, [picture, text, chosen], [gallery, summary])
        gr.Examples([[str(e)] for e in sorted((HERE / "examples").glob("*.jpg"))], [picture])
    with gr.Tab("Video"):
        with gr.Row():
            clip = gr.Video(label="Video")
            with gr.Column():
                vtext = gr.Textbox(label="Categories", value="")
                model = gr.Dropdown(NAMES, value=CONFIG["camera_default"], label="Model")
                every = gr.Slider(1, 30, value=3, step=1, label="Detect every n-th frame")
                limit = gr.Slider(30, 900, value=300, step=30, label="Frames to process")
                track = gr.Checkbox(value=True, label="Track across frames")
                vrun = gr.Button("Run", variant="primary")
        annotated = gr.Video(label="Annotated", streaming=True, autoplay=True)
        stats = gr.JSON(label="Statistics")
        vrun.click(video, [clip, vtext, model, every, limit, track], [annotated, stats])
    gr.api(frame, api_name="frame")

if __name__ == "__main__":
    app, _, _ = page.launch(prevent_thread_lock=True, ssr_mode=False, show_error=True)
    app.add_api_route("/camera", camera_page)
    app.add_api_route("/camera/config", camera_config)
    app.add_api_route("/web/{version}/{name}", web_file)
    page.block_thread()

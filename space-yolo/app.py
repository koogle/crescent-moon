"""YOLO11 on a camera stream: a Gradio Space on CPU, a reference for Crescent Moon.

Pushed with
    huggingface-cli upload jfrick/yolo11-camera space-yolo . --repo-type space
"""

import time

import gradio as gr
from ultralytics import YOLO

SIZES = {
    "YOLO11n (2.6M)": "yolo11n.pt",
    "YOLO11s (9.4M)": "yolo11s.pt",
    "YOLO11m (20M)": "yolo11m.pt",
}
models = {}


def detect(frame, size, confidence):
    """The frame with YOLO's boxes drawn, and the time the pass took."""
    if frame is None:
        return None, ""
    if size not in models:
        models[size] = YOLO(SIZES[size])
    start = time.perf_counter()
    (result,) = models[size].predict(frame, conf=confidence, verbose=False)
    seconds = time.perf_counter() - start
    # `plot` takes and returns BGR; the stream is RGB.
    drawn = result.plot(img=frame[..., ::-1].copy())[..., ::-1]
    return drawn, f"{len(result.boxes)} boxes in {seconds * 1000:.0f} ms on the Space's CPU"


with gr.Blocks(title="YOLO11 Camera") as page:
    gr.Markdown(
        "# YOLO11 on a camera stream\nUltralytics YOLO11, trained on COCO's 80 categories, "
        "on the Space's CPU. Allow the camera and press record to start."
    )
    with gr.Row():
        size = gr.Dropdown(list(SIZES), value="YOLO11n (2.6M)", label="Model")
        confidence = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
    with gr.Row():
        camera = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
        output = gr.Image(label="Detections", streaming=True)
    status = gr.Markdown()
    camera.stream(
        detect,
        [camera, size, confidence],
        [output, status],
        stream_every=0.2,
        time_limit=600,
        concurrency_limit=1,
    )

if __name__ == "__main__":
    page.launch()

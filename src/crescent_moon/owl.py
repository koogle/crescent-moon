"""OWLv2, an open-vocabulary detector, as a reference that takes its categories as text.

One pass scores every image patch against every phrase asked. The model returns many
overlapping boxes per object, so boxes of one phrase that overlap by IoU 0.5 are reduced
to the most confident one (non-maximum suppression) before they are scored.
"""

import time

import torch


class OwlDetector:
    def __init__(self, name, device, confidence=0.05, duplicate_iou=0.5):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(device).eval()
        self.device, self.confidence, self.duplicate_iou = device, confidence, duplicate_iou

    @torch.no_grad()
    def detect(self, image, phrases):
        """Return {"boxes", "scores": per-phrase lists, "shared_seconds", "phrase_seconds"}."""
        from torchvision.ops import batched_nms

        start = time.perf_counter()
        inputs = self.processor(
            text=[[f"a photo of a {phrase}" for phrase in phrases]],
            images=image,
            return_tensors="pt",
        ).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device != "cpu"):
            outputs = self.model(**inputs)
        # The processor pads the image to a square at the bottom and right, so boxes come
        # back in the square's pixels, which are the image's own.
        side = max(image.size)
        (result,) = self.processor.post_process_object_detection(
            outputs, threshold=self.confidence, target_sizes=torch.tensor([[side, side]])
        )
        corners = result["boxes"].float()
        corners[:, 0::2] = corners[:, 0::2].clamp(0, image.width)
        corners[:, 1::2] = corners[:, 1::2].clamp(0, image.height)
        keep = batched_nms(corners, result["scores"].float(), result["labels"], self.duplicate_iou)
        shared = time.perf_counter() - start
        boxes = [[] for _ in phrases]
        kept = [[] for _ in phrases]
        for index in keep.tolist():
            position = int(result["labels"][index])
            boxes[position].append(corners[index].tolist())
            kept[position].append(float(result["scores"][index]))
        return {
            "boxes": boxes,
            "scores": kept,
            "shared_seconds": shared,
            "phrase_seconds": [0.0] * len(phrases),
        }

"""A closed-set detector through Ultralytics, as a reference for speed and size.

YOLO models trained on COCO return all 80 categories in one pass, with a confidence per
box; their class indices follow COCO's order, as `tasks.COCO_NAMES` does.
"""

import time

from crescent_moon.tasks import COCO_NAMES


class YoloDetector:
    def __init__(self, name, device, confidence=0.001):
        from ultralytics import YOLO

        self.model = YOLO(name)
        self.device, self.confidence = device, confidence

    def detect(self, image, phrases):
        """Return {"boxes", "scores": per-phrase lists, "shared_seconds", "phrase_seconds"}:
        one pass over the image, its boxes handed to the phrases that name their category."""
        start = time.perf_counter()
        (result,) = self.model.predict(
            image, conf=self.confidence, device=self.device, verbose=False
        )
        found = result.boxes
        corners, scores, classes = found.xyxy.tolist(), found.conf.tolist(), found.cls.tolist()
        shared = time.perf_counter() - start
        boxes = [[] for _ in phrases]
        kept = [[] for _ in phrases]
        for box, score, index in zip(corners, scores, classes, strict=True):
            for position, phrase in enumerate(phrases):
                if phrase == COCO_NAMES[int(index)]:
                    boxes[position].append(box)
                    kept[position].append(score)
        return {
            "boxes": boxes,
            "scores": kept,
            "shared_seconds": shared,
            "phrase_seconds": [0.0] * len(phrases),
        }

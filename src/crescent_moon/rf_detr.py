"""RF-DETR, a real-time set-prediction detector trained on COCO, as a closed-set reference.

Like `yolo.YoloDetector`: one pass returns all 80 categories with a confidence per box,
and the boxes are handed to the phrases that name their category.
"""

import time

from crescent_moon.tasks import COCO_NAMES


class RfDetrDetector:
    def __init__(self, name, device, confidence=0.001):
        import rfdetr
        from rfdetr.util.coco_classes import COCO_CLASSES

        self.model = getattr(rfdetr, name)(device=device)
        self.confidence = confidence
        self.names = dict(COCO_CLASSES)
        missing = set(COCO_NAMES) - set(self.names.values())
        if missing:
            raise ValueError(f"RF-DETR's classes lack {sorted(missing)}")

    def detect(self, image, phrases):
        """Return {"boxes", "scores": per-phrase lists, "shared_seconds", "phrase_seconds"}."""
        start = time.perf_counter()
        found = self.model.predict(image, threshold=self.confidence)
        shared = time.perf_counter() - start
        boxes = [[] for _ in phrases]
        kept = [[] for _ in phrases]
        for box, score, index in zip(
            found.xyxy.tolist(), found.confidence.tolist(), found.class_id.tolist(), strict=True
        ):
            for position, phrase in enumerate(phrases):
                if phrase == self.names.get(int(index)):
                    boxes[position].append(box)
                    kept[position].append(score)
        return {
            "boxes": boxes,
            "scores": kept,
            "shared_seconds": shared,
            "phrase_seconds": [0.0] * len(phrases),
        }

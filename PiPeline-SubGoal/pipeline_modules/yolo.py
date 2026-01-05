# Scopo: detector YOLO e selezione bbox/centroide.
import numpy as np

from .utils import normalize_label


class YOLODetector:
    def __init__(self, model_path: str, conf: float, imgsz: int, target_label: str):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics not available. Install ultralytics and torch.") from e
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.target_label = normalize_label(target_label)

    def detect(self, frame_rgb: np.ndarray):
        results = self.model.predict(source=frame_rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if not results:
            return None, None
        res0 = results[0]
        boxes = res0.boxes
        if boxes is None or boxes.cls is None or len(boxes) == 0:
            return None, None
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy().astype(float)
        names = getattr(res0, "names", None) or getattr(self.model, "names", {}) or {}
        masks = getattr(res0, "masks", None)
        masks_data = None
        if masks is not None and getattr(masks, "data", None) is not None:
            masks_data = masks.data.detach().cpu().numpy().astype(np.float32)

        selected_idx = None
        if self.target_label:
            for i, cid in enumerate(cls_ids):
                name = normalize_label(str(names.get(int(cid), cid)))
                if name == self.target_label:
                    selected_idx = i if selected_idx is None else selected_idx
                    if confs[i] > confs[selected_idx]:
                        selected_idx = i
            if selected_idx is None:
                return None, None
        if selected_idx is None:
            selected_idx = int(np.argmax(confs))

        x1, y1, x2, y2 = xyxy[selected_idx].tolist()
        mask01 = None
        if masks_data is not None and selected_idx < len(masks_data):
            mask01 = masks_data[selected_idx]
        if mask01 is not None and mask01.size > 0:
            ys, xs = np.where(mask01 > 0.5)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
            else:
                cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
        else:
            cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)

        label = str(names.get(int(cls_ids[selected_idx]), cls_ids[selected_idx]))
        det = {
            "label": label,
            "score": float(confs[selected_idx]),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "centroid_px": [int(cx), int(cy)],
            "has_mask": bool(mask01 is not None),
        }
        return det, mask01

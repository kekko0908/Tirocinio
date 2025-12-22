from common import (
    EnvConfig,
    make_controller,
    get_rgb_bgr,
    data_dir,
    save_rgb,
    print_ok,
    print_warn,
    load_json,
    save_json,
)
import argparse
import numpy as np
import cv2


def centroid_from_mask(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())

def bbox_from_mask(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2, y2]


def pick_detection_idx(result, target_label: str):
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return None

    names = getattr(result, "names", None) or {}
    target = (target_label or "").strip().lower()

    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy().astype(float)

    best_any = int(np.argmax(confs))
    best_match = None
    best_match_conf = -1.0

    for i, (cid, conf) in enumerate(zip(cls_ids, confs)):
        name = str(names.get(int(cid), cid)).lower()
        if target and name == target and conf > best_match_conf:
            best_match = i
            best_match_conf = conf

    return best_match if best_match is not None else best_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt", help="Path/preset (es. yolov8n-seg.pt o data/models/yolo-seg.pt)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pred_conf_min", type=float, default=0.05, help="Conf minima usata per la predizione (per non perdere target piccoli).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--target", default="", help="Label YOLO da cercare (default: objectType da teleport_state).")
    ap.add_argument("--oracle_assist", action="store_true", help="Se attivo, usa instance_masks di THOR per fallback/validazione quando YOLO sbaglia.")
    ap.add_argument("--max_detections", type=int, default=20, help="Quante detections salvare nel json (debug).")
    ap.add_argument("--out_state", default="centroid_state.json", help="Output in data/state/")
    ap.add_argument("--debug_name", default="03_yolo_centroid.png", help="Output debug in data/outputs/")
    args = ap.parse_args()

    tele = load_json("teleport_state.json")
    scene = tele["scene"]
    target_id = tele["target"]["objectId"]
    target_type = tele["target"]["objectType"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    target_label = args.target.strip() or str(target_type).lower()
    target_label = target_label.lower()

    cfg = EnvConfig(scene=scene, render_depth=False, render_instance_segmentation=bool(args.oracle_assist))
    controller = make_controller(cfg)
    controller.step(action="Teleport", position=agent_pos, rotation=agent_rot, horizon=horizon)
    event = controller.last_event

    rgb_bgr = get_rgb_bgr(event)
    # Salva sempre un frame standard per i passi successivi (VLM/debug)
    try:
        save_rgb(rgb_bgr, "02_teleport_rgb.png")
    except Exception:
        pass
    oracle_mask01 = None
    oracle_centroid = None
    oracle_bbox = None
    if args.oracle_assist and getattr(event, "instance_masks", None) is not None:
        masks = event.instance_masks
        if isinstance(masks, dict) and target_id in masks:
            oracle_mask01 = masks[target_id].astype(np.float32)
            oracle_centroid = centroid_from_mask(oracle_mask01)
            oracle_bbox = bbox_from_mask(oracle_mask01)

    try:
        from ultralytics import YOLO
    except Exception as e:
        controller.stop()
        raise RuntimeError(
            "Ultralytics non disponibile. Installa: pip install ultralytics (e torch adatto alla tua macchina)."
        ) from e

    model = YOLO(args.model)
    # Per target piccoli (es. Apple) conviene predire con conf piu' bassa e filtrare dopo.
    pred_conf = min(float(args.conf), float(args.pred_conf_min))
    results = model.predict(source=event.frame, conf=pred_conf, imgsz=int(args.imgsz), verbose=False)
    if not results:
        print_warn("YOLO non ha restituito risultati.")
        controller.stop()
        return

    res0 = results[0]
    boxes = res0.boxes
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        print_warn("Nessuna detection trovata (boxes vuote).")
        controller.stop()
        return

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy().astype(float)
    names = getattr(res0, "names", None) or getattr(model, "names", {}) or {}

    masks = getattr(res0, "masks", None)
    masks_data = None
    if masks is not None and getattr(masks, "data", None) is not None:
        masks_data = masks.data.detach().cpu().numpy().astype(np.float32)

    detections = []
    for i in range(len(confs)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        det_cls = int(cls_ids[i])
        det_conf = float(confs[i])
        det_name = str(names.get(det_cls, det_cls))
        det_name_l = det_name.lower()
        mask01 = None
        if masks_data is not None and i < len(masks_data):
            mask01 = masks_data[i]
        c = centroid_from_mask(mask01) if mask01 is not None else None
        if c is None:
            c = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
        area = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
        detections.append(
            {
                "i": int(i),
                "label": det_name,
                "label_l": det_name_l,
                "score": det_conf,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "centroid_px": [int(c[0]), int(c[1])],
                "has_mask": bool(mask01 is not None),
                "bbox_area": area,
            }
        )

    # --- selection ---
    selected_idx = None
    selected_reason = ""

    # 1) match esatto label
    matches = [d for d in detections if d["label_l"] == target_label]
    if matches:
        matches.sort(key=lambda d: d["score"], reverse=True)
        selected_idx = matches[0]["i"]
        selected_reason = "label_match"

    # 2) oracle assist: seleziona per overlap con target mask anche se label sbagliata
    if selected_idx is None and args.oracle_assist and oracle_mask01 is not None and masks_data is not None:
        best_iou = -1.0
        best_i = None
        gt = oracle_mask01 > 0.5
        gt_area = float(gt.sum())
        if gt_area > 0:
            for i in range(min(len(masks_data), len(detections))):
                pred = masks_data[i] > 0.5
                inter = float(np.logical_and(pred, gt).sum())
                union = float(np.logical_or(pred, gt).sum())
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
            if best_i is not None and best_iou >= 0.02:
                selected_idx = int(best_i)
                selected_reason = f"oracle_overlap_iou={best_iou:.3f}"

    # 3) fallback: migliore per score (ma almeno conf >= args.conf se possibile)
    if selected_idx is None:
        above = [d for d in detections if d["score"] >= float(args.conf)]
        pool = above if above else detections
        pool.sort(key=lambda d: d["score"], reverse=True)
        selected_idx = pool[0]["i"]
        selected_reason = "best_score"

    # 4) se YOLO non trova davvero Apple, usa oracle bbox/centroid come fallback forte
    use_oracle_output = False
    if args.oracle_assist and oracle_centroid is not None:
        # se la label selezionata non coincide col target, preferisci oracle (evita bowl->apple)
        sel_label = detections[selected_idx]["label_l"] if selected_idx is not None else ""
        if sel_label != target_label:
            use_oracle_output = True
            selected_reason = f"oracle_fallback (yolo_label={sel_label})"

    if use_oracle_output:
        cx, cy = oracle_centroid
        x1, y1, x2, y2 = oracle_bbox if oracle_bbox else [0, 0, 0, 0]
        mask01 = oracle_mask01
        det_name = f"ORACLE({target_label})"
        det_conf = 1.0
        selected_bbox = [float(x1), float(y1), float(x2), float(y2)]
        has_mask = True
    else:
        x1, y1, x2, y2 = xyxy[selected_idx].tolist()
        det_cls = int(cls_ids[selected_idx])
        det_conf = float(confs[selected_idx])
        det_name = str(names.get(det_cls, det_cls))
        mask01 = None
        if masks_data is not None and selected_idx < len(masks_data):
            mask01 = masks_data[selected_idx]
        cx, cy = detections[selected_idx]["centroid_px"]
        selected_bbox = [float(x1), float(y1), float(x2), float(y2)]
        has_mask = bool(mask01 is not None)

    # debug overlay
    overlay = rgb_bgr.copy()
    if mask01 is not None:
        mask8 = (mask01 > 0.5).astype(np.uint8) * 255
        color = np.zeros_like(overlay)
        color[:, :, 1] = mask8  # green
        overlay = cv2.addWeighted(overlay, 1.0, color, 0.35, 0.0)

    cv2.rectangle(overlay, (int(selected_bbox[0]), int(selected_bbox[1])), (int(selected_bbox[2]), int(selected_bbox[3])), (255, 255, 0), 2)
    cv2.circle(overlay, (int(cx), int(cy)), 6, (0, 0, 255), -1)
    cv2.putText(
        overlay,
        f"{det_name} {det_conf:.2f} target={target_label} ({selected_reason})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_img = data_dir("outputs") / args.debug_name
    cv2.imwrite(str(out_img), overlay)

    # state json (manteniamo compatibilita' con src/04_pregrasp_point.py)
    out_state = {
        "scene": scene,
        "target": {"objectId": target_id, "objectType": target_type},
        "centroid": {"x": int(cx), "y": int(cy)},
        "yolo": {
            "model": args.model,
            "conf_select": float(args.conf),
            "conf_predict": float(pred_conf),
            "imgsz": int(args.imgsz),
            "requested_label": target_label,
            "detected_label": det_name,
            "score": det_conf,
            "bbox_xyxy": selected_bbox,
            "has_mask": bool(has_mask),
            "selected_reason": selected_reason,
            "detections": [
                {
                    "label": d["label"],
                    "score": float(d["score"]),
                    "bbox_xyxy": d["bbox_xyxy"],
                    "centroid_px": d["centroid_px"],
                    "has_mask": bool(d["has_mask"]),
                }
                for d in sorted(detections, key=lambda d: d["score"], reverse=True)[: max(1, int(args.max_detections))]
            ],
        },
    }

    if args.oracle_assist:
        out_state["oracle"] = {
            "available": bool(oracle_mask01 is not None),
            "centroid": {"x": int(oracle_centroid[0]), "y": int(oracle_centroid[1])} if oracle_centroid else None,
            "bbox_xyxy": [int(x) for x in oracle_bbox] if oracle_bbox else None,
        }

    save_json(out_state, args.out_state)
    print_ok(f"Target tele: {target_type} ({target_id})")
    print_ok(f"YOLO det: {det_name} conf={det_conf:.3f} | centroid px=({cx},{cy})")
    print_ok(f"Debug: {out_img}")
    print_ok(f"Salvato: data/state/{args.out_state}")

    controller.stop()


if __name__ == "__main__":
    main()

"""
WBC detection + classification website.

Workflow:
  1. User attaches a folder of images from the browser.
  2. Images are shown one at a time, in order, with pan/zoom.
  3. "Auto detect" first stain-normalises the field (Reinhard LAB mean/std
     match onto the classifier's training stain - see stain_eval/stain.py),
     then runs the fine-tuned GroundingDINO checkpoint (WBC/finetuned/best.pth)
     to find white-blood-cell boxes, then crops each box (same crop geometry
     used to train the classifier - cell box fills ~42% of the crop) and runs
     the YOLOv8 classifier
     (label_website/wbc_website_train/rgb_cells_setting_website_v8m/weights/best.pt)
     to assign a class label to every box.
  4. The user can drag/resize boxes and change their class; edits are saved
     as YOLO-format label files.

Run:
    python3 app.py
Then open http://localhost:5080
"""
import io
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

_here = Path(__file__).resolve().parent

# ------------------------------------------------------------------ paths ---
_vision_root = Path(os.environ.get(
    "VISION_ROOT", "/home/skuba/skuba_ws/src/try_vision_project")).resolve()
_vectection = _vision_root / "vectection"
_gdino_repo = _vectection / "GroundingDINO"
if str(_gdino_repo) not in sys.path:
    sys.path.insert(0, str(_gdino_repo))
_stain_eval = _vision_root / "stain_eval"
if str(_stain_eval) not in sys.path:
    sys.path.insert(0, str(_stain_eval))

import groundingdino.datasets.transforms as gd_transforms  # noqa: E402
from groundingdino.util.inference import load_model, predict  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from stain import reinhard  # noqa: E402  (stain_eval/stain.py)

DET_WEIGHTS = Path(os.environ.get("WBC_DET_WEIGHTS", _vectection / "WBC/finetuned/best.pth"))
GDINO_CONFIG = str(_gdino_repo / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
TEXT_PROMPT = os.environ.get("WBC_PROMPT", "white blood cell")

CLS_WEIGHTS = Path(os.environ.get(
    "WBC_CLS_WEIGHTS",
    _vision_root / "label_website/wbc_website_train/rgb_cells_setting_website_v8m/weights/best.pt",
))

# LAB mean/std of the classifier's training crops (fit_reference.py). Fields
# get Reinhard-matched onto this before detection/classification so their
# stain looks like what the models were trained on.
STAIN_REFERENCE = Path(os.environ.get("WBC_STAIN_REFERENCE", _stain_eval / "stain_reference.npz"))
_stain_ref = np.load(STAIN_REFERENCE) if STAIN_REFERENCE.exists() else None
STAIN_LAB_MEAN = _stain_ref["lab_mean"].astype(np.float32) if _stain_ref is not None else None
STAIN_LAB_STD = _stain_ref["lab_std"].astype(np.float32) if _stain_ref is not None else None

UPLOAD_ROOT = _here / "uploads"
LABEL_ROOT = _here / "labels"
# per-image review bookkeeping (model output snapshot + reviewed flag), kept
# next to the labels so it rides on the same mounted volume
META_ROOT = LABEL_ROOT / "_meta"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
LABEL_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TRANSCODE_EXT = {".tif", ".tiff", ".bmp"}
JPEG_QUALITY = 88

# crop geometry copied from stain_eval/detect_rgb_twostage.py: in the
# classifier's training crops the cell box fills ~42% of the crop width, so
# the crop side is ~2.4x the box side. Matching that puts cells at the scale
# the classifier actually saw during training.
CROP_SCALE = 1 / 0.42
CROP_PX = 384

DETECT_DEFAULTS = {
    "box_threshold": 0.30,
    "text_threshold": 0.25,
    "max_box_area_frac": 0.20,
    "nms_iou": 0.5,
    "normalize_stain": True,
}

_GDINO_TRANSFORM = gd_transforms.Compose([
    gd_transforms.RandomResize([800], max_size=1333),
    gd_transforms.ToTensor(),
    gd_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = Flask(__name__, static_folder=str(_here / "static"), static_url_path="/static")

GPU_LOCK = threading.Lock()
_det_model = None
_cls_model = None
_model_lock = threading.Lock()


def get_det_model():
    global _det_model
    with _model_lock:
        if _det_model is None:
            if not DET_WEIGHTS.exists():
                raise RuntimeError(f"detector weights not found: {DET_WEIGHTS}")
            print(f"loading detector {DET_WEIGHTS} on {DEVICE} ...", flush=True)
            _det_model = load_model(GDINO_CONFIG, str(DET_WEIGHTS), device=DEVICE)
            print("detector ready", flush=True)
    return _det_model


def get_cls_model():
    global _cls_model
    with _model_lock:
        if _cls_model is None:
            if not CLS_WEIGHTS.exists():
                raise RuntimeError(f"classifier weights not found: {CLS_WEIGHTS}")
            print(f"loading classifier {CLS_WEIGHTS} ...", flush=True)
            _cls_model = YOLO(str(CLS_WEIGHTS))
            print("classifier ready", flush=True)
    return _cls_model


CLASSES = list(get_cls_model().names.values()) if CLS_WEIGHTS.exists() else []
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}


# ------------------------------------------------------------------ sessions
# One "session" = one folder the user attached. Sessions live under
# uploads/<session_id>/ and never get mixed up with each other.

def session_dir(sid):
    return UPLOAD_ROOT / sid


def safe_session(sid):
    if not sid or not sid.replace("-", "").isalnum():
        return None
    d = session_dir(sid)
    return sid if d.is_dir() else None


def safe_image(sid, name):
    base = session_dir(sid).resolve()
    p = (base / name).resolve()
    if base != p.parent or not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return p


def label_path(sid, name):
    p = LABEL_ROOT / sid / (Path(name).stem + ".txt")
    return p


def read_label(sid, name):
    p = label_path(sid, name)
    if not p.exists():
        return None
    boxes = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, cx, cy, w, h = parts
        boxes.append({"class_id": int(cid), "cx": float(cx), "cy": float(cy),
                      "w": float(w), "h": float(h)})
    return boxes


def write_label(sid, name, boxes):
    p = label_path(sid, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{int(b['class_id'])} {float(b['cx']):.6f} {float(b['cy']):.6f} "
            f"{float(b['w']):.6f} {float(b['h']):.6f}" for b in boxes]
    p.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)


# ---------------------------------------------------------- review metadata --
# labels/_meta/<sid>/<stem>.json = {"pred": [model boxes] | null,
#                                    "reviewed": bool, "saves": int}
# `pred` is the last auto-detect output for the image; the saved label file is
# the human-corrected result. Comparing the two gives the edit / accuracy stats.

META_LOCK = threading.Lock()
MATCH_IOU = 0.5     # pred and final box are "the same cell" above this IoU
ADJUST_IOU = 0.9    # ...but count as a box edit (moved/resized) below this


def meta_path(sid, name):
    return META_ROOT / sid / (Path(name).stem + ".json")


def read_meta(sid, name):
    p = meta_path(sid, name)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"pred": None, "reviewed": False, "saves": 0}


def update_meta(sid, name, **changes):
    with META_LOCK:
        meta = read_meta(sid, name)
        meta.update(changes)
        p = meta_path(sid, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta))
        return meta


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2, a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx0, by0, bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)


def match_boxes(pred, final):
    """Greedy one-to-one IoU matching -> (pairs[(pi, fi, iou)], unmatched_pred, unmatched_final)."""
    cand = sorted(((_iou(p, f), pi, fi) for pi, p in enumerate(pred) for fi, f in enumerate(final)),
                  reverse=True)
    used_p, used_f, pairs = set(), set(), []
    for iou, pi, fi in cand:
        if iou < MATCH_IOU:
            break
        if pi in used_p or fi in used_f:
            continue
        used_p.add(pi); used_f.add(fi); pairs.append((pi, fi, iou))
    return (pairs,
            [i for i in range(len(pred)) if i not in used_p],
            [i for i in range(len(final)) if i not in used_f])


def compute_stats(sids):
    """Aggregate annotation progress, edit counts and model accuracy over sessions."""
    n = len(CLASSES)
    out = {
        "images_total": 0, "images_annotated": 0, "images_reviewed": 0,
        "images_edited": 0, "boxes_final": 0,
        "edits": {"class_changed": 0, "box_adjusted": 0, "deleted": 0, "added": 0, "total": 0},
        "matched": 0, "model_boxes": 0, "class_correct": 0,
    }
    confusion = [[0] * n for _ in range(n)]   # [model class][final class], matched boxes
    added_by_class = [0] * n                    # boxes the model missed entirely
    deleted_by_class = [0] * n                  # model boxes the human removed
    for sid in sids:
        up = UPLOAD_ROOT / sid
        if up.is_dir():
            out["images_total"] += sum(1 for p in up.iterdir()
                                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        ldir = LABEL_ROOT / sid
        if not ldir.is_dir():
            continue
        for lp in ldir.glob("*.txt"):
            out["images_annotated"] += 1
            final = read_label(sid, lp.name) or []
            out["boxes_final"] += len(final)
            meta = read_meta(sid, lp.name)
            if not meta.get("reviewed"):
                continue
            out["images_reviewed"] += 1
            pred = meta.get("pred")
            if pred is None:
                continue  # hand-labelled, nothing to compare against
            pairs, lost_p, lost_f = match_boxes(pred, final)
            img_edits = 0
            for pi, fi, iou in pairs:
                pc, fc = pred[pi]["class_id"], final[fi]["class_id"]
                if 0 <= pc < n and 0 <= fc < n:
                    confusion[pc][fc] += 1
                if pc == fc:
                    out["class_correct"] += 1
                else:
                    out["edits"]["class_changed"] += 1; img_edits += 1
                if iou < ADJUST_IOU:
                    out["edits"]["box_adjusted"] += 1; img_edits += 1
            for pi in lost_p:
                if 0 <= pred[pi]["class_id"] < n:
                    deleted_by_class[pred[pi]["class_id"]] += 1
            for fi in lost_f:
                if 0 <= final[fi]["class_id"] < n:
                    added_by_class[final[fi]["class_id"]] += 1
            out["edits"]["deleted"] += len(lost_p)
            out["edits"]["added"] += len(lost_f)
            img_edits += len(lost_p) + len(lost_f)
            out["matched"] += len(pairs)
            out["model_boxes"] += len(pred)
            out["images_edited"] += 1 if img_edits else 0
    e = out["edits"]
    e["total"] = e["class_changed"] + e["box_adjusted"] + e["deleted"] + e["added"]

    per_class = []
    for i, name in enumerate(CLASSES):
        predicted = sum(confusion[i]) + deleted_by_class[i]   # model said class i
        correct = confusion[i][i]
        actual = sum(row[i] for row in confusion) + added_by_class[i]  # truly class i
        per_class.append({
            "class": name, "predicted": predicted, "correct": correct,
            "wrong": predicted - correct, "actual": actual,
            "precision": correct / predicted if predicted else None,
            "recall": correct / actual if actual else None,
        })
    out["per_class"] = per_class
    out["confusion"] = confusion
    out["classes"] = CLASSES
    out["class_accuracy"] = out["class_correct"] / out["matched"] if out["matched"] else None
    out["detection_precision"] = out["matched"] / out["model_boxes"] if out["model_boxes"] else None
    return out


def stats_sessions(sid_arg):
    if sid_arg:
        return [sid_arg] if safe_session(sid_arg) else None
    ids = {p.name for p in UPLOAD_ROOT.iterdir() if p.is_dir()}
    ids |= {p.name for p in LABEL_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")}
    return sorted(ids)


# --------------------------------------------------------------- detection --

def nms(boxes, scores, iou_threshold, contain_threshold=0.8):
    """Greedy NMS over cxcywh boxes -> indices to keep."""
    if len(boxes) < 2:
        return list(range(len(boxes)))
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    areas = (w * h).clamp(min=1e-12)
    order = torch.argsort(scores, descending=True).tolist()

    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        rest = torch.tensor(order)
        ix0 = torch.maximum(x0[i], x0[rest])
        iy0 = torch.maximum(y0[i], y0[rest])
        ix1 = torch.minimum(x1[i], x1[rest])
        iy1 = torch.minimum(y1[i], y1[rest])
        inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter)
        ios = inter / torch.minimum(areas[i], areas[rest])
        drop = (iou > iou_threshold) | (ios > contain_threshold)
        order = [o for o, d in zip(order, drop.tolist()) if not d]
    return keep


def normalize_stain_bgr(bgr):
    """Reinhard LAB mean/std match onto the classifier's training stain."""
    if STAIN_LAB_MEAN is None:
        return bgr
    return reinhard(bgr, STAIN_LAB_MEAN, STAIN_LAB_STD)


def crop_square(bgr, cx_px, cy_px, side):
    """Centred square crop at CROP_PX, edge-replicated where it runs off the image.

    Matches stain_eval/detect_rgb_twostage.py's crop_square so cells land at
    the scale/framing the classifier was trained on.
    """
    H, W = bgr.shape[:2]
    half = side / 2.0
    x0, y0 = int(round(cx_px - half)), int(round(cy_px - half))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - W), max(0, y1 - H)
    sub = bgr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if sub.size == 0:
        return None
    if pl or pt or pr or pb:
        sub = cv2.copyMakeBorder(sub, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return cv2.resize(sub, (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA)


def classify_crop(bgr_field, box):
    """Crop `box` (cxcywh, normalised) out of the (already stain-matched) field and classify it."""
    H, W = bgr_field.shape[:2]
    cx, cy, w, h = box
    side = max(w * W, h * H) * CROP_SCALE
    crop = crop_square(bgr_field, cx * W, cy * H, side)
    if crop is None:
        return None, 0.0

    model = get_cls_model()
    with GPU_LOCK:
        res = model.predict(crop, verbose=False, device=DEVICE)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, 0.0
    confs = res.boxes.conf
    best = int(torch.argmax(confs))
    cid = int(res.boxes.cls[best])
    score = float(confs[best])
    return cid, score


def gdino_predict_bgr(bgr, box_threshold, text_threshold):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    image_transformed, _ = _GDINO_TRANSFORM(pil_img, None)
    model = get_det_model()
    with GPU_LOCK:
        boxes, logits, phrases = predict(
            model=model, image=image_transformed, caption=TEXT_PROMPT,
            box_threshold=box_threshold, text_threshold=text_threshold,
            device=DEVICE,
        )
    return boxes, logits


def detect_and_classify(image_path, box_threshold, text_threshold,
                        max_box_area_frac, nms_iou, normalize_stain):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        # cv2 cannot read some TIFF variants - fall back to PIL
        bgr = cv2.cvtColor(np.asarray(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = normalize_stain_bgr(bgr) if normalize_stain else bgr

    boxes, logits = gdino_predict_bgr(bgr, box_threshold, text_threshold)
    if len(boxes):
        area = boxes[:, 2] * boxes[:, 3]
        keep = area <= max_box_area_frac
        boxes, logits = boxes[keep], logits[keep]
    if nms_iou and len(boxes) > 1:
        idx = nms(boxes, logits, float(nms_iou))
        boxes, logits = boxes[idx], logits[idx]

    out = []
    for (cx, cy, w, h), det_score in zip(boxes.tolist(), logits.tolist()):
        cid, cls_score = classify_crop(bgr, (cx, cy, w, h))
        out.append({
            "class_id": cid if cid is not None else -1,
            "class_name": CLASSES[cid] if cid is not None and cid < len(CLASSES) else "Unknown",
            "cx": cx, "cy": cy, "w": w, "h": h,
            "det_score": round(float(det_score), 4),
            "cls_score": round(float(cls_score), 4),
        })
    return out


# ------------------------------------------------------------------- routes --

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config")
def api_config():
    return jsonify({
        "det_weights": str(DET_WEIGHTS),
        "cls_weights": str(CLS_WEIGHTS),
        "device": DEVICE,
        "classes": CLASSES,
        "defaults": DETECT_DEFAULTS,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept a batch of files (a whole attached folder) and start a session."""
    sid = request.form.get("session_id") or str(uuid.uuid4())
    d = session_dir(sid)
    d.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in request.files.getlist("files"):
        rel = f.filename or ""
        name = Path(rel).name  # flatten - ignore sub-folder structure
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # de-dupe flattened names
        target = d / name
        if target.exists():
            target = d / f"{Path(name).stem}_{uuid.uuid4().hex[:6]}{Path(name).suffix}"
        f.save(target)
        saved.append(target.name)

    if not saved:
        return jsonify({"error": "no images found in the attached folder"}), 400

    saved.sort()
    return jsonify({"session_id": sid, "images": saved})


@app.route("/api/images")
def api_images():
    sid = safe_session(request.args.get("session_id", ""))
    if not sid:
        return jsonify({"error": "unknown session"}), 404
    names = sorted(p.name for p in session_dir(sid).iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    labeled_stems = {p.stem for p in (LABEL_ROOT / sid).glob("*.txt")} if (LABEL_ROOT / sid).is_dir() else set()
    return jsonify([{"name": n, "labeled": Path(n).stem in labeled_stems} for n in names])


@app.route("/img/<sid>/<path:name>")
def api_img(sid, name):
    sid = safe_session(sid)
    p = safe_image(sid, name) if sid else None
    if not p:
        return jsonify({"error": "not found"}), 404
    if p.suffix.lower() not in TRANSCODE_EXT:
        return send_file(str(p))
    buf = io.BytesIO()
    with Image.open(p) as im:
        im.convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/label")
def api_get_label():
    sid = safe_session(request.args.get("session_id", ""))
    name = request.args.get("name", "")
    if not sid or not safe_image(sid, name):
        return jsonify({"error": "not found"}), 404
    boxes = read_label(sid, name)
    return jsonify({"saved": boxes is not None, "boxes": boxes or []})


@app.route("/api/label", methods=["POST"])
def api_put_label():
    body = request.get_json(force=True) or {}
    sid = safe_session(body.get("session_id", ""))
    name = body.get("name", "")
    if not sid or not safe_image(sid, name):
        return jsonify({"error": "not found"}), 404
    n = write_label(sid, name, body.get("boxes", []))
    # Detect All writes raw model output (reviewed=false); an explicit Save by
    # the annotator is a review, so only those feed the accuracy table.
    meta = read_meta(sid, name)
    update_meta(sid, name, reviewed=bool(body.get("reviewed", True)),
                saves=meta.get("saves", 0) + 1)
    return jsonify({"ok": True, "num_boxes": n})


@app.route("/api/stats")
def api_stats():
    sids = stats_sessions(request.args.get("session_id", ""))
    if sids is None:
        return jsonify({"error": "unknown session"}), 404
    return jsonify(compute_stats(sids))


@app.route("/api/stats.csv")
def api_stats_csv():
    sids = stats_sessions(request.args.get("session_id", ""))
    if sids is None:
        return jsonify({"error": "unknown session"}), 404
    st = compute_stats(sids)
    fmt = lambda v: "" if v is None else f"{v:.4f}"
    lines = ["class,model_predicted,correct,wrong,actual,precision,recall"]
    for c in st["per_class"]:
        lines.append(f"{c['class']},{c['predicted']},{c['correct']},{c['wrong']},{c['actual']},"
                     f"{fmt(c['precision'])},{fmt(c['recall'])}")
    lines += ["", "confusion (rows=model, cols=final)," + ",".join(st["classes"])]
    for name, row in zip(st["classes"], st["confusion"]):
        lines.append(name + "," + ",".join(map(str, row)))
    return app.response_class("\n".join(lines) + "\n", mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=model_accuracy.csv"})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    body = request.get_json(force=True) or {}
    sid = safe_session(body.get("session_id", ""))
    name = body.get("name", "")
    p = safe_image(sid, name) if sid else None
    if not p:
        return jsonify({"error": "not found"}), 404
    try:
        boxes = detect_and_classify(
            p,
            float(body.get("box_threshold", DETECT_DEFAULTS["box_threshold"])),
            float(body.get("text_threshold", DETECT_DEFAULTS["text_threshold"])),
            float(body.get("max_box_area_frac", DETECT_DEFAULTS["max_box_area_frac"])),
            float(body.get("nms_iou", DETECT_DEFAULTS["nms_iou"])),
            bool(body.get("normalize_stain", DETECT_DEFAULTS["normalize_stain"])),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # snapshot the raw model output so later human edits can be measured against it
    update_meta(sid, name, reviewed=False, pred=[{k: b[k] for k in ("class_id", "cx", "cy", "w", "h")} for b in boxes])
    return jsonify({"boxes": boxes})


if __name__ == "__main__":
    print(f"detector  : {DET_WEIGHTS}")
    print(f"classifier: {CLS_WEIGHTS}")
    print(f"device    : {DEVICE}")
    print(f"classes   : {CLASSES}")
    port = int(os.environ.get("PORT", 5080))
    print(f"open http://localhost:{port}")
    # load the detector now so the first Auto Detect click is not a long stall
    threading.Thread(target=get_det_model, daemon=True).start()
    try:
        from waitress import serve
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)
    else:
        # folders are uploaded in batches of large .tif files: allow big bodies
        # and slow clients instead of dropping the connection
        serve(app, host="0.0.0.0", port=port,
              threads=int(os.environ.get("WEB_THREADS", 8)),
              max_request_body_size=16 * 1024 ** 3,
              channel_timeout=600, cleanup_interval=60)

"""
WBC label review website (CPU only, no model).

Workflow:
  1. User attaches (a) label files - YOLO .txt (one per image, `class cx cy w h`,
     matched to the image by file name) or a .csv - and (b) the images.
  2. As soon as both are chosen they are uploaded, the CSV rows are matched to
     the images by file name and the review workspace opens.
  3. The user can drag/resize boxes and change their class; edits are saved
     as YOLO-format label files.

CSV columns (header row, names are case-insensitive, any order):
  image : filename | file | image | image_name | img | path   (extension optional)
  class : class | class_name | label | category | cls | class_id (name or number)
  box   : EITHER  cx,cy,w,h            (YOLO, normalised 0-1; also x_center,y_center,width,height)
          OR      xmin,ymin,xmax,ymax  (pixels; also x1,y1,x2,y2)
          OR      x,y,w,h              (pixels, top-left corner + size)

Run:
    python3 app.py
Then open http://localhost:5080
"""
import csv
import io
import json
import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

_here = Path(__file__).resolve().parent

DEFAULT_CLASSES = ['Band Neutrophil', 'Basophil', 'Blast cell', 'Eosinophil', 'Lymphocyte',
                   'Mast cell', 'Monocyte', 'Segmented Neutrophil', 'Toxic neutrophil',
                   'nRBC', 'Giant platelet', 'Macrophage']
CLASSES = list(DEFAULT_CLASSES)

UPLOAD_ROOT = _here / "uploads"
LABEL_ROOT = _here / "labels"
# per-image review bookkeeping (original CSV boxes snapshot + reviewed flag)
META_ROOT = LABEL_ROOT / "_meta"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
LABEL_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TRANSCODE_EXT = {".tif", ".tiff", ".bmp"}
JPEG_QUALITY = 88

app = Flask(__name__, static_folder=str(_here / "static"), static_url_path="/static")


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
    classes = session_classes(sids[0]) if len(sids) == 1 else list(CLASSES)
    n = len(classes)
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
    for i, name in enumerate(classes):
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
    out["classes"] = classes
    out["class_accuracy"] = out["class_correct"] / out["matched"] if out["matched"] else None
    out["detection_precision"] = out["matched"] / out["model_boxes"] if out["model_boxes"] else None
    return out


def stats_sessions(sid_arg):
    if sid_arg:
        return [sid_arg] if safe_session(sid_arg) else None
    ids = {p.name for p in UPLOAD_ROOT.iterdir() if p.is_dir()}
    ids |= {p.name for p in LABEL_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")}
    return sorted(ids)



# ------------------------------------------------------- session classes ----
def classes_file(sid):
    return LABEL_ROOT / sid / "_classes.json"


def session_classes(sid):
    try:
        return json.loads(classes_file(sid).read_text())
    except (OSError, ValueError):
        return list(CLASSES)


def save_session_classes(sid, names):
    p = classes_file(sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(names))


# ------------------------------------------------------- YOLO .txt import ---
def import_labels(sid, files):
    """files = [(filename, text)]. Handles YOLO .txt (one per image, `class cx cy w h`) and .csv."""
    images = {}
    for p in session_dir(sid).iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(p.stem.lower(), p)
    names = session_classes(sid)
    txts = [(n, t) for n, t in files if n.lower().endswith(".txt")]
    csvs = [(n, t) for n, t in files if n.lower().endswith(".csv")]
    # optional class-name list shipped with the labels (YOLO convention)
    for n, t in txts:
        if n.lower() in ("classes.txt", "obj.names", "names.txt"):
            lst = [x.strip() for x in t.splitlines() if x.strip()]
            if lst:
                names = lst
                save_session_classes(sid, names)
    res = {"matched_images": 0, "boxes": 0, "images_total": len(images), "unmatched_names": [],
           "unmatched_count": 0, "bad_rows": 0, "classes": names}
    for n, t in txts:
        if n.lower() in ("classes.txt", "obj.names", "names.txt"):
            continue
        img = images.get(Path(n).stem.lower())
        if img is None:
            res["unmatched_count"] += 1
            res["unmatched_names"].append(n)
            continue
        boxes = []
        for line in t.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
                if len(parts) < 5:
                    raise ValueError
            except ValueError:
                res["bad_rows"] += 1
                continue
            boxes.append({"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h})
        write_label(sid, img.name, boxes)
        update_meta(sid, img.name, reviewed=False, pred=boxes)
        res["matched_images"] += 1
        res["boxes"] += len(boxes)
    # class ids beyond the known list get placeholder names so the UI can show them
    top = max((b["class_id"] for n, t in txts for b in [{"class_id": int(float(l.split()[0]))}
               for l in t.splitlines() if l.split() and l.split()[0].lstrip("-").replace(".", "").isdigit()]), default=-1)
    while len(names) <= top:
        names.append(f"class {len(names)}")
    save_session_classes(sid, names)
    res["classes"] = names
    for n, t in csvs:
        r = import_csv(sid, t)
        for k in ("matched_images", "boxes", "unmatched_count", "bad_rows"):
            res[k] += r[k]
        res["unmatched_names"] += r["unmatched_names"]
        res["classes"] = r["classes"]
    res["unlabeled_images"] = sorted(p.name for p in set(images.values())
                                     if not label_path(sid, p.name).exists())
    if not txts and not csvs:
        raise ValueError("no .txt or .csv label files found")
    return res


# ------------------------------------------------------------- CSV import ---
_ALIASES = {
    "image": ("filename", "file", "image", "image_name", "image_id", "img", "path", "file_name", "name"),
    "cls": ("class", "class_name", "classname", "label", "category", "cls", "class_id", "classid"),
    "cx": ("cx", "x_center", "xc", "x_centre"),
    "cy": ("cy", "y_center", "yc", "y_centre"),
    "w": ("w", "width", "bbox_width", "box_width"),
    "h": ("h", "height", "bbox_height", "box_height"),
    "x0": ("xmin", "x_min", "x1", "left"),
    "y0": ("ymin", "y_min", "y1", "top"),
    "x1": ("xmax", "x_max", "x2", "right"),
    "y1": ("ymax", "y_max", "y2", "bottom"),
    "x": ("x",),
    "y": ("y",),
}


def _pick(fields, key):
    for a in _ALIASES[key]:
        if a in fields:
            return fields[a]
    return None


def _resolve_class(token, names):
    t = str(token).strip()
    if t.lstrip("-").isdigit() and 0 <= int(t) < len(names):
        return int(t)
    low = t.lower()
    for i, n in enumerate(names):
        if n.lower() == low:
            return i
    if not t:
        return -1
    names.append(t)  # a class that is not in the default list: add it to this session
    return len(names) - 1


def import_csv(sid, text):
    """Parse the label CSV and write YOLO labels for every image already uploaded to `sid`."""
    text = text.lstrip("\ufeff")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("the CSV file has no data rows")
    header = [h.strip().lower().replace(" ", "_") for h in rows[0]]

    images = {}
    for p in session_dir(sid).iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images[p.name.lower()] = p
            images.setdefault(p.stem.lower(), p)
    sizes = {}
    names = session_classes(sid)
    per_image, unmatched, bad = {}, set(), 0

    for r in rows[1:]:
        f = {header[i]: r[i].strip() for i in range(min(len(header), len(r)))}
        fn = _pick(f, "image")
        if not fn:
            raise ValueError("no image-name column found (expected one of: "
                             + ", ".join(_ALIASES["image"]) + ")")
        key = Path(fn.replace("\\", "/")).name.lower()
        img = images.get(key) or images.get(Path(key).stem)
        if img is None:
            unmatched.add(fn)
            continue
        try:
            cls = _resolve_class(_pick(f, "cls") or "", names)
            cx, cy, w, h = (_pick(f, k) for k in ("cx", "cy", "w", "h"))
            x0, y0, x1, y1 = (_pick(f, k) for k in ("x0", "y0", "x1", "y1"))
            x, y = _pick(f, "x"), _pick(f, "y")
            if img not in sizes:
                with Image.open(img) as im:
                    sizes[img] = im.size
            W, H = sizes[img]
            if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
                x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                if max(x0, y0, x1, y1) <= 1.0:      # already normalised corners
                    x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
                cx, cy, w, h = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H
            elif None not in (cx, cy, w, h):
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                if max(cx, cy, w, h) > 1.0:          # pixel centre/size
                    cx, w, cy, h = cx / W, w / W, cy / H, h / H
            elif None not in (x, y, w, h):
                x, y, w, h = float(x), float(y), float(w), float(h)
                if max(x, y, w, h) <= 1.0:
                    cx, cy = x + w / 2, y + h / 2
                else:
                    cx, cy, w, h = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
            else:
                raise ValueError("no box columns")
        except (ValueError, TypeError, OSError):
            bad += 1
            continue
        per_image.setdefault(img, []).append(
            {"class_id": cls, "cx": cx, "cy": cy, "w": w, "h": h})

    for img, boxes in per_image.items():
        write_label(sid, img.name, boxes)
        update_meta(sid, img.name, reviewed=False, pred=boxes)
    save_session_classes(sid, names)
    all_imgs = {p for p in images.values()}
    return {"matched_images": len(per_image), "boxes": sum(len(b) for b in per_image.values()),
            "images_total": len(all_imgs), "images_without_labels": len(all_imgs) - len(per_image),
            "unmatched_names": sorted(unmatched), "unmatched_count": len(unmatched),
            "bad_rows": bad, "classes": names}


# ------------------------------------------------------------------- routes --

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config")
def api_config():
    return jsonify({"classes": CLASSES, "defaults": {}, "models": {}, "default_model": "12"})


@app.route("/api/session_classes")
def api_session_classes():
    sid = safe_session(request.args.get("session_id", ""))
    if not sid:
        return jsonify({"error": "unknown session"}), 404
    return jsonify({"classes": session_classes(sid)})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept a batch of image files and start / extend a session."""
    sid = request.form.get("session_id") or str(uuid.uuid4())
    if not sid.replace("-", "").isalnum():
        return jsonify({"error": "bad session id"}), 400
    d = session_dir(sid)
    d.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in request.files.getlist("files"):
        name = Path(f.filename or "").name  # flatten - ignore sub-folder structure
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target = d / name
        if target.exists():
            target = d / f"{Path(name).stem}_{uuid.uuid4().hex[:6]}{Path(name).suffix}"
        f.save(target)
        saved.append(target.name)

    if not saved:
        return jsonify({"error": "no images found in the selected files"}), 400
    saved.sort()
    return jsonify({"session_id": sid, "images": saved})


@app.route("/api/upload_csv", methods=["POST"])
def api_upload_csv():
    """Attach the CSV label file to a session (after its images are uploaded)."""
    sid = safe_session(request.form.get("session_id", ""))
    fl = request.files.getlist("csv")
    if not sid or not fl:
        return jsonify({"error": "missing session or label files"}), 400
    try:
        files = [(Path(f.filename or "").name, f.read().decode("utf-8-sig", errors="replace")) for f in fl]
        return jsonify(import_labels(sid, files))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/images")
def api_images():
    sid = safe_session(request.args.get("session_id", ""))
    if not sid:
        return jsonify({"error": "unknown session"}), 404
    names = sorted(p.name for p in session_dir(sid).iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    labeled_stems = {p.stem for p in (LABEL_ROOT / sid).glob("*.txt")} if (LABEL_ROOT / sid).is_dir() else set()
    # "reviewed" = a human saved this image; labeled-but-not-reviewed means the
    # label is still raw model output waiting for a person to check it
    return jsonify([{"name": n,
                     "labeled": Path(n).stem in labeled_stems,
                     "reviewed": bool(read_meta(sid, n).get("reviewed", False))}
                    for n in names])


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


@app.route("/api/export.zip")
def api_export_zip():
    """All saved labels of a session as a zip (fallback when the browser has no folder picker)."""
    import io, zipfile
    sid = safe_session(request.args.get("session_id", ""))
    if not sid:
        return jsonify({"error": "not found"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((LABEL_ROOT / sid).glob("*.txt")):
            z.write(f, f.name)
        if request.args.get("crops"):
            names = session_classes(sid)
            for lf in sorted((LABEL_ROOT / sid).glob("*.txt")):
                img_p = next((q for q in (UPLOAD_ROOT / sid).iterdir() if q.stem == lf.stem), None) \
                    if (UPLOAD_ROOT / sid).exists() else None
                if img_p is None:
                    continue
                try:
                    im = Image.open(img_p).convert("RGB")
                except Exception:
                    continue
                W, H = im.size
                for k, b in enumerate(read_label(sid, lf.stem) or []):
                    w, h = b["w"] * W, b["h"] * H
                    x0, y0 = b["cx"] * W - w / 2, b["cy"] * H - h / 2
                    c = im.crop((int(x0), int(y0), int(x0 + w) + 1, int(y0 + h) + 1))
                    cn = names[b["class_id"]] if 0 <= b["class_id"] < len(names) else "Unknown"
                    cn = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cn)
                    bio = io.BytesIO(); c.save(bio, "PNG")
                    z.writestr(f"crops/{cn}/{lf.stem}_{k + 1}.png", bio.getvalue())
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="labels.zip")


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5080))
    print(f"open http://localhost:{port}")
    try:
        from waitress import serve
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)
    else:
        serve(app, host="0.0.0.0", port=port,
              threads=int(os.environ.get("WEB_THREADS", 8)),
              max_request_body_size=16 * 1024 ** 3,
              channel_timeout=600, cleanup_interval=60)

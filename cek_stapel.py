import sys
import time
import math
from collections import deque

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
# STAPEL_TWO_LEVELS_THRESH = 2.7
# ===== Config =====
MODEL_PATH   = "lastbaru.pt"     # ganti jika path model berbeda
CONF_THRESH  = 0.1               # disarankan > 0.01 untuk kurangi duplikasi
IOU_THRESH   = 0.45
CLASS_KEYWORD = "karung"         # ganti jika nama kelas di model berbeda
STAPEL_TWO_LEVELS_THRESH = 4.10  # m
TINGGI_KARUNG_CM = 30.0 #cm
TINGGI_TINGKAT_M = 0.35     # selisih antar tingkat sekitar 5 cm

# RealSense D435 resolutions (umum)
COLOR_W, COLOR_H, COLOR_FPS = 1920, 1080, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30

# Tracking & RMS
IOU_MATCH_THRESH = 0.2
TRACK_TTL_FRAMES = 30
RMS_WINDOW       = 50
MAX_BBOX_AREA_PX = 100_000

# === FILTER DIMENSI MINIMUM KARUNG (BAHAN BAKU) ===
MIN_BBOX_W_PX = 80
MIN_BBOX_H_PX = 40


# ---------- Util ----------
def median_depth(depth_frame_np, u, v, k=5):
    h, w = depth_frame_np.shape
    u0 = max(0, int(u - k//2)); v0 = max(0, int(v - k//2))
    u1 = min(w, int(u + k//2) + 1); v1 = min(h, int(v + k//2) + 1)
    patch = depth_frame_np[v0:v1, u0:u1]
    patch = patch[patch > 0]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))

def deproject_to_3d(intrinsics, u, v, depth_m):
    if depth_m <= 0:
        return None
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth_m)
    return np.array([X, Y, Z], dtype=np.float32)

def draw_text(img, text, org, scale=0.6, thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg:
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x, y - th - base), (x + tw, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def rms(values):
    arr = np.array(values, dtype=np.float32)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) > 0 else float("nan")

# === DEDUP ===
def _center_of(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _area_of(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)

def _contains(a, b, pad=0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (bx1 >= ax1 - pad and by1 >= ay1 - pad and
            bx2 <= ax2 + pad and by2 <= ay2 + pad)

def suppress_double_boxes(dets, iou_thresh=0.75, center_tol_px=12, area_sim_low=0.6, area_sim_high=1.4):
    dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)
    kept = []
    for d in dets_sorted:
        dup = False
        cx_d, cy_d = _center_of(d["xyxy"])
        area_d = max(1, _area_of(d["xyxy"]))
        for k in kept:
            if d["cls"] != k["cls"]:
                continue
            if iou(d["xyxy"], k["xyxy"]) >= iou_thresh:
                dup = True; break
            cx_k, cy_k = _center_of(k["xyxy"])
            if abs(cx_d - cx_k) <= center_tol_px and abs(cy_d - cy_k) <= center_tol_px:
                area_k = max(1, _area_of(k["xyxy"]))
                ratio = area_d / area_k
                if area_sim_low <= ratio <= area_sim_high:
                    dup = True; break
            if _contains(k["xyxy"], d["xyxy"], pad=2) or _contains(d["xyxy"], k["xyxy"], pad=2):
                dup = True; break
        if not dup:
            kept.append(d)
    return kept

# === Orientation & grouping ===
def bbox_orientation(x1, y1, x2, y2, ratio_thresh=1.05):
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    # print(h/w,w/h)
    if h / w >= ratio_thresh: return "portrait"
    elif w / h >= ratio_thresh: return "landscape"
    else: return "square"

def draw_group_box(img, group_dets, label):
    xs1, ys1, xs2, ys2 = [], [], [], []
    for d in group_dets:
        x1, y1, x2, y2 = d["xyxy"]
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    x1g, y1g, x2g, y2g = min(xs1), min(ys1), max(xs2), max(ys2)
    cv2.rectangle(img, (x1g, y1g), (x2g, y2g), (0, 140, 255), 3)
    draw_text(img, label, (x1g, max(0, y1g - 10)), scale=0.7, thickness=2, bg=True)

def find_row_sequences(dets, k=3, orientation="portrait", y_tol_frac=0.25, max_gap_frac=1.6, min_overlap_frac=-0.2):
    cand = []
    # print(orientation)
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        if bbox_orientation(x1, y1, x2, y2) != orientation:
            continue
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        w  = max(1, x2 - x1); h  = max(1, y2 - y1)
        cand.append({**d, "cx": cx, "cy": cy, "w": w, "h": h})
    # print(len(cand))
    if len(cand) < k: return []
    cand.sort(key=lambda d: d["cx"])
    used, groups, i = set(), [], 0
    while i <= len(cand) - k:
        seq = [cand[i]]; avg_h = cand[i]["h"]; avg_w = cand[i]["w"]
        for j in range(i + 1, len(cand)):
            if len(seq) == k: break
            dprev = seq[-1]; dcur  = cand[j]
            tmp_h = (avg_h * len(seq) + dcur["h"]) / (len(seq) + 1)
            tmp_w = (avg_w * len(seq) + dcur["w"]) / (len(seq) + 1)
            y_tol = y_tol_frac * tmp_h
            
            if abs(dcur["cy"] - seq[0]["cy"]) > y_tol: continue
            gap = dcur["cx"] - dprev["cx"]
            overlap = (min(dcur["cx"] + dcur["w"]/2, dprev["cx"] + dprev["w"]/2)
                       - max(dcur["cx"] - dcur["w"]/2, dprev["cx"] - dprev["w"]/2))
            overlap_frac = overlap / max(tmp_w, 1.0)
            # find_row_sequences(enriched, k=3, orientation="portrait",
            #                                 y_tol_frac=0.25, max_gap_frac=30.0,  min_overlap_frac=-0.5)
            
            if gap < 0: continue
            if gap > max_gap_frac * tmp_w: continue
            if overlap_frac < min_overlap_frac: continue
            seq.append(dcur); avg_h, avg_w = tmp_h, tmp_w
        
        if len(seq) == k:
            seq_ids = tuple(id(s) for s in seq)
            if not any(sid in used for sid in seq_ids):
                groups.append(seq); used.update(seq_ids); i += k; continue
        i += 1
        # print(groups)
    return groups

# === STAPEL (duo+trio vertical) ===
def group_bbox(group):
    xs1, ys1, xs2, ys2 = [], [], [], []
    for d in group:
        x1, y1, x2, y2 = d["xyxy"]
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    return (min(xs1), min(ys1), max(xs2), max(ys2))

def _width_h(b):
    return max(1, b[2]-b[0]), max(1, b[3]-b[1])

def _x_overlap_frac(b1, b2):
    left  = max(b1[0], b2[0]); right = min(b1[2], b2[2])
    overlap = max(0, right - left)
    w1, _ = _width_h(b1); w2, _ = _width_h(b2)
    return overlap / max(1, min(w1, w2))

def _vertical_gap(b_top, b_bottom):
    return max(0, b_bottom[1] - b_top[3])

def find_stacks(duos, trios):
    """
    STAPEL terbentuk hanya jika DUO & TRIO nempel secara vertikal (gap ~ 0) atau sedikit overlap.
    One-to-one matching: 1 DUO ↔ 1 TRIO, tidak boleh dipakai ganda.
    """
    # --- parameter bisa kamu sesuaikan ---
    min_x_overlap_frac = 0.1   # seberapa segaris di sumbu X (0.2 terlalu longgar di banyak kasus)
    touch_tol_px        = 25      # toleransi 'nempel' (0–2 px)
    overlap_tol_px      = 160      # toleransi overlap (berapa piksel saling masuk masih dianggap valid)

    candidates = []

    def _gbb_wh_cy(grp):
        b = group_bbox(grp)                  # (x1,y1,x2,y2)
        w = max(1, b[2]-b[0]); h = max(1, b[3]-b[1])
        cy = (b[1] + b[3]) / 2.0
        return b, w, h, cy

    duos_info  = [(d, *_gbb_wh_cy(d)) for d in duos]   # (grp, bbox, w, h, cy)
    trios_info = [(t, *_gbb_wh_cy(t)) for t in trios]

    # helper: raw gap (bisa negatif kalau overlap)
    def _vertical_gap_raw(b_top, b_bottom):
        return b_bottom[1] - b_top[3]  # tanpa clamp; negatif = overlap

    for d, bd, _, hd, cyd in duos_info:
        for t, bt, _, ht, cyt in trios_info:
            # 1) Harus overlap sumbu-X cukup
            x_overlap = _x_overlap_frac(bd, bt)
            
            if x_overlap < min_x_overlap_frac:
                continue

            # 2) Tentukan siapa di atas/bawah pakai center y
           
            if cyd < cyt:
                # DUO di atas, TRIO di bawah
                raw_gap = _vertical_gap_raw(bd, bt)
                # Valid jika nempel (≈0) atau sedikit overlap
                # print(raw_gap,touch_tol_px)
                # print(overlap_tol_px,raw_gap,touch_tol_px)
                if -overlap_tol_px <= raw_gap <= touch_tol_px:
                    candidates.append((abs(raw_gap), "duo_top", d, t))
            else:
                # TRIO di atas, DUO di bawah
                raw_gap = _vertical_gap_raw(bt, bd)
                # print(overlap_tol_px,raw_gap,touch_tol_px)
                if -overlap_tol_px <= raw_gap <= touch_tol_px:
                    candidates.append((abs(raw_gap), "trio_top", d, t))

    # Pilih pasangan paling rapat dulu, one-to-one
    candidates.sort(key=lambda x: x[0])
    used_duo, used_trio, stacks = set(), set(), []
    for gap_abs, mode, d, t in candidates:
        if id(d) in used_duo or id(t) in used_trio:
            continue
        stacks.append((d, t, mode))
        used_duo.add(id(d)); used_trio.add(id(t))

    return stacks



# ---------- Tracker ----------
# ---------- Tracker ----------
class Track:
    _next_id = 1

    def __init__(self, bbox, cls, name):
        # assign unique id
        self.id = Track._next_id
        Track._next_id += 1

        # basic info
        self.bbox = tuple(bbox)
        self.cls  = cls
        self.name = name

        # tracking meta
        self.last_seen = 0

        # metric buffers
        self.radial_buf   = deque(maxlen=RMS_WINDOW)
        self.center3d_buf = deque(maxlen=RMS_WINDOW)
        self.tinggi_buf   = deque(maxlen=RMS_WINDOW)

    def update_bbox(self, bbox, frame_idx):
        self.bbox = tuple(bbox)
        self.last_seen = frame_idx

    def update_metrics(self, radial, center3d, tinggi, frame_idx):
        # append only valid numeric values
        try:
            if not (math.isnan(radial) or math.isinf(radial)):
                self.radial_buf.append(float(radial))
        except Exception:
            pass
        try:
            if not (math.isnan(center3d) or math.isinf(center3d)):
                self.center3d_buf.append(float(center3d))
        except Exception:
            pass
        try:
            if not (math.isnan(tinggi) or math.isinf(tinggi)):
                self.tinggi_buf.append(float(tinggi))
        except Exception:
            pass
        self.last_seen = frame_idx

    def rms_triplet(self):
        return rms(list(self.radial_buf)), rms(list(self.center3d_buf)), rms(list(self.tinggi_buf))


class Tracker:
    def __init__(self):
        self.tracks = []

    def match_and_update(self, detections, frame_idx):
        unmatched = list(range(len(detections)))
        matched_pairs = []

        # try match existing tracks by IoU
        for tr in list(self.tracks):
            best_j = -1
            best_iou = 0.0
            for j in unmatched:
                iou_val = iou(tr.bbox, detections[j]["xyxy"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j != -1 and best_iou >= IOU_MATCH_THRESH:
                det = detections[best_j]
                tr.update_bbox(tuple(map(int, det["xyxy"])), frame_idx)
                r, c3d, t = det.get("metrics", (float("nan"), float("nan"), float("nan")))
                tr.update_metrics(r, c3d, t, frame_idx)
                matched_pairs.append((tr, det))
                unmatched.remove(best_j)

        # create new tracks for unmatched detections
        new_tracks = []
        for j in unmatched:
            det = detections[j]
            tr = Track(tuple(map(int, det["xyxy"])), det.get("cls"), det.get("name"))
            r, c3d, t = det.get("metrics", (float("nan"), float("nan"), float("nan")))
            tr.update_metrics(r, c3d, t, frame_idx)
            tr.last_seen = frame_idx
            self.tracks.append(tr)
            new_tracks.append(tr)
            matched_pairs.append((tr, det))

        # expire old tracks
        self.tracks = [t for t in self.tracks if (frame_idx - t.last_seen) <= TRACK_TTL_FRAMES]

        return matched_pairs, new_tracks


# ---------- Main ----------
def main():
    # Load YOLO
    try:
        model = YOLO(MODEL_PATH)
        names = getattr(model, "names", None)
        if names is None and hasattr(model, "model") and hasattr(model.model, "names"):
            names = model.model.names
        print("[INFO] Loaded Ultralytics YOLO.")
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}"); sys.exit(1)

    karung_idxs = []
    if names is not None:
        for i, n in enumerate(names):
            if CLASS_KEYWORD.lower() in str(n).lower():
                karung_idxs.append(i)
        if karung_idxs: print(f"[INFO] Filter kelas mengandung '{CLASS_KEYWORD}': {karung_idxs}")
        else:           print(f"[WARN] Tidak ada nama kelas mengandung '{CLASS_KEYWORD}'. Semua deteksi dipakai.")

    # RealSense
    pipeline = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    profile = pipeline.start(config); align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # === Adaptasi Kondisi Cahaya (Auto Exposure Dinamis) ===
    try:
        # Ambil sensor warna (RGB)
        color_sensor = profile.get_device().query_sensors()[1]  # 0 = depth, 1 = color
        
        # Aktifkan auto exposure agar kamera menyesuaikan otomatis
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        # Batasi exposure agar tidak terlalu ekstrem (anti flicker)
        if depth_sensor.supports(rs.option.auto_exposure_limit):
            depth_sensor.set_option(rs.option.auto_exposure_limit, 15000)  # μs
        if depth_sensor.supports(rs.option.auto_gain_limit):
            depth_sensor.set_option(rs.option.auto_gain_limit, 64)

        if color_sensor.supports(rs.option.auto_exposure_limit):
            color_sensor.set_option(rs.option.auto_exposure_limit, 200)  # nilai kecil = lebih cepat respon
        if color_sensor.supports(rs.option.auto_gain_limit):
            color_sensor.set_option(rs.option.auto_gain_limit, 128)

        # Percepat reaksi auto exposure (prioritas tinggi)
        if depth_sensor.supports(rs.option.auto_exposure_priority):
            depth_sensor.set_option(rs.option.auto_exposure_priority, 1.0)
        if color_sensor.supports(rs.option.auto_exposure_priority):
            color_sensor.set_option(rs.option.auto_exposure_priority, 1.0)

        # Aktifkan IR emitter untuk bantu depth di kondisi gelap
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

        print("[INFO] Auto exposure adaptif aktif ✓")

    except Exception as e:
        print(f"[WARN] Auto exposure adaptif tidak tersedia: {e}")

    print(f"[INFO] Depth scale: {depth_scale} m/unit")

    hole_filling = rs.hole_filling_filter(); intr = None

    fps_q = deque(maxlen=20); last = time.time(); frame_idx = 0
    tracker = Tracker()

    try:
        while True:
            frame_idx += 1
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame: continue
            depth_frame = hole_filling.process(depth_frame)
            if intr is None:
                intr = color_frame.profile.as_video_stream_profile().intrinsics

            depth_raw = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())
            H, W = color_img.shape[:2]; cx, cy = W // 2, H // 2
            depth_m = depth_raw.astype(np.float32) * depth_scale

            # Inference (native res)
            results = model.predict(color_img, imgsz=1920, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

            detections = []
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None: continue
                rnames = r.names if hasattr(r, "names") and isinstance(r.names, dict) else None
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0]); cls  = int(box.cls[0])
                    name = rnames.get(cls, str(cls)) if rnames is not None else (names[cls] if names is not None else str(cls))
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
                    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
                    if x2 <= x1 or y2 <= y1: continue
                    # === FILTER DIMENSI MINIMUM (SEBELUM JADI SINGLE) ===
                    w = x2 - x1
                    h = y2 - y1
                    if w < MIN_BBOX_W_PX or h < MIN_BBOX_H_PX:
                        continue

                    # hitung area dan FILTER
                    # area_px = (x2 - x1) * (y2 - y1)
                    # # if not (50000 <= area_px <= 100000):
                    # #     continue
                    # u = int((x1 + x2) / 2)
                    # v = int((y1 + y2) / 2)

                    # ================== FILTER DINAMIS BERDASARKAN DEPTH ==================

                    area_px = (x2 - x1) * (y2 - y1)
                    u = int((x1 + x2) / 2)
                    v = int((y1 + y2) / 2)

                    # Ambil depth rata-rata objek
                    d_obj = median_depth(depth_m, u, v, k=7)
                    if d_obj <= 0:
                        continue

                    fx, fy = intr.fx, intr.fy

                    # Ukuran karung fisik (rentang dimensi dalam meter)
                    KARUNG_W_MIN, KARUNG_W_MAX = 0.50, 0.60  # panjang
                    KARUNG_H_MIN, KARUNG_H_MAX = 0.30, 0.40  # lebar

                    # Hitung area karung dalam piksel sesuai jarak
                    expected_w_min_px = (fx * KARUNG_W_MIN) / d_obj
                    expected_w_max_px = (fx * KARUNG_W_MAX) / d_obj
                    expected_h_min_px = (fy * KARUNG_H_MIN) / d_obj
                    expected_h_max_px = (fy * KARUNG_H_MAX) / d_obj

                    expected_area_min = expected_w_min_px * expected_h_min_px
                    expected_area_max = expected_w_max_px * expected_h_max_px
                    min_area = expected_area_min * 0.7
                    max_area = expected_area_max * 1.3

                    # Variasi depth (cek tekstur permukaan)
                    patch = depth_m[max(0, v-7):min(H, v+7), max(0, u-7):min(W, u+7)]
                    nonzero_patch = patch[patch > 0]
                    depth_var = np.var(nonzero_patch) if len(nonzero_patch) > 0 else 0

                    # Cek selisih kedalaman terhadap sekitar (gradient)
                    d_left  = median_depth(depth_m, max(u-30,0), v, k=5)
                    d_right = median_depth(depth_m, min(u+30,W-1), v, k=5)
                    d_top   = median_depth(depth_m, u, max(v-30,0), k=5)
                    d_bottom= median_depth(depth_m, u, min(v+30,H-1), k=5)
                    depth_gradient = max(abs(d_obj - d_left), abs(d_obj - d_right),
                                        abs(d_obj - d_top), abs(d_obj - d_bottom))
                    # --- Stabilizer antar frame (kurangi noise depth) ---
                    if 'prev_grad' in locals():
                        if abs(depth_gradient - prev_grad) < 0.15:
                            depth_gradient = (depth_gradient + prev_grad) / 2
                        if abs(depth_var - prev_var) < 0.0002:
                            depth_var = (depth_var + prev_var) / 2
                    prev_grad, prev_var = depth_gradient, depth_var

                    # === Debug (boleh matikan setelah yakin) ===
                    # print(f"[DEBUG-FILT] depth={d_obj:.2f}m | grad={depth_gradient:.3f}m | var={depth_var:.6f} | area={area_px:.0f}")

                    # if depth_var > 0.02 or depth_gradient > 1.0:
                        # print(f"[SKIP NOISY] grad={depth_gradient:.3f} var={depth_var:.6f}")
                    # elif depth_var < 0.00002 or depth_gradient < 0.05:
                        # print(f"[SKIP FLAT] grad={depth_gradient:.3f} var={depth_var:.6f}")

                  # === Filter adaptif versi dinamis (v5.0) ===

                    # Batasi rentang depth efektif kamera
                    if not (0.3 <= d_obj <= 3.1):
                        continue

                    # Hitung faktor jarak (supaya filter makin toleran kalau jauh)
                    depth_factor = np.clip((d_obj - 1.5) / (3.1 - 1.5), 0, 1)  # 0–1
                    grad_upper = 2.0 + 0.8 * depth_factor   # jauh => toleransi naik ke 1.6
                    grad_lower = 0.01 - 0.005 * depth_factor # jauh => toleransi turun ke 0.02
                    var_upper  = 0.10 + 0.08 * depth_factor # jauh => bisa sampai 0.1
                    var_lower  = 0.000005

                    # --- Stabilizer antar frame ---
                    if 'prev_grad' in locals():
                        if abs(depth_gradient - prev_grad) < 0.15:
                            depth_gradient = (depth_gradient + prev_grad) / 2
                        if abs(depth_var - prev_var) < 0.0002:
                            depth_var = (depth_var + prev_var) / 2
                    prev_grad, prev_var = depth_gradient, depth_var

                    # --- Filter berdasarkan hasil stabilisasi ---
                    if depth_var < var_lower or depth_gradient < grad_lower:
                        print(f"[SKIP FLAT] grad={depth_gradient:.3f} var={depth_var:.6f}")
                        continue
                    if depth_var > var_upper or depth_gradient > grad_upper:
                        print(f"[SKIP NOISY] grad={depth_gradient:.3f} var={depth_var:.6f}")
                        continue

                    # --- Filter area adaptif ---
                    if not (min_area * 0.9 <= area_px <= max_area * 2.0):
                        continue


                    # --- Filter glare (area terlalu terang) ---
                    roi = color_img[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray_roi)

                    if brightness > 200 or brightness < 30:
                        print(f"[SKIP LIGHT] brightness={brightness:.1f}")
                        continue




                    # ================== AKHIR FILTER DINAMIS ==================

                    detections.append({"xyxy": (x1, y1, x2, y2), "conf": conf, "cls": cls, "name": name, "bbox_area_px": area_px})

            # Filter kelas + dedup
            if karung_idxs:
                detections = [d for d in detections if int(d["cls"]) in karung_idxs]
            detections.sort(key=lambda d: d["conf"], reverse=True)
            detections = suppress_double_boxes(detections, iou_thresh=0.75, center_tol_px=12, area_sim_low=0.6, area_sim_high=1.4)

            # Titik 3D center frame
            center_depth = median_depth(depth_m, cx, cy, k=7)
            P_center = deproject_to_3d(intr, cx, cy, center_depth)

            # Build enriched dets
            enriched = []
            for idx_d, d in enumerate(detections):
                x1, y1, x2, y2 = d["xyxy"]
                u = int((x1 + x2) / 2.0); v = int((y1 + y2) / 2.0)
                d_obj = median_depth(depth_m, u, v, k=7)
                P_obj = deproject_to_3d(intr, u, v, d_obj)
                radial = float(np.linalg.norm(P_obj)) if P_obj is not None else float("nan")
                center3D = float(np.linalg.norm(P_obj - P_center)) if (P_obj is not None and P_center is not None) else float("nan")
                tinggi_pitagoras = math.sqrt(max(radial*2 - center3D*2, 0.0)) if (not (math.isnan(radial) or math.isnan(center3D)) and radial >= center3D) else float("nan")
                area_px = max(0, (x2 - x1) * (y2 - y1))
                d_en = dict(d)
                d_en.update({
                    "center_px": (u, v),
                    "P_obj": P_obj,
                    "metrics": (radial, center3D, tinggi_pitagoras),
                    "bbox_area_px": area_px,
                    "det_id": idx_d,
                    "orientation": bbox_orientation(x1, y1, x2, y2)
                })
                enriched.append(d_en)
                       # === Ambil tinggi maksimum di frame ini (setelah semua deteksi selesai) ===
            tinggi_list = [d["metrics"][2] for d in enriched if not math.isnan(d["metrics"][2])]
            max_tinggi_frame = max(tinggi_list) if tinggi_list else float("nan")



            # === Tracker dulu, agar bisa ambil RMS per objek ===
            matched_pairs, _ = tracker.match_and_update(enriched, frame_idx)
            detid_to_track = { det["det_id"]: tr for (tr, det) in matched_pairs }

            # === Cari DUO & TRIO (berjejer) ===
            # === Cari DUO & TRIO (berjejer) ===
            trio_portrait = find_row_sequences(enriched, k=3, orientation="portrait",
                                            y_tol_frac=0.25, max_gap_frac=130.0,  min_overlap_frac=-20)
            duo_landscape = find_row_sequences(enriched, k=2, orientation="landscape",
                                            y_tol_frac=0.25, max_gap_frac=130.0, min_overlap_frac=-20)

            # === STAPEL: pasangan 1 DUO + 1 TRIO (one-to-one, vertikal); 
            #     jika terbentuk STAPEL, bbox DUO/TRIO yang terlibat TIDAK digambar. ===
            stacks = find_stacks(duo_landscape, trio_portrait)

            # tandai anggota duo/trio yang terpasang stapel agar tidak digambar sebagai single
            suppressed_det_ids = set()
            stacked_duo_ptrs  = set(id(duo_grp)  for (duo_grp,  trio_grp, mode) in stacks)
            stacked_trio_ptrs = set(id(trio_grp) for (duo_grp,  trio_grp, mode) in stacks)

            # === STAPEL: hitung & gambar (5 atau 10 karung per stapel) ===
            stapel_counts = []      # simpan kontribusi karung dari setiap STAPEL (5 atau 10)
            count_stapel_10 = 0     # hanya untuk tampilan ringkas
            count_stapel_5  = 0

            def group_avg_rms(grp):
                rms_t_list, rms_c_list = [], []
                for d in grp:
                    tr = detid_to_track.get(d["det_id"])
                    if tr is not None:
                        _, rms_c, rms_t = tr.rms_triplet()
                    else:
                        _, c_now, t_now = d["metrics"]
                        rms_c, rms_t = c_now, t_now
                    if not (math.isnan(rms_t) or math.isinf(rms_t)): rms_t_list.append(rms_t)
                    if not (math.isnan(rms_c) or math.isinf(rms_c)): rms_c_list.append(rms_c)
                avg_t = float(np.mean(rms_t_list)) if rms_t_list else float("nan")
                avg_c = float(np.mean(rms_c_list)) if rms_c_list else float("nan")
                return avg_t, avg_c

            # suppress det anggota stapel lebih dulu supaya tidak digambar individual
            for duo_grp, trio_grp, mode in stacks:
                for d in (duo_grp + trio_grp):
                    suppressed_det_ids.add(d["det_id"])

            # gambar + hitung karung stapel (5/10) sekali saja
            for idx, (duo_grp, trio_grp, mode) in enumerate(stacks, start=1):
                # safety: pastikan 2 + 3 = 5 item unik
                unique_ids = {d["det_id"] for d in (duo_grp + trio_grp)}
                if len(duo_grp) != 2 or len(trio_grp) != 3 or len(unique_ids) != 5:
                    continue

                t_duo, c_duo   = group_avg_rms(duo_grp)
                t_trio, c_trio = group_avg_rms(trio_grp)
                t_avg_all = np.nanmean([t_duo, t_trio])   # tinggi STAPEL (gabungan)
                merged = duo_grp + trio_grp

                # === Tentukan jumlah tingkat berdasarkan tinggi STAPEL ===
                # STAPEL_TWO_LEVELS_THRESH = 2.2   # meter, jarak kamera ke lantai
                # TINGGI_KARUNG_CM = 11.0
                tinggi_karung_m = TINGGI_KARUNG_CM / 100

                # Hitung tinggi aktual tumpukan
                tinggi_stapel = STAPEL_TWO_LEVELS_THRESH - t_avg_all  # makin kecil t_avg_all → makin tinggi

                # Konversi ke jumlah tingkat
                jumlah_tingkat = int(round(tinggi_stapel / tinggi_karung_m))
                jumlah_tingkat = max(1, min(15, jumlah_tingkat))  # batasi 1–15 tingkat

                # Hitung total karung per STAPEL
                karung_stapel = jumlah_tingkat * 5
                tier_label = f"x{karung_stapel} ({jumlah_tingkat} tingkat)"

                # Simpan statistik berdasarkan tingkat
                if jumlah_tingkat == 1:
                    count_stapel_5 += 1
                elif jumlah_tingkat == 2:
                    count_stapel_10 += 1
                # kamu bisa tambahkan counter lain jika mau (misal count_stapel_15 untuk 3 tingkat, dst.)

                stapel_counts.append(karung_stapel)
                # label_mode = "DUO di Atas" if mode == "duo_top" else "TRIO di Atas"
                # draw_group_box(
                #     color_img, merged,
                #     f"STAPEL #{idx} ({label_mode}) | tinggi={t_avg_all:.2f}m | {tier_label}"
                # )


            # 2) (Tetap) gambar TRIO/DUO yang tidak ikut STAPEL + suppress
            for grp in trio_portrait:
                if id(grp) in {id(t) for (_, t, _) in stacks}:
                    continue
                avg_t, avg_c = group_avg_rms(grp)
                draw_group_box(color_img, grp, f"TRIO PORTRAIT | tinggi={avg_t:.2f}m")
                for d in grp: suppressed_det_ids.add(d["det_id"])

            for grp in duo_landscape:
                if id(grp) in {id(d) for (d, _, _) in stacks}:
                    continue
                avg_t, avg_c = group_avg_rms(grp)
                draw_group_box(color_img, grp, f"DUO LANDSCAPE | tinggi={avg_t:.2f}m")
                for d in grp: suppressed_det_ids.add(d["det_id"])


            # === STAPEL: pasangan DUO+TRIO (opsional visual tambahan) ===
            # === STAPEL: pasangan 1 DUO + 1 TRIO (one-to-one, vertikal) ===
            # stacks = find_stacks(duo_landscape, trio_portrait)
            # print(stacks)
            # tandai anggota duo/trio yang terpasang stapel agar tidak digambar sebagai single
            for (duo_grp, trio_grp, mode) in stacks:
                
                for d in (duo_grp + trio_grp):
                    suppressed_det_ids.add(d["det_id"])

           # gambar kotak gabungan & info ringkas
            for idx, (duo_grp, trio_grp, mode) in enumerate(stacks, start=1):
                merged = duo_grp + trio_grp

                # hitung rata-rata RMS tinggi & jarak (gabungan dua grup)
                def group_avg_rms(grp):
                    rms_t_list, rms_c_list = [], []
                    for d in grp:
                        tr = detid_to_track.get(d["det_id"])
                        if tr is not None:
                            _, rms_c, rms_t = tr.rms_triplet()
                        else:
                            _, c_now, t_now = d["metrics"]
                            rms_c, rms_t = c_now, t_now
                        if not (math.isnan(rms_t) or math.isinf(rms_t)): rms_t_list.append(rms_t)
                        if not (math.isnan(rms_c) or math.isinf(rms_c)): rms_c_list.append(rms_c)
                    avg_t = float(np.mean(rms_t_list)) if rms_t_list else float("nan")
                    avg_c = float(np.mean(rms_c_list)) if rms_c_list else float("nan")
                    return avg_t, avg_c

                # === Hitung rata-rata tinggi STAPEL (gabungan DUO + TRIO) ===
                t_duo, c_duo   = group_avg_rms(duo_grp)
                t_trio, c_trio = group_avg_rms(trio_grp)
                t_avg_all = np.nanmean([t_duo, t_trio])
                c_avg_all = np.nanmean([c_duo, c_trio])

                # === Tentukan jumlah tingkat (1–5) berdasarkan jarak kamera ===
                # STAPEL_TWO_LEVELS_THRESH = 2.2   # m, asumsi jarak kamera ke lantai
                # TINGGI_KARUNG_CM = 11.0
                # tinggi_karung_m = TINGGI_KARUNG_CM / 100

                # hitung selisih antara lantai dan puncak karung
                tinggi_stapel_m = STAPEL_TWO_LEVELS_THRESH - t_avg_all

                jumlah_tingkat = int(round(tinggi_stapel_m / tinggi_karung_m))
                jumlah_tingkat = max(1, min(15, jumlah_tingkat))  # batasi 1–5 tingkat

                # total karung per stapel
                # karung_stapel = jumlah_tingkat * 5
                # tier_label = f"x{karung_stapel} ({jumlah_tingkat} tingkat)"

                # label posisi (TRIO di atas atau DUO di atas)
                label_mode = "DUO di Atas" if mode == "duo_top" else "TRIO di Atas"

                # tampilkan kotak stapel
                draw_group_box(
                    color_img, merged,
                    f"STAPEL #{idx} ({label_mode}) | tinggi={t_avg_all:.2f}m"
                )



            # === Gambar deteksi individual (kecuali yang merupakan anggota DUO/TRIO) ===
            # === Gambar deteksi individual (kecuali yang merupakan anggota DUO/TRIO) ===
            for tr, det in matched_pairs:
                if det["det_id"] in suppressed_det_ids:
                    continue  # hilangkan bbox satuan jika sudah berpasangan (duo/trio)

                x1, y1, x2, y2 = map(int, det["xyxy"])
                u, v = det["center_px"]

                # kotak + titik tengah
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_img, (u, v), 5, (0, 255, 255), -1)

                # tinggi RMS + luas bbox (pakai nilai yang sudah dihitung di enriched)
                _, _, rms_t = tr.rms_triplet()
                area_px = int(det.get("bbox_area_px", (x2 - x1) * (y2 - y1)))

                # tampilkan satu baris
                draw_text(color_img, f"tinggi={rms_t:.2f}m", (x1, y2 + 18))


            # Header info
            now = time.time(); fps = 1.0 / max(1e-6, (now - last)); last = now
            fps_q.append(fps); fps_smoothed = sum(fps_q) / len(fps_q)
                        
                  # Fungsi bantu untuk menghitung rata-rata tinggi (rms_t) dari satu grup
            def group_avg_tinggi(grp):
                rms_t_list = []
                for d in grp:
                    tr = detid_to_track.get(d["det_id"])
                    if tr is not None:
                        _, _, rms_t = tr.rms_triplet()
                    else:
                        _, _, rms_t = d["metrics"]
                    if not (math.isnan(rms_t) or math.isinf(rms_t)):
                        rms_t_list.append(rms_t)
                return float(np.mean(rms_t_list)) if rms_t_list else float("nan")
            
            def hitung_total_karung_dinamis(jumlah_tingkat, jumlah_karung_terdeteksi):
                """
                Setiap tingkat penuh = 20 karung
                Tingkat paling atas = karung terdeteksi YOLO
                """
                if jumlah_tingkat <= 1:
                    return jumlah_karung_terdeteksi

                tingkat_penuh = jumlah_tingkat - 1
                total = tingkat_penuh * 20 + jumlah_karung_terdeteksi
                return total


      
                        # === HITUNG JUMLAH KARUNG ===
            # === HITUNG JUMLAH TOTAL KARUNG (tanpa duplikasi) ===

            # Hitung total dari STAPEL yang sudah tervalidasi
            total_stapel = sum(stapel_counts)

            # Hitung karung sisa (non-stapel) tanpa bonus dulu
            base_total_non_stapel = (
                len([g for g in trio_portrait if id(g) not in {id(t) for (_, t, _) in stacks}]) * 3 +
                len([g for g in duo_landscape if id(g) not in {id(d) for (d, _, _) in stacks}]) * 2 +
                len([d for d in enriched if d["det_id"] not in suppressed_det_ids])
            )

            # Hitung total karung awal (tanpa bonus)
            total_karung = total_stapel + base_total_non_stapel

            # Bonus hanya dihitung untuk non-stapel yang tinggi < 2.65 (opsional)
            bonus_non_stapel = 0
            for grp in trio_portrait:
                if id(grp) in {id(t) for (_, t, _) in stacks}: continue
                t_avg = group_avg_tinggi(grp)
                if not math.isnan(t_avg) and t_avg < STAPEL_TWO_LEVELS_THRESH:
                    bonus_non_stapel += 3
            for grp in duo_landscape:
                if id(grp) in {id(d) for (d, _, _) in stacks}: continue
                t_avg = group_avg_tinggi(grp)
                if not math.isnan(t_avg) and t_avg < STAPEL_TWO_LEVELS_THRESH:
                    bonus_non_stapel += 2
            for d in enriched:
                if d["det_id"] in suppressed_det_ids: continue
                tr = detid_to_track.get(d["det_id"])
                if tr is not None:
                    _, _, t_single = tr.rms_triplet()
                else:
                    _, _, t_single = d["metrics"]
                if not (math.isnan(t_single) or math.isinf(t_single)) and t_single < STAPEL_TWO_LEVELS_THRESH:
                    bonus_non_stapel += 1

            # --- Hitung tinggi tumpukan berdasarkan nilai depth terbesar (maks) ---
            if not math.isnan(max_tinggi_frame):
                # Data empiris (dari hasil pengamatan kamu)
                
                TOLERANSI_M = 0.025         # toleransi 2.5 cm agar tidak loncat tingkat

                delta_tinggi = STAPEL_TWO_LEVELS_THRESH - max_tinggi_frame
                jumlah_tingkat_auto = int((delta_tinggi + TOLERANSI_M) / TINGGI_TINGKAT_M)
                jumlah_tingkat_auto = max(1, min(15, jumlah_tingkat_auto))  # batasi sampai 15 tingkat

                # hitung total dinamis: tingkat penuh * 5 + jumlah deteksi di tingkat atas
                jumlah_karung_terdeteksi = len(enriched)
                total_karung = hitung_total_karung_dinamis(
                    jumlah_tingkat_auto,
                    jumlah_karung_terdeteksi
                )

                draw_text(color_img, f"Total karung: {total_karung} (tingkat={jumlah_tingkat_auto}, deteksi={jumlah_karung_terdeteksi})", (12, 72))
            else:
                draw_text(color_img, f"Total karung: {total_karung} (bonus={bonus_non_stapel})", (12, 52))



            # Center marker
            cv2.drawMarker(color_img, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # Tampilkan
            display_img = cv2.resize(color_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow("Karung Detector: DUO/TRIO RMS + STAPEL", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                ts = int(time.time()); cv2.imwrite(f"karung_{ts}.png", color_img)
                print(f"[INFO] Saved karung_{ts}.png")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

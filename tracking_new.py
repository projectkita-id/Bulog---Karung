import cv2
import numpy as np
import pyrealsense2 as rs
import time
from ultralytics import YOLO

# ================== KONFIGURASI ==================
MODEL_PATH = "bestbaru.pt"
IMGSZ = 1920
CONF_THRES = 0.5
DEVICE = "cuda"

COLOR_WIDTH, COLOR_HEIGHT, FPS = 1920, 1080, 30
DEPTH_WIDTH, DEPTH_HEIGHT = 1280, 720
SCALE = 0.6

# Posisi garis vertikal (sumbu X)
keluarA_line_x = 1000  # Gate OUT A
keluarB_line_x = 1000  # Gate OUT B

# Panjang garis (fraksi dari tinggi frame, 0.0 = atas, 1.0 = bawah)
A_y1_frac = 0.00   # mulai garis A
A_y2_frac = 0.40   # akhir garis A

B_y1_frac = 0.60   # mulai garis B
B_y2_frac = 1.00   # akhir garis B


# ================== MAIN ==================
def main():
    # Load YOLO model dengan tracking
    model = YOLO(MODEL_PATH)

    last_gate = {}          # status terakhir per ID
    previous_positions = {} # posisi terakhir per ID
    total_A, total_B = 0, 0

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    t_last = time.time()
    print("[INFO] Tekan 'q' untuk keluar...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            h, w = frame.shape[:2]
            annotated = frame.copy()

            # Hitung koordinat Y garis berdasarkan tinggi frame
            A_y1 = int(h * A_y1_frac)
            A_y2 = int(h * A_y2_frac)
            B_y1 = int(h * B_y1_frac)
            B_y2 = int(h * B_y2_frac)

            # Deteksi & tracking
            results = model.track(frame, imgsz=IMGSZ, conf=CONF_THRES, device=DEVICE, persist=True)
            r = results[0]

            if r.boxes is not None:
                for det in r.boxes:
                    x1, y1, x2, y2 = det.xyxy.cpu().numpy()[0]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    obj_id = int(det.id.cpu().numpy()[0]) if det.id is not None else hash((x1, y1, x2, y2)) % 100000

                    prev_cx, prev_cy = previous_positions.get(obj_id, (cx, cy))
                    color = (0, 255, 0)

                    # ================== DETEKSI GATE A ==================
                    if A_y1 <= cy <= A_y2:  # hanya jika titik pusat dalam area garis A
                        if prev_cx < keluarA_line_x <= cx:
                            total_A += 1
                        elif prev_cx > keluarA_line_x >= cx:
                            total_A -= 1

                    # ================== DETEKSI GATE B ==================
                    if B_y1 <= cy <= B_y2:  # hanya jika titik pusat dalam area garis B
                        if prev_cx < keluarB_line_x <= cx:
                            total_B += 1
                        elif prev_cx > keluarB_line_x >= cx:
                            total_B -= 1

                    # Gambar objek
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(annotated, f"ID {obj_id}", (cx-10, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(annotated, (cx, cy), 4, color, -1)

                    previous_positions[obj_id] = (cx, cy)

            # ================== GAMBAR GARIS ==================
            cv2.line(annotated, (keluarA_line_x, A_y1), (keluarA_line_x, A_y2), (255, 0, 0), 4)
            cv2.putText(annotated, "Out A", (keluarA_line_x+10, A_y1+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.line(annotated, (keluarB_line_x, B_y1), (keluarB_line_x, B_y2), (0, 0, 255), 4)
            cv2.putText(annotated, "Out B", (keluarB_line_x+10, B_y1+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ================== FPS + COUNTER ==================
            now = time.time()
            fps = 1.0 / max(now - t_last, 1e-6)
            t_last = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"A: {total_A}  B: {total_B}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            # Tampilkan hasil (resize biar ringan)
            annotated = cv2.resize(annotated, (int(w*SCALE), int(h*SCALE)), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Tracking Karung + Counter", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user (Ctrl+C)")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] camera stopped, jendela ditutup.")


if __name__ == "__main__":
    main()

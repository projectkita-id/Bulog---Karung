import cv2
import numpy as np
import pyrealsense2 as rs
import time
from ultralytics import YOLO

# ================== KONFIGURASI ==================
MODEL_PATH = "bestbaru.pt"
IMGSZ = 416
CONF_THRES = 0.25
DEVICE = "cpu"

COLOR_WIDTH, COLOR_HEIGHT, FPS = 1920, 1080, 30
DEPTH_WIDTH, DEPTH_HEIGHT = 1280, 720
SCALE = 0.6

# ================== MAIN ==================
def nothing(x): pass

def main():
    # Load YOLO
    model = YOLO(MODEL_PATH)

    last_gate = {}
    previous_positions = {}
    total_A, total_B = 0, 0

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Buat window + trackbar
    cv2.namedWindow("Tracking + Kalibrasi")

    cv2.createTrackbar("A_x", "Tracking + Kalibrasi", 1000, COLOR_WIDTH, nothing)
    cv2.createTrackbar("A_y1", "Tracking + Kalibrasi", 0, 100, nothing)
    cv2.createTrackbar("A_y2", "Tracking + Kalibrasi", 40, 100, nothing)

    cv2.createTrackbar("B_x", "Tracking + Kalibrasi", 1000, COLOR_WIDTH, nothing)
    cv2.createTrackbar("B_y1", "Tracking + Kalibrasi", 60, 100, nothing)
    cv2.createTrackbar("B_y2", "Tracking + Kalibrasi", 100, 100, nothing)

    t_last = time.time()
    print("[INFO] Tekan 's' untuk simpan kalibrasi, 'q' untuk keluar")

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

            # Deteksi YOLO
            results = model.track(frame, imgsz=IMGSZ, conf=CONF_THRES, device=DEVICE, persist=True)
            r = results[0]

            if r.boxes is not None:
                for det in r.boxes:
                    x1, y1, x2, y2 = det.xyxy.cpu().numpy()[0]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    obj_id = int(det.id.cpu().numpy()[0]) if det.id is not None else hash((x1, y1, x2, y2)) % 100000

                    prev_cx, prev_cy = previous_positions.get(obj_id, (cx, cy))

                    # Ambil nilai trackbar (garis A & B)
                    A_x  = cv2.getTrackbarPos("A_x", "Tracking + Kalibrasi")
                    A_y1 = int(h * cv2.getTrackbarPos("A_y1", "Tracking + Kalibrasi") / 100)
                    A_y2 = int(h * cv2.getTrackbarPos("A_y2", "Tracking + Kalibrasi") / 100)

                    B_x  = cv2.getTrackbarPos("B_x", "Tracking + Kalibrasi")
                    B_y1 = int(h * cv2.getTrackbarPos("B_y1", "Tracking + Kalibrasi") / 100)
                    B_y2 = int(h * cv2.getTrackbarPos("B_y2", "Tracking + Kalibrasi") / 100)

                    # Hitung crossing garis A
                    if prev_cx < A_x <= cx and A_y1 <= cy <= A_y2:
                        total_A += 1
                    elif prev_cx > A_x >= cx and A_y1 <= cy <= A_y2:
                        total_A -= 1

                    # Hitung crossing garis B
                    if prev_cx < B_x <= cx and B_y1 <= cy <= B_y2:
                        total_B += 1
                    elif prev_cx > B_x >= cx and B_y1 <= cy <= B_y2:
                        total_B -= 1

                    # Gambar bounding box
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(annotated, f"ID {obj_id}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.circle(annotated, (cx, cy), 4, (0,255,0), -1)

                    previous_positions[obj_id] = (cx, cy)

            # Ambil nilai trackbar lagi (buat gambar garis)
            A_x  = cv2.getTrackbarPos("A_x", "Tracking + Kalibrasi")
            A_y1 = int(h * cv2.getTrackbarPos("A_y1", "Tracking + Kalibrasi") / 100)
            A_y2 = int(h * cv2.getTrackbarPos("A_y2", "Tracking + Kalibrasi") / 100)

            B_x  = cv2.getTrackbarPos("B_x", "Tracking + Kalibrasi")
            B_y1 = int(h * cv2.getTrackbarPos("B_y1", "Tracking + Kalibrasi") / 100)
            B_y2 = int(h * cv2.getTrackbarPos("B_y2", "Tracking + Kalibrasi") / 100)

            # Gambar garis A & B
            cv2.line(annotated, (A_x, A_y1), (A_x, A_y2), (255,0,0), 4)
            cv2.putText(annotated, "Out A", (A_x+10, A_y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            cv2.line(annotated, (B_x, B_y1), (B_x, B_y2), (0,0,255), 4)
            cv2.putText(annotated, "Out B", (B_x+10, B_y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # FPS & Counter
            now = time.time()
            fps = 1.0 / max(now - t_last, 1e-6)
            t_last = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(annotated, f"A: {total_A}  B: {total_B}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

            # Resize & show
            annotated = cv2.resize(annotated, (int(w*SCALE), int(h*SCALE)), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Tracking + Kalibrasi", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                with open("kalibrasi.txt", "w") as f:
                    f.write(f"A_x={A_x}\nA_y1={A_y1}\nA_y2={A_y2}\n")
                    f.write(f"B_x={B_x}\nB_y1={B_y1}\nB_y2={B_y2}\n")
                print("[INFO] Nilai kalibrasi disimpan ke kalibrasi.txt")

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user (Ctrl+C)")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] camera stopped, jendela ditutup.")

if __name__ == "__main__":
    main()

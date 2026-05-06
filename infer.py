import cv2 as cv
import cvzone
import torch
import numpy as np
import pywt
import math
import csv
import json
import time
import os
from datetime import datetime
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops
from collections import defaultdict

# Config
VIDEO_PATH      = "./assets/drone-footage.mp4"
MODEL_PATH      = "./models/yolo26m.pt"
OUTPUT_DIR      = "./analysis"
VEHICLE_CLASSES = [2, 3, 5, 7]    # car, motorbike, bus, truck
TRACK_HISTORY   = 30              # frames to keep per track
GRAPHCUT_EVERY  = 5               # run graph cut every N frames
HEATMAP_RADIUS  = 20

session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = os.path.join(OUTPUT_DIR, session_id)
os.makedirs(session_dir, exist_ok=True)

csv_path      = os.path.join(session_dir, "crossings.csv")
json_path     = os.path.join(session_dir, "summary.json")
video_out     = os.path.join(session_dir, "annotated.mp4")
features_path = os.path.join(session_dir, "features.csv")
speeds_path   = os.path.join(session_dir, "speeds.csv")

print(f"Session : {session_id}")
print(f"Output  : {session_dir}\n")

# Wavelet denoising
def waveletDenoise(frame):
    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(float)
    coeffs = pywt.wavedec2(y, wavelet="db1", level=2)
    sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(y.size))
    coeffsThres = [coeffs[0]] + [
        tuple(pywt.threshold(c, threshold, mode="soft") for c in detail)
        for detail in coeffs[1:]
    ]
    y_clean = np.clip(pywt.waverec2(coeffsThres, "db1"), 0, 255).astype(np.uint8)
    ycrcb[:, :, 0] = y_clean[:frame.shape[0], :frame.shape[1]]
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

# Graph cut on bounding box
def graph_cut_patch(img, x1, y1, w, h, margin=8):
    px1 = max(0, x1 - margin)
    py1 = max(0, y1 - margin)
    px2 = min(img.shape[1], x1 + w + margin)
    py2 = min(img.shape[0], y1 + h + margin)
    patch = img[py1:py2, px1:px2].copy()
    if patch.shape[0] < 20 or patch.shape[1] < 20:
        return patch
    mask     = np.zeros(patch.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    
    rect_w = min(max(1, w), patch.shape[1] - margin - 1)
    rect_h = min(max(1, h), patch.shape[0] - margin - 1)

    # Prevent invalid rects if the bounding box is highly clipped
    if rect_w <= 1 or rect_h <= 1:
        return patch 

    rect = (margin, margin, rect_w, rect_h)
    cv.grabCut(patch, mask, rect, bg_model, fg_model, 3, cv.GC_INIT_WITH_RECT)
    
    fg = np.where((mask == cv.GC_PR_FGD) | (mask == cv.GC_FGD), 255, 0).astype(np.uint8)
    result = patch.copy()
    result[fg == 0] = 0
    return result

# SIFT + GLCM feature extraction
sift = cv.SIFT_create()

def extract_features(patch):
    if patch is None or patch.size == 0:
        return None
    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    if gray.shape[0] < 10 or gray.shape[1] < 10:
        return None
    _, desc = sift.detectAndCompute(gray, None)
    sift_feat = np.mean(desc, axis=0) if desc is not None else np.zeros(128)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_feat = np.array([
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
    ])
    return np.concatenate([sift_feat, glcm_feat])

# Geometry helpers
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def lineIntersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# Interactive setup
cap = cv.VideoCapture(VIDEO_PATH)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Hardware Acceleration: Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using Hardware Acceleration: NVIDIA CUDA")
else:
    device = torch.device("cpu")
    print("No GPU found. Falling back to CPU.")
print(f"Using device: {device}")
model = YOLO(MODEL_PATH).to(device)

ret, first_frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

lines            = []
calibration_line = []
current_line     = []
setup_state      = "COUNTING"

def drawUi(event, x, y, flags, params):
    global current_line, lines, calibration_line, setup_state, frame_copy
    if event == cv.EVENT_LBUTTONDOWN:
        if setup_state == "COUNTING":
            current_line.append((x, y))
            cv.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            if len(current_line) == 2:
                lines.append(tuple(current_line))
                cv.line(frame_copy, current_line[0], current_line[1], (0, 0, 255), 3)
                current_line = []
        elif setup_state == "CALIBRATION":
            if len(calibration_line) < 2:
                calibration_line.append((x, y))
                cv.circle(frame_copy, (x, y), 5, (255, 255, 0), -1)
                if len(calibration_line) == 2:
                    cv.line(frame_copy, calibration_line[0], calibration_line[1], (255, 255, 0), 3)
    cv.imshow("Setup Tool", frame_copy)

frame_copy = first_frame.copy()
cv.imshow("Setup Tool", frame_copy)
cv.setMouseCallback("Setup Tool", drawUi)

print("Step 1 — Counting Lines")
print("Click to draw counting lines. Press Enter when done.\n")
while setup_state == "COUNTING":
    if cv.waitKey(1) & 0xFF == 13:
        setup_state = "CALIBRATION"

print("Step 2 — Speed Calibration")
print("Draw ONE line over a known real-world distance. Press Enter when done.\n")
while setup_state == "CALIBRATION":
    if cv.waitKey(1) & 0xFF == 13:
        if len(calibration_line) == 2:
            setup_state = "DONE"
        else:
            print("Warning: draw exactly 2 points before pressing Enter.")
cv.destroyAllWindows()

pxDist = math.hypot(
    calibration_line[1][0] - calibration_line[0][0],
    calibration_line[1][1] - calibration_line[0][1]
)
print(f"Calibration line: {pxDist:.2f} px")
realMeters = float(input("Real-world length in meters (e.g. 3.5): "))
pixelPerMeter = pxDist / realMeters
print(f"Scale: {pixelPerMeter:.2f} px/m")
print("Starting analysis")

# Re-initialise stream
cap.release()
cap = cv.VideoCapture(VIDEO_PATH)
cap.set(cv.CAP_PROP_POS_FRAMES, 1)

frame_h, frame_w = first_frame.shape[:2]
fps = cap.get(cv.CAP_PROP_FPS) or 30

# Video writer
fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer = cv.VideoWriter(video_out, fourcc, fps, (frame_w, frame_h))

# State
track_history   = defaultdict(list)      # track_id -> [(cx, cy, timestamp), ...]
line_crossed    = {i: set() for i in range(len(lines))}   # per-line crossed IDs
line_counts     = {i: 0     for i in range(len(lines))}
heatmap_layer   = np.zeros((frame_h, frame_w), dtype=np.float32)
graphcut_cache  = {}                     # track_id -> refined patch
speed_records   = {}                     # track_id -> latest speed (kmh)
frame_count     = 0
start_time      = time.time()

# CSV writers
crossings_file   = open(csv_path, "w", newline="")
crossings_writer = csv.writer(crossings_file)
crossings_writer.writerow(["timestamp", "frame", "track_id", "line_index", "cx", "cy", "speed_kmh"])

features_file   = open(features_path, "w", newline="")
feat_header     = ["frame", "track_id"] + [f"sift_{i}" for i in range(128)] + \
                  ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
features_writer = csv.writer(features_file)
features_writer.writerow(feat_header)

speeds_file   = open(speeds_path, "w", newline="")
speeds_writer = csv.writer(speeds_file)
speeds_writer.writerow(["frame", "track_id", "speed_kmh"])

# Main loop
while True:
    ret, img = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_count += 1
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Wavelet denoising
    imgClean = waveletDenoise(img)

    # YOLO + ByteTrack
    results = model.track(imgClean, persist=True, tracker="./custom_tracker.yml", classes=VEHICLE_CLASSES, verbose=False, device=device)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh
        track_ids = results[0].boxes.id.int().tolist()

        for box, track_id in zip(boxes, track_ids):
            cx, cy, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1 = cx - w // 2
            y1 = cy - h // 2

            # Track history
            current_time = frame_count / fps
            track = track_history[track_id]
            track.append((cx, cy, current_time))
            if len(track) > TRACK_HISTORY:
                track.pop(0)

            # Speed estimation
            speed_kmh = 0
            if len(track) >= 10:
                prev_cx, prev_cy, prev_time = track[-10]
                dist_px = math.hypot(cx - prev_cx, cy - prev_cy)
                dist_m  = dist_px / pixelPerMeter
                elapsed = current_time - prev_time
                if elapsed > 0:
                    speed_kmh = (dist_m / elapsed) * 3.6
                    speed_records[track_id] = round(speed_kmh, 1)
                    if frame_count % 10 == 0:
                        speeds_writer.writerow([frame_count, track_id, round(speed_kmh, 1)])

            # Graph cut (cached)
            if frame_count % GRAPHCUT_EVERY == 0 or track_id not in graphcut_cache:
                graphcut_cache[track_id] = graph_cut_patch(imgClean, x1, y1, w, h)

            refined_patch = graphcut_cache.get(track_id)

            # Feature extraction (every 10 frames)
            if frame_count % 10 == 0 and refined_patch is not None:
                feats = extract_features(refined_patch)
                if feats is not None:
                    features_writer.writerow([frame_count, track_id] + feats.tolist())

            # Heatmap
            temp_mask = np.zeros_like(heatmap_layer)
            cv.circle(temp_mask, (cx, cy), HEATMAP_RADIUS, 1, -1)
            heatmap_layer += temp_mask

            # Line crossing (per-line)
            if len(track) >= 2:
                prev_pt = (track[-2][0], track[-2][1])
                curr_pt = (cx, cy)
                for i, line in enumerate(lines):
                    if track_id not in line_crossed[i] and \
                       lineIntersect(prev_pt, curr_pt, line[0], line[1]):
                        line_counts[i] += 1
                        line_crossed[i].add(track_id)
                        crossings_writer.writerow([
                            timestamp, frame_count, track_id, i,
                            cx, cy, round(speed_kmh, 1)
                        ])
                        cv.line(img, line[0], line[1], (0, 255, 0), 5)

            # Draw trail
            for j in range(1, len(track)):
                cv.line(img, track[j-1][:2], track[j][:2], (0, 200, 255), 1)

            # Bounding box + label
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"ID:{track_id} | {int(speed_kmh)} km/h",
                               (max(0, x1), max(30, y1)), scale=1, thickness=1, offset=3)

    # Counting lines UI
    for i, line in enumerate(lines):
        cv.line(img, line[0], line[1], (0, 0, 255), 2)
        cvzone.putTextRect(img, f"Line {i}: {line_counts[i]}",
                           (line[0][0], line[0][1] - 10), scale=1, thickness=1)

    # Heatmap overlay
    heatmap_norm  = cv.normalize(heatmap_layer, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    heatmap_color = cv.applyColorMap(heatmap_norm, cv.COLORMAP_JET)
    hot_mask      = heatmap_norm > 5
    img[hot_mask] = cv.addWeighted(img, 0.5, heatmap_color, 0.5, 0)[hot_mask]

    writer.write(img)
    cv.imshow("Traffic Vision", img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Teardown
cap.release()
writer.release()
cv.destroyAllWindows()
crossings_file.close()
features_file.close()
speeds_file.close()

# Save final heatmap
heatmap_final = cv.normalize(heatmap_layer, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imwrite(os.path.join(session_dir, "heatmap.png"), cv.applyColorMap(heatmap_final, cv.COLORMAP_JET))

# JSON summary
elapsed = round(time.time() - start_time, 2)
summary = {
    "session_id":    session_id,
    "video":         VIDEO_PATH,
    "total_frames":  frame_count,
    "duration_sec":  elapsed,
    "fps_processed": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
    "pixels_per_meter": round(pixelPerMeter, 3),
    "line_counts":   line_counts,
    "outputs": {
        "annotated_video": video_out,
        "crossings_csv":   csv_path,
        "features_csv":    features_path,
        "speeds_csv":      speeds_path,
        "heatmap_png":     os.path.join(session_dir, "heatmap.png"),
        "summary_json":    json_path,
    }
}
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n Session complete")
print(f"  Frames processed : {frame_count}")
print(f"  Duration         : {elapsed}s")
for i, count in line_counts.items():
    print(f"  Line {i} count    : {count}")
print(f"  Saved to         : {session_dir}")
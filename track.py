# FOOTBALL PLAYER TRACKING + TEAM CLASSIFICATION PIPELINE
# Using YOLOv8 + SigLIP embeddings + UMAP + DBSCAN + ByteTrack

# Import required libraries
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import umap
from more_itertools import chunked
from PIL import Image


VIDEO_PATH = "15sec_input_720p.mp4"     # Input video path
YOLO_MODEL_PATH = "best.pt"             # Path to fine-tuned YOLOv8 weights
SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"   # SigLIP pretrained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available; here i have used CPU


# Load YOLOv8 model for object detection (football players, ball, referee)
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load SigLIP model for visual embeddings (used for appearance-based team classification)
siglip_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)


cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Extract player crops periodically from video for appearance embedding calculation
def extract_crops(frames, stride=10):
    crops, frame_idxs, coords = [], [], []
    for idx in tqdm(range(0, len(frames), stride), desc="Extracting crops"):
        frame = frames[idx]
        results = yolo_model(frame)
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].cpu().numpy())
            if cls != 2: continue  # Only keep players (class_id = 2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            crops.append(crop)
            frame_idxs.append(idx)
            coords.append((x1, y1, x2, y2))
    return crops, frame_idxs, coords

crops, crop_frames, coords = extract_crops(frames)


# Convert player crops into SigLIP embeddings (appearance features)
def extract_embeddings(crops, batch_size=32):
    embeddings = []
    batches = list(chunked(crops, batch_size))
    for batch in tqdm(batches, desc="Extracting embeddings"):
        pil_batch = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in batch]
        inputs = siglip_processor(images=pil_batch, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = siglip_model(**inputs)
        embed = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        embeddings.append(embed)
    embeddings = np.concatenate(embeddings)
    return embeddings

embeddings = extract_embeddings(crops)


# Use UMAP to reduce embeddings to 2D and DBSCAN to cluster into teams
umap_reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = umap_reducer.fit_transform(embeddings)
dbscan = DBSCAN(eps=0.4, min_samples=2)
cluster_labels = dbscan.fit_predict(embeddings_2d)

# Backup team classifier based on average color (blue vs red jerseys)
def simple_team_classifier(cluster_labels, crops):
    team_labels = []
    for crop in crops:
        mean_color = np.mean(crop, axis=(0, 1))
        if mean_color[0] > mean_color[2]:  # Blue dominance
            team_labels.append(1)
        else:
            team_labels.append(0)
    return np.array(team_labels)

team_class_ids = simple_team_classifier(cluster_labels, crops)

# Store team assignments for each frame
frame_cluster_map = dict()
for idx, frame_num in enumerate(crop_frames):
    if frame_num not in frame_cluster_map:
        frame_cluster_map[frame_num] = []
    frame_cluster_map[frame_num].append({
        "bbox": coords[idx],
        "team_class": team_class_ids[idx],
        "cluster_id": cluster_labels[idx],
        "crop": crops[idx]
    })

# Create visual annotators for bounding boxes & labels
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493"]),  # blue & pink
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493"]),
    text_color=sv.Color.from_hex("#000000"),
    text_position=sv.Position.BOTTOM_CENTER
)

tracker = sv.ByteTrack()  # Use ByteTrack for identity tracking


# Output video writer
out = cv2.VideoWriter("output_final.avi", cv2.VideoWriter_fourcc(*"XVID"), 24, (frames[0].shape[1], frames[0].shape[0]))

for frame_num, frame in enumerate(frames):
    frame_copy = frame.copy()

    # Run YOLO detection on each frame
    results = yolo_model(frame)
    boxes = results[0].boxes

    detections_xyxy, confidences, features, assigned_classes = [], [], [], []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        if cls == 2 and conf > 0.5:
            detections_xyxy.append([x1, y1, x2, y2])
            confidences.append(conf)

            # Compute SigLIP embedding for each player crop
            crop = frame[y1:y2, x1:x2]
            pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inputs = siglip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = siglip_model(**inputs)
            feature = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy().flatten()
            features.append(feature)

            # Assign team using previously clustered team assignments
            assigned_team = 0
            if frame_num in frame_cluster_map:
                for cluster in frame_cluster_map[frame_num]:
                    cx1, cy1, cx2, cy2 = cluster["bbox"]
                    iou_x = max(0, min(x2, cx2) - max(x1, cx1))
                    iou_y = max(0, min(y2, cy2) - max(y1, cy1))
                    if iou_x > 0 and iou_y > 0:
                        assigned_team = cluster["team_class"]
                        break
            assigned_classes.append(assigned_team)

    # Build Supervision detections object
    if len(detections_xyxy) > 0:
        detections = sv.Detections(
            xyxy=np.array(detections_xyxy),
            confidence=np.array(confidences),
            class_id=np.array(assigned_classes),
            tracker_id=None,
            data={"feature": np.array(features)}
        )

        # Apply non-max suppression for stability
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        # Apply ByteTrack tracking
        tracked_detections = tracker.update_with_detections(detections)

        # Assign tracking IDs for annotation
        labels = [f"#{track_id}" for track_id in tracked_detections.tracker_id]

        # Draw final results
        frame_copy = ellipse_annotator.annotate(frame_copy, tracked_detections)
        frame_copy = label_annotator.annotate(frame_copy, tracked_detections, labels=labels)
    else:
        tracked_detections = tracker.update_with_detections(sv.Detections.empty())

    # Write to output video
    out.write(frame_copy)
    cv2.imshow("Tracking", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cv2.destroyAllWindows()


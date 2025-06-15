# Football Video Tracking with YOLO + ReID

This project tracks players, goalkeepers, referees, and the ball in football videos using YOLO object detection, advanced multi-object tracking, re-identification, and team identification via jersey color segmentation (HSV).

---

## System Overview

- **Object Detection:** YOLOv8 (Ultralytics)
- **Multi-Object Tracking:** Custom built tracker with:
  - Hungarian assignment (frame-to-frame association)
  - Appearance-based Re-Identification (ReID)
  - Memory-based ID recovery
- **Team Classification:** Jersey color extraction (HSV masking)
- **Input:** Football match video
- **Output:** Visualized video with bounding boxes, IDs, and team classification

---

## Setup 

1) Download the fine tuned model
https://drive.google.com/file/d/1slzfkV2egwx2Rc7xm-csQbG2DGjKrBdB/view

2) Clone the repo and add your model in the directory 

3) Create a virtual environment and install all the requirements by running - 
pip install -r requirement.txt 

4) python tracker.py 
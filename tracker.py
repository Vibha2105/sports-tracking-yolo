# This system tracks players, goalkeepers, referees, and the ball in football
# videos using YOLO object detection, advanced tracking algorithms, and 
# team identification based on jersey colors.

from ultralytics import YOLO        
import cv2                           
import numpy as np                 
from scipy.spatial.distance import cdist  
from scipy.optimize import linear_sum_assignment 

# Dictionary to store currently active object tracks
# Structure: {track_id: {bbox, class_id, center, last_seen, confidence, etc.}}
tracks = {}

# Dictionary to store recently lost tracks for potential re-identification
# When an object disappears, we keep its information here for a while
track_memory = {}

# Dictionary mapping player IDs to their team numbers (0 or 1)
team_assignments = {}

# Counter for assigning unique IDs to new tracks
next_track_id = 0

# Maximum number of frames an object can be missing before moving to memory
max_disappeared = 15

# Maximum number of frames to keep lost tracks in memory for re-identification
max_memory_time = 60

# Maximum pixel distance between detection and track for association
max_distance = 100

# Larger distance threshold specifically for re-identification (more lenient)
reid_distance_threshold = 120

# HSV color ranges for team identification
# HSV is better than RGB because it's less affected by lighting changes

# Blue team jersey detection (light blue/sky blue range)
blue_range = (np.array([90, 50, 50]), np.array([130, 255, 255]))

# Red team jersey detection (red/pink range)
# Note: Red color wraps around in HSV space, so we need two ranges
red_ranges = [(np.array([0, 50, 50]), np.array([10, 255, 255])),  # Lower red range (0-10 degrees)
              (np.array([170, 50, 50]), np.array([180, 255, 255]))]  # Upper red range (170-180 degrees)

def identify_team_by_color(frame, bbox):
    """
    Identify which team a player belongs to based on jersey color
    
    Args:
        frame: Current video frame (BGR image)
        bbox: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        int: Team ID (0=Blue team, 1=Red team, -1=Unknown)
    """
    x1, y1, x2, y2 = bbox
    
    # Extract the player region from the frame
    roi = frame[y1:y2, x1:x2]  # Region of Interest
    
    # Check if the extracted region is valid
    if roi.size == 0:
        return -1 
    
    # Convert the player region from BGR to HSV color space
    # HSV is better for color-based segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Create binary masks for each team color
    # Pixels matching the color range will be white (255), others black (0)
    blue_pixels = cv2.countNonZero(cv2.inRange(hsv, *blue_range))
    
    # For red, we need two masks because red wraps around in HSV
    red_pixels = cv2.countNonZero(cv2.bitwise_or(cv2.inRange(hsv, *red_ranges[0]), 
                                                cv2.inRange(hsv, *red_ranges[1])))
    
    # Determine team based on which color is more dominant
    min_pixels = 50  # Minimum pixels required for reliable team identification
    
    if blue_pixels > red_pixels and blue_pixels > min_pixels:
        return 0  # Blue team
    elif red_pixels > blue_pixels and red_pixels > min_pixels:
        return 1  # Red team
    else:
        return -1  # Unknown team (not enough colored pixels)

def get_center(bbox):
    """
    Calculate the center point of a bounding box
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        tuple: Center coordinates (center_x, center_y)
    """
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def get_bbox_features(bbox, img):
    """
    Extract simple appearance features from a bounding box region
    Used for re-identification and tracking consistency
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        img: Input image
    
    Returns:
        numpy.array: Feature vector (average BGR color values)
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image boundaries
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Check if bounding box is valid
    if x2 <= x1 or y2 <= y1:
        return np.zeros(3)  # Return zero features if invalid bbox
    
    # Extract the region of interest
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(3)
    
    # Calculate average color values (simple but effective features)
    return np.mean(roi, axis=(0, 1))  # Average BGR values

def calculate_feature_similarity(features1, features2):
    """
    Calculate similarity between two feature vectors using cosine similarity
    
    Args:
        features1, features2: Feature vectors to compare
    
    Returns:
        float: Similarity score (0-1, higher is more similar)
    """
    if features1 is None or features2 is None:
        return 0.0
    
    # Calculate vector norms
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity: cos(θ) = (A·B) / (|A|*|B|)
    similarity = np.dot(features1, features2) / (norm1 * norm2)
    return max(0, similarity)  # Ensure non-negative result

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    IoU measures how much two boxes overlap (0 = no overlap, 1 = perfect overlap)
    
    Args:
        box1, box2: Bounding boxes [x1, y1, x2, y2]
    
    Returns:
        float: IoU score (0-1)
    """
    # Find coordinates of intersection rectangle
    x1_max = max(box1[0], box2[0])  # Left edge of intersection
    y1_max = max(box1[1], box2[1])  # Top edge of intersection
    x2_min = min(box1[2], box2[2])  # Right edge of intersection
    y2_min = min(box1[3], box2[3])  # Bottom edge of intersection
    
    # Calculate intersection area
    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    
    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Return IoU (handle division by zero)
    return intersection / union if union > 0 else 0

def try_reidentification(detection, frame_count, img):
    """
    Attempt to re-identify a new detection with a recently lost track
    This helps maintain consistent IDs when objects temporarily disappear
    
    Args:
        detection: Tuple (bbox, class_id, confidence, class_name)
        frame_count: Current frame number
        img: Current frame image
    
    Returns:
        tuple: (best_match_id, confidence_score) or (None, 0.0) if no match
    """
    bbox, class_id, conf, class_name = detection
    det_center = get_center(bbox)
    det_features = get_bbox_features(bbox, img)
    
    best_match_id = None
    best_score = 0.0
    min_reid_score = 0.4  # Minimum confidence score for re-identification
    
    # Check all tracks currently in memory
    for track_id, memory_info in track_memory.items():
        # Only consider tracks of the same object class
        if memory_info['class_id'] != class_id:
            continue
        
        # Skip tracks that have been in memory too long
        if frame_count - memory_info['last_seen'] > max_memory_time:
            continue
        
        # Calculate distance from detection to last known position
        last_center = memory_info['last_center']
        distance = np.sqrt((det_center[0] - last_center[0])**2 + 
                          (det_center[1] - last_center[1])**2)
        
        # Skip if detection is too far from last known position
        if distance > reid_distance_threshold:
            continue
        
        # Calculate appearance similarity
        feature_sim = calculate_feature_similarity(det_features, memory_info['features'])
        
        # Combine distance and appearance into a single score
        distance_score = max(0, 1 - distance / reid_distance_threshold)
        combined_score = 0.4 * distance_score + 0.6 * feature_sim
        
        # Update best match if this score is better
        if combined_score > best_score and combined_score > min_reid_score:
            best_score = combined_score
            best_match_id = track_id
    
    return best_match_id, best_score

def update_tracks(detections, frame_count, img):
    """
    Update the tracking system with new detections from the current frame
    This is the core function that manages the entire tracking process
    
    Args:
        detections: List of detections [(bbox, class_id, conf, class_name), ...]
        frame_count: Current frame number
        img: Current frame image
    """
    global next_track_id
    
    if not detections:
        return
    
    # Memory Management 
    # Move tracks that have been missing too long to memory
    tracks_to_memory = []
    for track_id, track_info in tracks.items():
        if frame_count - track_info['last_seen'] > max_disappeared:
            tracks_to_memory.append(track_id)
    
    # Transfer old tracks to memory for potential re-identification
    for track_id in tracks_to_memory:
        track_info = tracks[track_id]
        track_memory[track_id] = {
            'class_id': track_info['class_id'],
            'last_center': track_info['center'],
            'last_seen': track_info['last_seen'],
            'features': track_info.get('features', np.zeros(3))
        }
        del tracks[track_id]  # Remove from active tracks
    
    # Clean up very old memory entries
    memory_to_remove = []
    for track_id, memory_info in track_memory.items():
        if frame_count - memory_info['last_seen'] > max_memory_time:
            memory_to_remove.append(track_id)
    
    for track_id in memory_to_remove:
        del track_memory[track_id]
    
    # Reidentification Phase for long term identity recovery 
    # Try to re-identify detections with tracks in memory
    detections_with_reid = []  # Detections that couldn't be re-identified
    
    for detection in detections:
        reid_id, reid_score = try_reidentification(detection, frame_count, img)
        
        if reid_id is not None:
            # Successfully re-identified - Reactivate the track
            bbox, class_id, conf, class_name = detection
            features = get_bbox_features(bbox, img)
            
            # Get team assignment for player detections
            team_id = -1
            if class_id == 2:  # player class
                team_id = identify_team_by_color(img, bbox)
                team_assignments[reid_id] = team_id
            
            # Reactivate track with updated information
            tracks[reid_id] = {
                'bbox': bbox,
                'class_id': class_id,
                'center': get_center(bbox),
                'last_seen': frame_count,
                'confidence': conf,
                'class_name': class_name,
                'features': features,
                'team_id': team_id
            }
            
            # Remove from memory since it's now active again
            if reid_id in track_memory:
                del track_memory[reid_id]
            
            print(f"Re-identified {class_name} ID {reid_id} (score: {reid_score:.2f})")
        else:
            # Couldn't re-identify, will process normally
            detections_with_reid.append(detection)
    
    # Update detections list to only include non-re-identified ones
    detections = detections_with_reid
    
    # Handle cases with no active tracks 
    # Get currently active tracks (recently seen)
    active_tracks = {tid: info for tid, info in tracks.items() 
                    if frame_count - info['last_seen'] <= max_disappeared}
    
    if not active_tracks:
        # No active tracks, create new tracks for all detections
        for bbox, class_id, conf, class_name in detections:
            features = get_bbox_features(bbox, img)
            
            # Get team assignment for player detections
            team_id = -1
            if class_id == 2:  # player class
                team_id = identify_team_by_color(img, bbox)
                team_assignments[next_track_id] = team_id
            
            # Create new track
            tracks[next_track_id] = {
                'bbox': bbox,
                'class_id': class_id,
                'center': get_center(bbox),
                'last_seen': frame_count,
                'confidence': conf,
                'class_name': class_name,
                'features': features,
                'team_id': team_id
            }
            next_track_id += 1
        return
    
    # Hungarian assignment algorithm for frame to frame assignment optimization 
    # Process each object class separately (ball, player, goalkeeper, referee)
    for class_id in set([d[1] for d in detections] + [t['class_id'] for t in active_tracks.values()]):
        # Get detections and tracks for this specific class
        class_detections = [(i, d) for i, d in enumerate(detections) if d[1] == class_id]
        class_tracks = [(tid, info) for tid, info in active_tracks.items() if info['class_id'] == class_id]
        
        if not class_detections or not class_tracks:
            continue  # Skip if no detections or tracks for this class
        
        # Create arrays of detection and track centers
        det_centers = np.array([get_center(d[1][0]) for d in class_detections])
        track_centers = np.array([info['center'] for _, info in class_tracks])
        
        if len(det_centers) > 0 and len(track_centers) > 0:
            # Calculate distance matrix between all detections and tracks
            distance_matrix = cdist(det_centers, track_centers)
            
            # Initialize cost matrix with high values (impossible assignments)
            cost_matrix = np.full((len(class_detections), len(class_tracks)), 1000.0)
            
            # Calculate cost for each detection-track pair
            for i, (det_idx, (bbox, _, conf, class_name)) in enumerate(class_detections):
                det_features = get_bbox_features(bbox, img)
                
                for j, (track_id, track_info) in enumerate(class_tracks):
                    distance = distance_matrix[i, j]
                    iou = calculate_iou(bbox, track_info['bbox'])
                    
                    # Calculate appearance similarity
                    feature_sim = calculate_feature_similarity(det_features, 
                                                             track_info.get('features'))
                    
                    # Only consider assignments within reasonable distance
                    if distance < max_distance:
                        # Combined cost: distance penalty - IoU bonus - appearance bonus
                        cost = distance - (iou * 30) - (feature_sim * 20)
                        cost_matrix[i, j] = max(0, cost)
            
            # Apply Hungarian algorithm for optimal assignment
            try:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                assigned_detections = set()
                
                # Process each assignment
                for det_idx, track_idx in zip(row_indices, col_indices):
                    # Only accept assignments with reasonable cost
                    if cost_matrix[det_idx, track_idx] < max_distance:
                        detection_idx, (bbox, class_id, conf, class_name) = class_detections[det_idx]
                        track_id, track_info = class_tracks[track_idx]
                        
                        # Update track with new detection information
                        features = get_bbox_features(bbox, img)
                        old_features = track_info.get('features', features)
                        
                        # Use weighted average to update features (maintain consistency)
                        updated_features = 0.3 * features + 0.7 * old_features
                        
                        # Update team assignment for players
                        team_id = track_info.get('team_id', -1)
                        if class_id == 2:  # player class
                            new_team_id = identify_team_by_color(img, bbox)
                            if new_team_id != -1:
                                team_id = new_team_id
                                team_assignments[track_id] = team_id
                        
                        # Update track information
                        tracks[track_id].update({
                            'bbox': bbox,
                            'center': get_center(bbox),
                            'last_seen': frame_count,
                            'confidence': conf,
                            'class_name': class_name,
                            'features': updated_features,
                            'team_id': team_id
                        })
                        
                        assigned_detections.add(detection_idx)
                
                # Create new tracks for unassigned detections
                for det_idx, (bbox, class_id, conf, class_name) in class_detections:
                    if det_idx not in assigned_detections:
                        features = get_bbox_features(bbox, img)
                        
                        # Get team assignment for player detections
                        team_id = -1
                        if class_id == 2:  # player class
                            team_id = identify_team_by_color(img, bbox)
                            team_assignments[next_track_id] = team_id
                        
                        # Create new track
                        tracks[next_track_id] = {
                            'bbox': bbox,
                            'class_id': class_id,
                            'center': get_center(bbox),
                            'last_seen': frame_count,
                            'confidence': conf,
                            'class_name': class_name,
                            'features': features,
                            'team_id': team_id
                        }
                        next_track_id += 1
                        
            except Exception as e:
                print(f"Assignment error: {e}")
    
    # Handle remaining unprocessed detections
    # Find classes that were recently updated (have active assignments)
    all_assigned_classes = set()
    for tid, info in active_tracks.items():
        if frame_count - info['last_seen'] <= 1:  # Recently updated
            all_assigned_classes.add(info['class_id'])
    
    # Create new tracks for detections of classes without recent updates
    for bbox, class_id, conf, class_name in detections:
        if class_id not in all_assigned_classes:
            features = get_bbox_features(bbox, img)
            
            # Get team assignment for player detections
            team_id = -1
            if class_id == 2:  # player class
                team_id = identify_team_by_color(img, bbox)
                team_assignments[next_track_id] = team_id
            
            # Create new track
            tracks[next_track_id] = {
                'bbox': bbox,
                'class_id': class_id,
                'center': get_center(bbox),
                'last_seen': frame_count,
                'confidence': conf,
                'class_name': class_name,
                'features': features,
                'team_id': team_id
            }
            next_track_id += 1

def draw_tracks(img, frame_count):
    """
    Draw tracking information on the image including bounding boxes, IDs,
    team information, and statistics
    
    Args:
        img: Image to draw on
        frame_count: Current frame number
    """
    # Color scheme for different object classes
    class_colors = {
        0: (0, 255, 255),    # ball - yellow
        1: (255, 0, 255),    # goalkeeper - magenta  
        2: (0, 255, 0),      # player - green (default, overridden by team colors)
        3: (255, 255, 0)     # referee - cyan
    }
    
    # Team colors for players
    team_colors = {
        0: (255, 0, 0),      # Team 0 - Blue (displayed as blue in BGR)
        1: (0, 0, 255),      # Team 1 - Red (displayed as red in BGR)
        -1: (128, 128, 128)  # Unknown team - Gray
    }
    
    # Additional colors for multiple tracks of same class
    track_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 128, 255),
        (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128)
    ]
    
    # Draw all active tracks
    for track_id, track_info in tracks.items():
        # Skip tracks that haven't been seen recently
        if frame_count - track_info['last_seen'] > 2:
            continue
            
        # Extract track information
        bbox = track_info['bbox']
        class_id = track_info['class_id']
        class_name = track_info['class_name']
        confidence = track_info['confidence']
        center = track_info['center']
        team_id = track_info.get('team_id', -1)
        
        # Choose color: team color for players, track color for others
        if class_id == 2 and team_id in team_colors:  # player with known team
            color = team_colors[team_id]
        else:
            color = track_colors[track_id % len(track_colors)]
        
        # Draw bounding box with thick border
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Create label text with team info for players
        if class_id == 2 and team_id != -1:  # player with known team
            label = f"{class_name}-{track_id} T{team_id} {confidence:.2f}"
        else:
            label = f"{class_name}-{track_id} {confidence:.2f}"
        
        # Calculate label size for background rectangle
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw colored background for label
        cv2.rectangle(img, 
                     (bbox[0], bbox[1] - label_size[1] - 8),
                     (bbox[0] + label_size[0] + 4, bbox[1]), 
                     color, -1)
        
        # Draw label text in white
        cv2.putText(img, label, (bbox[0] + 2, bbox[1] - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point with circle and white border
        cv2.circle(img, center, 5, color, -1) 
        cv2.circle(img, center, 7, (255, 255, 255), 2) 

def main():
    # Load input video file
    cap = cv2.VideoCapture("15sec_input_720p.mp4")
    
    # Load pre-trained YOLO model for football object detection
    model = YOLO("best.pt")
    print("Model classes:", model.names)
    
    # Define class names for our specific model
    # These correspond to the classes the model was trained to detect
    class_names = ["ball", "goalkeeper", "player", "referee"]
    
    # Initialize frame counter
    frame_count = 0
    
    while True:
        # Read next frame from video
        success, img = cap.read()
        if not success:
            print("End of video or failed to read frame")
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        # Run YOLO detection on current frame
        results = model(img, stream=True)
        
        # Collect all detections from YOLO results
        detections = []
        for r in results:
            boxes = r.boxes  # Get detected bounding boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extract class ID and confidence score
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filter detections by confidence and valid class
                    if conf > 0.5 and cls < len(class_names):
                        class_name = class_names[cls]
                        detections.append(((x1, y1, x2, y2), cls, conf, class_name))
        
        print(f"Found {len(detections)} detections")
        
        # This handles track association, team identification, and re-identification
        update_tracks(detections, frame_count, img)
        
        # Draw tracking visualization
        draw_tracks(img, frame_count)
        
        cv2.imshow("Re-ID Football Tracking with Teams", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
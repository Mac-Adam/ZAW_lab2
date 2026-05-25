import numpy as np
import cv2
import os
from os.path import join
from scipy.optimize import linear_sum_assignment
def get_distinct_color(id_num):
    rng = np.random.default_rng(seed=id_num)
    h = rng.integers(0, 180) 
    s, v = 255, 255 
    hsv_pixel = np.uint8([[[h, s, v]]])
    
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
    
    return tuple(int(c) for c in bgr_pixel[0][0])

def parse_det(dataset_path):
    frames = {}
    with open(dataset_path + "/det/det.txt") as file:
        for l in file.readlines():
            vals = [float(x) for x in l.split(',')]
            frame = int(vals[0])
            if frame not in frames:
                frames[frame] = []
            frames[frame].append({
                "bbl": vals[2],
                "bbt": vals[3], 
                "bbw": vals[4],
                "bbh": vals[5],
                "conf": vals[6],
                "id": 0,
            })
    return frames

def display_results(dataset_path, detections):
    imgdir = join(dataset_path, 'img1')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()
    
    last_valid_detections = [] 
    
    for frame_idx, imgname in enumerate(imgnames, start=1):
        
        img = cv2.imread(join(imgdir, imgname))
        if img is None:
            continue
            
        # Check if we have new detections for this frame
        if frame_idx in detections and len(detections[frame_idx]) > 0:
            current_detections = detections[frame_idx]
            last_valid_detections = current_detections # Update fallback memory
        else:
            current_detections = last_valid_detections
        
        for det in current_detections:
            x, y = int(det["bbl"]), int(det["bbt"])
            w, h = int(det["bbw"]), int(det["bbh"])
            
            cv2.rectangle(img, (x, y), (x + w, y + h), get_distinct_color(det['id']), 2)
        
        cv2.imshow('MOT Detections', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Video stopped by user.")
            break

    # Clean up the OpenCV window after the loop finishes
    cv2.destroyAllWindows()
    
    
def calculate_iou(det1, det2):
    x1_A, y1_A = det1["bbl"], det1["bbt"]
    x2_A, y2_A = det1["bbl"] + det1["bbw"], det1["bbt"] + det1["bbh"]
    
    x1_B, y1_B = det2["bbl"], det2["bbt"]
    x2_B, y2_B = det2["bbl"] + det2["bbw"], det2["bbt"] + det2["bbh"]

    x_left   = max(x1_A, x1_B)
    y_top    = max(y1_A, y1_B)
    x_right  = min(x2_A, x2_B)
    y_bottom = min(y2_A, y2_B)

    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    area_A = det1["bbw"] * det1["bbh"]
    area_B = det2["bbw"] * det2["bbh"]

    union_area = float(area_A + area_B - inter_area)

    return inter_area / (union_area + 1e-6)
    
def assign_ids_greedy(detections, iou_threshold=0.5):
    frames = list(detections.keys())
    frames.sort()
    max_id = 0
    for frame in frames:
        if max_id == 0:
            for det in detections[frame]:
                max_id += 1
                det['id'] = max_id
            continue
        
        already_taken = set()  
        for det in detections[frame]:
            assigned = False
            
            for other_det in detections[frame-1]:
                if calculate_iou(det,other_det) >= iou_threshold:
                    if other_det['id'] in already_taken:
                        continue
                    already_taken.add(other_det['id'])
                    det['id'] = other_det['id']
                    assigned = True
                    break
            if not assigned:
                max_id +=1
                det['id'] = max_id
                
def calculate_cost(det1, det2):
    return -calculate_iou(det1, det2)

def assign_ids_hungarian(detections, iou_threshold=0.25):
    frames = list(detections.keys())
    frames.sort()
    max_id = 0
    for frame_idx, frame in enumerate(frames):
        if frame_idx == 0:
            for det in detections[frame]:
                max_id += 1
                det['id'] = max_id
            continue
        
        current_dets = detections[frame]
        prev_dets = detections.get(frame - 1, [])
        
        if len(current_dets) == 0:
            continue
            
        if len(prev_dets) == 0:
            for det in current_dets:
                max_id += 1
                det['id'] = max_id
            continue
            
        costs = np.zeros((len(current_dets), len(prev_dets)))
        for det_idx, det in enumerate(current_dets):
            for prev_det_idx, prev_det in enumerate(prev_dets):
                costs[det_idx, prev_det_idx] = calculate_cost(det, prev_det)
                
        row_ind, col_ind = linear_sum_assignment(costs)
        
        assigned_current = set()
        for r, c in zip(row_ind, col_ind):
            iou = calculate_iou(current_dets[r], prev_dets[c])
            if iou >= iou_threshold:
                current_dets[r]['id'] = prev_dets[c]['id']
                assigned_current.add(r)
                
        for r, det in enumerate(current_dets):
            if r not in assigned_current:
                max_id += 1
                det['id'] = max_id

def assign_ids_kalman(detections, iou_threshold=0.15, max_age=3, min_hits=1, q_std=0.1, r_std=2.0, output_coasted=False, img_size=None, edge_margin=10, velocity_decay=1, inflation_scale=0.8):
    from kalman_tracker import KalmanTracker
    
    if img_size is None and len(detections) > 0:
        # Estimate image size from detections
        max_x = 0
        max_y = 0
        for frame, dets in detections.items():
            for d in dets:
                max_x = max(max_x, d["bbl"] + d["bbw"])
                max_y = max(max_y, d["bbt"] + d["bbh"])
        img_size = (max_x, max_y)
        
    tracker = KalmanTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        q_std=q_std,
        r_std=r_std,
        output_coasted=output_coasted,
        img_size=img_size,
        edge_margin=edge_margin,
        velocity_decay=velocity_decay,
        inflation_scale=inflation_scale
    )
    
    frames = list(detections.keys())
    if not frames:
        return
    max_frame = max(frames)
    
    for frame in range(1, max_frame + 1):
        current_dets = detections.get(frame, [])
        tracked_dets = tracker.step(current_dets)
        detections[frame] = tracked_dets

def save_results(dataset_path,detections):
    out_dir = os.path.join(dataset_path, "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "res.txt"), 'w') as file:
        for frame, dets in detections.items():
            for det in dets:
                file.write(f"{frame},{det['id']},{det['bbl']},{det['bbt']},{det['bbw']},{det['bbh']},1,-1,-1,-1\n")
            

if __name__ == "__main__":
    dataset_path = "./data/evs_mot-train/MOT_02"
    detections = parse_det(dataset_path) 
    assign_ids_kalman(detections,output_coasted=True,max_age=50)
    # assign_ids_hungarian(detections)
    frames = list(detections.keys())
    frames.sort()
    print(frames[-1],len(frames))
    save_results(dataset_path,detections)
    display_results(dataset_path, detections)
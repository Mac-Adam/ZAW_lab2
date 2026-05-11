import numpy as np
import cv2
import os
from os.path import join

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
    
def assign_ids_greedy(detections):
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
                if calculate_iou(det,other_det) > 0.5:
                    if other_det['id'] in already_taken:
                        continue
                    already_taken.add(other_det['id'])
                    det['id'] = other_det['id']
                    assigned = True
                    break
            if not assigned:
                max_id +=1
                det['id'] = max_id
    
                
            

if __name__ == "__main__":
    dataset_path = "./data/evs_mot-train/MOT_02"
    detections = parse_det(dataset_path) 
    assign_ids_greedy(detections)
    frames = list(detections.keys())
    frames.sort()
    print(frames[-1],len(frames))
    display_results(dataset_path, detections)
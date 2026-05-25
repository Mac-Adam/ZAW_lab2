import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    bbox: [left, top, width, height]
    """
    x1_A, y1_A = bbox1[0], bbox1[1]
    x2_A, y2_A = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    
    x1_B, y1_B = bbox2[0], bbox2[1]
    x2_B, y2_B = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

    x_left   = max(x1_A, x1_B)
    y_top    = max(y1_A, y1_B)
    x_right  = min(x2_A, x2_B)
    y_bottom = min(y2_A, y2_B)

    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height

    area_A = bbox1[2] * bbox1[3]
    area_B = bbox2[2] * bbox2[3]

    union_area = float(area_A + area_B - inter_area)

    return inter_area / (union_area + 1e-6)

class KalmanFilter:
    """
    A simple 2D linear Kalman Filter for tracking bounding boxes.
    State space: [xc, yc, w, h, vx, vy]^T
    Measurement space: [xc, yc, w, h]^T
    """
    def __init__(self, x0, P0, Q, R):
        self.x = np.array(x0, dtype=np.float64).reshape(-1, 1)  # 6x1 state vector
        self.P = np.array(P0, dtype=np.float64)  # 6x6 covariance matrix
        self.Q = np.array(Q, dtype=np.float64)  # 6x6 process noise covariance
        self.R = np.array(R, dtype=np.float64)  # 4x4 measurement noise covariance
        
        # State transition matrix F (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Measurement matrix H (mapping state to measurement)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float64)

    def predict(self):
        """Predict the next state."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # Clamp width and height to be positive
        self.x[2, 0] = max(1.0, self.x[2, 0])
        self.x[3, 0] = max(1.0, self.x[3, 0])
        # Clamp velocity to prevent extreme jumps from noise
        max_vel = 40.0
        self.x[4, 0] = np.clip(self.x[4, 0], -max_vel, max_vel)
        self.x[5, 0] = np.clip(self.x[5, 0], -max_vel, max_vel)
        return self.x

    def update(self, z):
        """Update the state with a new measurement."""
        z = np.array(z, dtype=np.float64).reshape(-1, 1)
        y = z - np.dot(self.H, self.x)  # Innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        # Clamp width and height to be positive
        self.x[2, 0] = max(1.0, self.x[2, 0])
        self.x[3, 0] = max(1.0, self.x[3, 0])
        # Clamp velocity to prevent extreme jumps from noise
        max_vel = 40.0
        self.x[4, 0] = np.clip(self.x[4, 0], -max_vel, max_vel)
        self.x[5, 0] = np.clip(self.x[5, 0], -max_vel, max_vel)
        return self.x

class Track:
    """
    Represents a single tracked object with a Kalman Filter, ID, and lifecycle tracking.
    """
    def __init__(self, bbox, track_id, q_std=1.0, r_std=1.0):
        self.id = track_id
        
        # Initial state: [xc, yc, w, h, vx, vy]
        xc = bbox[0] + bbox[2] / 2.0
        yc = bbox[1] + bbox[3] / 2.0
        w = bbox[2]
        h = bbox[3]
        x0 = [xc, yc, w, h, 0.0, 0.0]
        
        # Initial state covariance P0
        # High uncertainty for initial velocity components
        P0 = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0])
        
        # Process noise covariance Q
        # Scale with q_std
        Q = np.diag([
            (q_std * 0.1) ** 2,
            (q_std * 0.1) ** 2,
            (q_std * 0.1) ** 2,
            (q_std * 0.1) ** 2,
            q_std ** 2,
            q_std ** 2
        ])
        
        # Measurement noise covariance R
        # Scale with r_std
        R = np.diag([
            r_std ** 2,
            r_std ** 2,
            (r_std * 2.0) ** 2,
            (r_std * 2.0) ** 2
        ])
        
        self.kf = KalmanFilter(x0, P0, Q, R)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
    def predict(self, velocity_decay=0.98):
        """Advance the state prediction and update tracking age."""
        if self.time_since_update > 0:
            # Decay velocity during coasting to prevent flying off
            self.kf.x[4, 0] *= velocity_decay
            self.kf.x[5, 0] *= velocity_decay
            
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()
        
    def update(self, bbox):
        """Update Kalman filter state with detection bbox."""
        xc = bbox[0] + bbox[2] / 2.0
        yc = bbox[1] + bbox[3] / 2.0
        w = bbox[2]
        h = bbox[3]
        self.kf.update([xc, yc, w, h])
        self.hits += 1
        self.time_since_update = 0
        
    def get_bbox(self):
        """Get current bounding box in [left, top, width, height] format."""
        x = self.kf.x
        xc, yc, w, h = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        bbl = xc - w / 2.0
        bbt = yc - h / 2.0
        return [bbl, bbt, w, h]

    def get_inflated_bbox(self, inflation_scale=0.0):
        """Get current bounding box inflated by position uncertainty standard deviation."""
        bbox = self.get_bbox()
        if inflation_scale <= 0.0:
            return bbox
            
        std_x = np.sqrt(self.kf.P[0, 0] + self.kf.R[0, 0])
        std_y = np.sqrt(self.kf.P[1, 1] + self.kf.R[1, 1])
        
        w_inflated = bbox[2] + inflation_scale * std_x
        h_inflated = bbox[3] + inflation_scale * std_y
        
        xc = bbox[0] + bbox[2] / 2.0
        yc = bbox[1] + bbox[3] / 2.0
        bbl_inflated = xc - w_inflated / 2.0
        bbt_inflated = yc - h_inflated / 2.0
        
        return [bbl_inflated, bbt_inflated, w_inflated, h_inflated]

class KalmanTracker:
    """
    Coordinates Kalman-based tracking of multiple objects.
    """
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.25, q_std=1.0, r_std=1.0, output_coasted=False, img_size=None, edge_margin=20, velocity_decay=0.98, inflation_scale=0.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.q_std = q_std
        self.r_std = r_std
        self.output_coasted = output_coasted
        self.img_size = img_size
        self.edge_margin = edge_margin
        self.velocity_decay = velocity_decay
        self.inflation_scale = inflation_scale
        self.tracks = []
        self.next_id = 1
        
    def step(self, detections):
        """
        Run predict, association, update, and track lifecycle steps for one frame.
        detections: list of dicts {"bbl", "bbt", "bbw", "bbh", "conf", "id"}
        Returns: list of dicts representing updated detections with assigned IDs
        """
        # 1. Predict state for all existing tracks
        for track in self.tracks:
            track.predict(velocity_decay=self.velocity_decay)
            
        # Extract bboxes of current detections
        det_bboxes = [[d["bbl"], d["bbt"], d["bbw"], d["bbh"]] for d in detections]
        
        # 2. Associate detections with tracks
        matched_tracks_and_dets = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks)
        
        if len(self.tracks) > 0 and len(detections) > 0:
            # Cost matrix: tracks (rows) vs detections (columns)
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for t_idx, track in enumerate(self.tracks):
                track_bbox = track.get_inflated_bbox(self.inflation_scale)
                for d_idx, det_bbox in enumerate(det_bboxes):
                    cost_matrix[t_idx, d_idx] = -calculate_iou(track_bbox, det_bbox)
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                track_bbox = self.tracks[r].get_inflated_bbox(self.inflation_scale)
                iou = calculate_iou(track_bbox, det_bboxes[c])
                if iou >= self.iou_threshold:
                    matched_tracks_and_dets.append((self.tracks[r], c))
                    if self.tracks[r] in unmatched_tracks:
                        unmatched_tracks.remove(self.tracks[r])
                    if c in unmatched_detections:
                        unmatched_detections.remove(c)
                        
        # 3. Update matched tracks
        for track, d_idx in matched_tracks_and_dets:
            track.update(det_bboxes[d_idx])
            detections[d_idx]["id"] = track.id
            
        # 4. Initialize new tracks for unmatched detections
        for d_idx in unmatched_detections:
            new_track = Track(det_bboxes[d_idx], self.next_id, self.q_std, self.r_std)
            self.next_id += 1
            self.tracks.append(new_track)
            detections[d_idx]["id"] = new_track.id
            
        # 5. Handle unmatched tracks (coasting & deletion)
        # Delete tracks that haven't been updated for too long, or are near/outside image borders
        alive_tracks = []
        for track in self.tracks:
            if track.time_since_update > self.max_age:
                continue
                
            if self.img_size is not None and len(self.img_size) == 2:
                W, H = self.img_size
                if track.time_since_update > 0:  # unmatched in this frame
                    # Get predicted bounding box center and bottom
                    bbl, bbt, w, h = track.get_bbox()
                    xc = bbl + w / 2.0
                    yc = bbt + h / 2.0
                    bbb = bbt + h
                    is_near_edge = (
                        xc <= self.edge_margin or 
                        xc >= W - self.edge_margin or 
                        yc <= self.edge_margin or 
                        bbb >= H - self.edge_margin
                    )
                    if is_near_edge:
                        continue  # do not coast: delete track
                        
            alive_tracks.append(track)
        self.tracks = alive_tracks
        
        # 6. Build the output list of active detections for this frame
        output_detections = []
        
        # Add actually matched detections if they belong to a confirmed track
        for track, d_idx in matched_tracks_and_dets:
            if track.hits >= self.min_hits:
                output_detections.append(detections[d_idx])
                
        # Add newly initialized tracks if they are immediately confirmed (min_hits == 1)
        for d_idx in unmatched_detections:
            # Look up track in the updated self.tracks list
            track_id = detections[d_idx]["id"]
            track = next((t for t in self.tracks if t.id == track_id), None)
            if track is not None and track.hits >= self.min_hits:
                output_detections.append(detections[d_idx])
                
        # Optional: Add coasted tracks as synthetic detections
        if self.output_coasted:
            for track in self.tracks:
                if track.time_since_update > 0 and track.hits >= self.min_hits:
                    bbox = track.get_bbox()
                    output_detections.append({
                        "bbl": bbox[0],
                        "bbt": bbox[1],
                        "bbw": bbox[2],
                        "bbh": bbox[3],
                        "conf": 0.8,  # slightly lower confidence for synthetic detections
                        "id": track.id
                    })
                    
        return output_detections

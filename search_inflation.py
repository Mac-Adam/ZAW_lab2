import os
import sys
import shutil
import subprocess
import csv

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ZAW_ROOT     = os.path.dirname(SCRIPT_DIR)
TRACKEVAL    = os.path.join(ZAW_ROOT, "TrackEval")
DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "evs_mot-train")
TR_ROOT      = os.path.join(TRACKEVAL, "data", "trackers", "mot_challenge", "Custom-train")

SEQUENCES = {
    "seq1": "MOT_02",
    "seq2": "MOT_03",
    "seq3": "MOT_04",
    "seq4": "MOT_05",
}

sys.path.insert(0, os.path.join(SCRIPT_DIR, "source"))
import main as m

def run_evaluation(inflation_scale, velocity_decay=0.98, q_std=0.1, r_std=2.0, iou_threshold=0.15):
    tracker_name = f"search_infl{inflation_scale:.2f}_dec{velocity_decay:.2f}_q{q_std:.2f}_r{r_std:.2f}"
    tracker_data_dir = os.path.join(TR_ROOT, tracker_name, "data")
    os.makedirs(tracker_data_dir, exist_ok=True)
    
    # Run tracking
    for seq_name, mot_folder in SEQUENCES.items():
        dataset_path = os.path.join(DATA_DIR, mot_folder)
        out_dir      = os.path.join(dataset_path, "out")
        os.makedirs(out_dir, exist_ok=True)
        
        detections = m.parse_det(dataset_path)
        m.assign_ids_kalman(
            detections, 
            iou_threshold=iou_threshold, 
            max_age=3, 
            min_hits=1, 
            q_std=q_std,
            r_std=r_std,
            output_coasted=False,
            velocity_decay=velocity_decay,
            inflation_scale=inflation_scale
        )
        m.save_results(dataset_path, detections)
        dest = os.path.join(tracker_data_dir, f"{seq_name}.txt")
        shutil.copyfile(os.path.join(out_dir, "res.txt"), dest)

    # Run TrackEval
    cmd = [
        sys.executable,
        "scripts/run_mot_challenge.py",
        "--BENCHMARK", "Custom",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--PRINT_CONFIG", "False",
        "--PRINT_RESULTS", "False",
        "--PLOT_CURVES", "False",
    ]
    subprocess.run(cmd, cwd=TRACKEVAL, capture_output=True)
    
    csv_path = os.path.join(TR_ROOT, tracker_name, "pedestrian_detailed.csv")
    if not os.path.exists(csv_path):
        return None
        
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["seq"] == "COMBINED":
                return {
                    "HOTA": float(row["HOTA(0)"]),
                    "MOTA": float(row["MOTA"]),
                    "IDF1": float(row["IDF1"]),
                    "IDSW": float(row["IDSW"]),
                }
    return None

def main():
    # Grid of parameters to test
    # (inflation_scale, velocity_decay)
    grid = [
        (0.0, 0.98),
        (0.1, 0.98),
        (0.2, 0.98),
        (0.3, 0.98),
        (0.4, 0.98),
        (0.5, 0.98),
        (0.6, 0.98),
        (0.7, 0.98),
        (0.8, 0.98),
        (0.9, 0.98),
        (1.0, 0.98),
        (1.2, 0.98),
    ]
    
    print(f"{'scale':<5} | {'decay':<5} | {'HOTA':<6} | {'MOTA':<6} | {'IDF1':<6} | {'IDSW':<5}")
    print("-" * 50)
    
    for scale, decay in grid:
        metrics = run_evaluation(scale, decay)
        if metrics:
            print(f"{scale:<5.2f} | {decay:<5.2f} | {metrics['HOTA']:.4f} | {metrics['MOTA']:.4f} | {metrics['IDF1']:.4f} | {int(metrics['IDSW']):<5}")
        else:
            print(f"{scale:<5.2f} | {decay:<5.2f} | Failed to evaluate")

if __name__ == "__main__":
    main()

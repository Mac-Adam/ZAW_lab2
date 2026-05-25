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

def run_evaluation(q_std, r_std):
    tracker_name = f"search_q{q_std}_r{r_std}"
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
            iou_threshold=0.15, 
            max_age=3, 
            min_hits=1, 
            q_std=q_std,
            r_std=r_std,
            output_coasted=False
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
    # Grid of (q_std, r_std) to test
    grid = [
        (1.0, 1.0),   # Baseline (fast update)
        (0.5, 1.0),
        (0.1, 1.0),
        (0.05, 1.0),
        (0.01, 1.0),  # Very slow velocity update
        (0.1, 2.0),
        (0.05, 2.0),
        (0.01, 2.0),
        (0.1, 5.0),
        (0.05, 5.0),
        (0.01, 5.0),
        (0.1, 10.0),
        (0.05, 10.0),
        (0.01, 10.0), # Maximum damping
    ]
    
    print(f"{'q_std':<6} | {'r_std':<6} | {'HOTA':<6} | {'MOTA':<6} | {'IDF1':<6} | {'IDSW':<5}")
    print("-" * 55)
    
    for q, r in grid:
        metrics = run_evaluation(q, r)
        if metrics:
            print(f"{q:<6.3f} | {r:<6.3f} | {metrics['HOTA']:.4f} | {metrics['MOTA']:.4f} | {metrics['IDF1']:.4f} | {int(metrics['IDSW']):<5}")
        else:
            print(f"{q:<6.3f} | {r:<6.3f} | Failed to evaluate")

if __name__ == "__main__":
    main()

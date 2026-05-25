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

def run_evaluation(iou_threshold, max_age, min_hits, output_coasted):
    tracker_name = f"search_iou{iou_threshold}_age{max_age}_hits{min_hits}_coast{int(output_coasted)}"
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
            max_age=max_age, 
            min_hits=min_hits, 
            output_coasted=output_coasted
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
    grid = [
        # (iou_threshold, max_age, min_hits, output_coasted)
        (0.30, 3, 1, False), # Baseline
        (0.15, 3, 1, False), # Previous best (IDSW=545)
        (0.15, 3, 2, False),
        (0.15, 2, 1, False),
        (0.15, 2, 2, False),
        (0.15, 1, 1, False),
        (0.15, 2, 1, True),
        (0.15, 2, 2, True),
        (0.20, 3, 1, False),
        (0.20, 2, 1, False),
        (0.20, 2, 2, False),
        (0.20, 2, 1, True),
    ]
    
    print(f"{'iou':<5} | {'age':<3} | {'hits':<4} | {'coast':<5} | {'HOTA':<6} | {'MOTA':<6} | {'IDF1':<6} | {'IDSW':<5}")
    print("-" * 65)
    
    for iou, age, hits, coast in grid:
        metrics = run_evaluation(iou, age, hits, coast)
        if metrics:
            print(f"{iou:<5.2f} | {age:<3} | {hits:<4} | {str(coast):<5} | {metrics['HOTA']:.4f} | {metrics['MOTA']:.4f} | {metrics['IDF1']:.4f} | {int(metrics['IDSW']):<5}")
        else:
            print(f"{iou:<5.2f} | {age:<3} | {hits:<4} | {str(coast):<5} | Failed to evaluate")

if __name__ == "__main__":
    main()

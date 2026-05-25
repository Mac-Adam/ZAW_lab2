"""
run_test.py - Run the optimized Kalman tracker on the test dataset sequences
and pack the results into the required Codabench submission ZIP format.

Usage:
    uvenv uni_ai
    python run_test.py
"""

import os
import sys

# Add source directory to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "source"))

import main as m

# Test directory setup
TEST_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "evs_mot-test")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")

TEST_SEQUENCES = ["MOT_01", "MOT_06", "MOT_07"]

# Optimal parameters from search
IOU_THRESHOLD = 0.15
MAX_AGE = 3
MIN_HITS = 1
OUTPUT_COASTED = False
VELOCITY_DECAY = 0.98
INFLATION_SCALE = 0.8

def save_test_results(output_filepath, detections):
    with open(output_filepath, 'w') as file:
        frames = sorted(list(detections.keys()))
        for frame in frames:
            for det in detections[frame]:
                # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, -1, -1, -1
                file.write(f"{frame},{det['id']},{det['bbl']},{det['bbt']},{det['bbw']},{det['bbh']},1,-1,-1,-1\n")

def main():
    print(f"[*] Starting tracking on test dataset...")
    print(f"    Parameters: iou={IOU_THRESHOLD}, max_age={MAX_AGE}, min_hits={MIN_HITS}, output_coasted={OUTPUT_COASTED}, velocity_decay={VELOCITY_DECAY}, inflation_scale={INFLATION_SCALE}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_list = []
    
    for seq in TEST_SEQUENCES:
        seq_path = os.path.join(TEST_DATA_DIR, seq)
        if not os.path.exists(seq_path):
            print(f"[!] Warning: Sequence folder {seq} not found in {TEST_DATA_DIR}. Skipping.")
            continue
            
        print(f"[*] Processing {seq}...")
        detections = m.parse_det(seq_path)
        
        # Run the Kalman filter tracker
        m.assign_ids_kalman(
            detections,
            iou_threshold=IOU_THRESHOLD,
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            output_coasted=OUTPUT_COASTED,
            velocity_decay=VELOCITY_DECAY,
            inflation_scale=INFLATION_SCALE
        )
        
        # Save results
        out_filename = f"{seq}.txt"
        out_filepath = os.path.join(OUTPUT_DIR, out_filename)
        save_test_results(out_filepath, detections)
        file_list.append(out_filename)
        print(f"    Saved tracking results to {out_filepath}")
        
    if not file_list:
        print("[!] No results generated. Exit.")
        return
        
    print(f"\n[+] Successfully generated tracking files in: {OUTPUT_DIR}")
    for filename in file_list:
        print(f"      - {filename}")
    print("\n[+] Done!")

if __name__ == "__main__":
    main()

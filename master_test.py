"""
master_test.py - Run all tracking algorithm/threshold combinations over all datasets and evaluate with TrackEval.

Usage:
    uvenv uni_ai
    python master_test.py

Each configuration is automatically named (e.g. "greedy_0.15", "hungarian_0.60") and its results
are appended to results.md and results_data.js for viewing in dashboard.html.
"""

import sys
import os
import shutil
import subprocess
import csv
import json
import datetime

# ─── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ZAW_ROOT     = os.path.dirname(SCRIPT_DIR)
TRACKEVAL    = os.path.join(ZAW_ROOT, "TrackEval")
DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "evs_mot-train")
GT_ROOT      = os.path.join(TRACKEVAL, "data", "gt",       "mot_challenge", "Custom-train")
TR_ROOT      = os.path.join(TRACKEVAL, "data", "trackers", "mot_challenge", "Custom-train")
SEQMAP_FILE  = os.path.join(TRACKEVAL, "data", "gt",       "mot_challenge", "seqmaps", "Custom-train.txt")
RESULTS_MD   = os.path.join(SCRIPT_DIR, "results.md")
RESULTS_JS   = os.path.join(SCRIPT_DIR, "results_data.js")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results.json")

# Dataset sequence mapping: seq_name -> MOT_folder
SEQUENCES = {
    "seq1": "MOT_02",
    "seq2": "MOT_03",
    "seq3": "MOT_04",
    "seq4": "MOT_05",
}

# Configurations to evaluate
ALGORITHMS  = ["greedy", "hungarian"]
THRESHOLDS  = [0.15, 0.30, 0.60]

# Metrics to extract from pedestrian_detailed.csv (COMBINED row)
KEY_METRICS = ["HOTA(0)", "MOTA", "IDF1", "IDSW", "IDs", "GT_IDs"]


# ─── TrackEval setup ──────────────────────────────────────────────────────────

def setup_trackeval_gt():
    """Create seq1..seq4 ground truth folders and seqmap in TrackEval."""
    print("[*] Setting up TrackEval ground truth folders...")

    # Write seqmap
    os.makedirs(os.path.dirname(SEQMAP_FILE), exist_ok=True)
    with open(SEQMAP_FILE, "w") as f:
        f.write("name\n")
        for seq in SEQUENCES:
            f.write(seq + "\n")
    print(f"    Wrote seqmap: {SEQMAP_FILE}")

    for seq_name, mot_folder in SEQUENCES.items():
        src_mot  = os.path.join(DATA_DIR, mot_folder)
        dst_seq  = os.path.join(GT_ROOT, seq_name)
        dst_gt   = os.path.join(dst_seq, "gt")
        os.makedirs(dst_gt, exist_ok=True)

        # Copy gt.txt
        src_gt = os.path.join(src_mot, "gt", "gt.txt")
        dst_gt_file = os.path.join(dst_gt, "gt.txt")
        if not os.path.exists(dst_gt_file):
            shutil.copyfile(src_gt, dst_gt_file)
            print(f"    Copied gt: {mot_folder}/gt/gt.txt -> {seq_name}/gt/gt.txt")

        # Write adjusted seqinfo.ini (name must match seq_name for TrackEval)
        src_ini = os.path.join(src_mot, "seqinfo.ini")
        dst_ini = os.path.join(dst_seq, "seqinfo.ini")
        with open(src_ini, "r") as f:
            ini_content = f.read()
        # Replace name= line
        lines = ini_content.splitlines()
        new_lines = []
        for line in lines:
            if line.strip().lower().startswith("name="):
                new_lines.append(f"name={seq_name}")
            else:
                new_lines.append(line)
        with open(dst_ini, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"    Wrote seqinfo.ini for {seq_name} (was {mot_folder})")

    print("[+] Ground truth setup complete.\n")


# ─── Tracking ─────────────────────────────────────────────────────────────────

def run_tracker(algorithm, threshold):
    """
    Run the tracking algorithm over all datasets and write results
    to the correct TrackEval tracker directory.
    Returns the tracker_name used.
    """
    # Import tracking functions from source/main.py
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "source"))
    import importlib
    import main as m
    importlib.reload(m)  # ensure fresh state on each call

    tracker_name = f"{algorithm}_{threshold:.2f}"
    print(f"[*] Running tracker: {tracker_name}")

    tracker_data_dir = os.path.join(TR_ROOT, tracker_name, "data")
    os.makedirs(tracker_data_dir, exist_ok=True)

    for seq_name, mot_folder in SEQUENCES.items():
        dataset_path = os.path.join(DATA_DIR, mot_folder)
        out_dir      = os.path.join(dataset_path, "out")
        os.makedirs(out_dir, exist_ok=True)

        detections = m.parse_det(dataset_path)

        if algorithm == "greedy":
            m.assign_ids_greedy(detections, iou_threshold=threshold)
        elif algorithm == "hungarian":
            m.assign_ids_hungarian(detections, iou_threshold=threshold)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Save results into dataset out folder and into TrackEval tracker folder
        m.save_results(dataset_path, detections)
        dest = os.path.join(tracker_data_dir, f"{seq_name}.txt")
        shutil.copyfile(os.path.join(out_dir, "res.txt"), dest)
        print(f"    {mot_folder} ({seq_name}) -> {dest}")

    print(f"[+] Tracking done for {tracker_name}\n")
    return tracker_name


# ─── Evaluation ───────────────────────────────────────────────────────────────

def run_trackeval(tracker_name):
    """Run TrackEval for a specific tracker and return parsed metrics per sequence."""
    print(f"[*] Evaluating with TrackEval: {tracker_name}")
    cmd = [
        sys.executable,
        "scripts/run_mot_challenge.py",
        "--BENCHMARK", "Custom",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--PRINT_CONFIG", "False",
        "--PRINT_RESULTS", "True",
        "--PLOT_CURVES", "False",
    ]
    result = subprocess.run(cmd, cwd=TRACKEVAL, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] TrackEval failed for {tracker_name}:")
        print(result.stderr[-2000:])
        return None

    # Parse the pedestrian_detailed.csv produced by TrackEval
    csv_path = os.path.join(TR_ROOT, tracker_name, "pedestrian_detailed.csv")
    if not os.path.exists(csv_path):
        print(f"[!] CSV not found: {csv_path}")
        return None

    metrics = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["seq"]
            metrics[seq] = {k: row.get(k, "N/A") for k in KEY_METRICS}

    print(f"[+] Evaluation done for {tracker_name}\n")
    return metrics


# ─── Results recording ────────────────────────────────────────────────────────

def load_results_json():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            return json.load(f)
    return {}


def save_results_json(all_results):
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)


def save_results_js(all_results):
    """Write results_data.js so dashboard.html can load it without a server."""
    js_content = "// Auto-generated by master_test.py — do not edit manually\n"
    js_content += "const RESULTS_DATA = " + json.dumps(all_results, indent=2) + ";\n"
    with open(RESULTS_JS, "w") as f:
        f.write(js_content)


def append_results_md(run_label, algorithm, threshold, metrics):
    """Append a markdown section for this run configuration."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seqs = [s for s in metrics if s != "COMBINED"]
    seqs_sorted = sorted(seqs)

    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(f"\n## {run_label} — `{algorithm}` threshold=`{threshold}` — {timestamp}\n\n")

        # Header
        header = "| Sequence | " + " | ".join(KEY_METRICS) + " |"
        sep    = "|" + "|".join(["---"] * (len(KEY_METRICS) + 1)) + "|"
        f.write(header + "\n")
        f.write(sep    + "\n")

        for seq in seqs_sorted + ["COMBINED"]:
            if seq not in metrics:
                continue
            vals = metrics[seq]
            row_vals = []
            for k in KEY_METRICS:
                v = vals.get(k, "N/A")
                try:
                    row_vals.append(f"{float(v):.3f}")
                except (ValueError, TypeError):
                    row_vals.append(str(v))
            f.write(f"| {seq} | " + " | ".join(row_vals) + " |\n")
        f.write("\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    setup_trackeval_gt()

    all_results = load_results_json()
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    configs = [(alg, thr) for alg in ALGORITHMS for thr in THRESHOLDS]
    total   = len(configs)

    for i, (algorithm, threshold) in enumerate(configs, 1):
        run_label = f"{algorithm}_{threshold:.2f}"
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] {run_label}")
        print(f"{'='*60}")

        tracker_name = run_tracker(algorithm, threshold)
        metrics      = run_trackeval(tracker_name)

        if metrics is None:
            print(f"[!] Skipping recording for {run_label} due to evaluation error.")
            continue

        # Store in aggregated results
        if run_label not in all_results:
            all_results[run_label] = []
        all_results[run_label].append({
            "timestamp": timestamp,
            "algorithm": algorithm,
            "threshold": threshold,
            "metrics":   metrics,
        })

        append_results_md(run_label, algorithm, threshold, metrics)
        save_results_json(all_results)
        save_results_js(all_results)

    print("\n" + "="*60)
    print("  All configurations evaluated.")
    print(f"  Results written to:")
    print(f"    {RESULTS_MD}")
    print(f"    {RESULTS_JSON}")
    print(f"    {RESULTS_JS}")
    print(f"  Open dashboard.html in your browser to view the results.")
    print("="*60)


if __name__ == "__main__":
    main()

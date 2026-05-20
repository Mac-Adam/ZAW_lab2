import os
import shutil
import subprocess
import sys

def main():
    # 1. Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zaw_root = os.path.dirname(script_dir)
    
    # Tracking results source
    src_file = os.path.join(script_dir, "data", "evs_mot-train", "MOT_02", "out", "res.txt")
    
    # Destination inside TrackEval
    dest_dir = os.path.join(zaw_root, "TrackEval", "data", "trackers", "mot_challenge", "Custom-train", "MyTracker", "data")
    dest_file = os.path.join(dest_dir, "seq1.txt")
    
    print(f"[*] Copying tracking results...")
    print(f"    From: {src_file}")
    print(f"    To:   {dest_file}")
    
    if not os.path.exists(src_file):
        print(f"[!] Error: Tracking results file '{src_file}' does not exist.")
        print(f"    Please run 'python source/main.py' first to generate results.")
        sys.exit(1)
        
    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(src_file, dest_file)
        print("[+] Copy completed successfully.")
    except Exception as e:
        print(f"[!] Error copying file: {e}")
        sys.exit(1)
        
    # 2. Run TrackEval evaluation script
    trackeval_dir = os.path.join(zaw_root, "TrackEval")
    print(f"\n[*] Running TrackEval in: {trackeval_dir}")
    
    cmd = [
        sys.executable,
        "scripts/run_mot_challenge.py",
        "--BENCHMARK", "Custom",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", "MyTracker"
    ]
    
    try:
        # Run TrackEval script and output results to stdout/stderr in real-time
        result = subprocess.run(cmd, cwd=trackeval_dir)
        if result.returncode == 0:
            print("\n[+] TrackEval validation finished successfully.")
        else:
            print(f"\n[!] TrackEval evaluation failed with return code {result.returncode}")
            sys.exit(result.returncode)
    except Exception as e:
        print(f"[!] Error executing TrackEval: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

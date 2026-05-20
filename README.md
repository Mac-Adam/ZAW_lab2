# projekt2 — MOT Tracking Evaluation

Multi-Object Tracking (MOT) pipeline with automated evaluation using TrackEval.
Compares Greedy and Hungarian assignment algorithms across multiple IoU thresholds and datasets.

---

## Repository Layout

```
ZAW/
├── TrackEval/          ← TrackEval evaluation library (cloned separately)
└── projekt2/
    ├── source/
    │   └── main.py     ← Core tracking code (parse, assign IDs, save)
    ├── data/
    │   └── evs_mot-train/
    │       ├── MOT_02/ ← Sequence 1 (1050 frames, 1920×1080)
    │       ├── MOT_03/ ← Sequence 2 (837 frames, 640×480)
    │       ├── MOT_04/ ← Sequence 3 (525 frames, 1920×1080)
    │       └── MOT_05/ ← Sequence 4 (654 frames, 1920×1080)
    ├── master_test.py  ← Master test runner (main entry point)
    ├── validate.py     ← Single-sequence helper (MOT_02 only)
    ├── dashboard.html  ← Zero-dependency results viewer
    ├── results.md      ← Append-only markdown results table
    ├── results_data.js ← Data file loaded by dashboard.html
    └── results.json    ← Raw results data (machine-readable)
```

---

## Prerequisites

### 1. Python Environment

The project uses a `uv`-managed virtual environment named `uni_ai`.

**Install `uv`** (if not already installed):
```powershell
pip install uv
```

**Create and activate the environment:**
```powershell
uv venv uni_ai
uvenv uni_ai    # activates it (custom alias for: .venv\Scripts\activate or similar)
```

**Install dependencies:**
```powershell
pip install numpy scipy opencv-python
```

### 2. TrackEval Setup

TrackEval must be cloned as a sibling directory to `projekt2`:

```powershell
cd c:\Users\<you>\Documents\AGH\SEM8\ZAW
git clone https://github.com/JonathonLuiten/TrackEval.git
```

**Install TrackEval's minimum requirements:**
```powershell
uvenv uni_ai
pip install -r TrackEval/minimum_requirements.txt
```

### 3. Patch NumPy Compatibility (required for NumPy ≥ 1.20)

The TrackEval library uses deprecated aliases (`np.float`, `np.int`) that were removed in NumPy 1.24.
Apply the following patches:

**`TrackEval/trackeval/datasets/mot_challenge_2d_box.py`**
- Line 228: `dtype=np.float` → `dtype=float`
- Line 359: `np.array([], np.int)` → `np.array([], int)`
- Line 413: `.astype(np.int)` → `.astype(int)`
- Line 420: `.astype(np.int)` → `.astype(int)`

**`TrackEval/trackeval/metrics/hota.py`**
- Lines 31, 37, 38, 42, 43: `dtype=np.float` → `dtype=float`

**`TrackEval/trackeval/metrics/identity.py`**
- Lines 83, 84, 85: `.astype(np.int)` → `.astype(int)`

> These patches are already applied if you are using this repository's copy of TrackEval.

---

## Running the Master Test

The master test script runs **all combinations** of:
- **Algorithms:** `greedy`, `hungarian`
- **Thresholds:** `0.15`, `0.30`, `0.60`

over **all 4 sequences** (`MOT_02` to `MOT_05`).

```powershell
uvenv uni_ai
python master_test.py
```

This will:
1. Set up TrackEval ground truth folders (`seq1`–`seq4`) automatically.
2. Run each configuration (6 total), tracking and saving results.
3. Evaluate each with TrackEval (HOTA, MOTA, IDF1, IDSW, IDs, GT_IDs).
4. Append results to `results.md` (markdown table) and update `results_data.js` + `results.json`.

**Full run takes approximately 5–15 minutes** depending on machine speed.

---

## Viewing Results

### Option A — Markdown Table (quickest)

Open `results.md` in any markdown viewer or text editor.
Each run appends a table like:

| Sequence | HOTA(0) | MOTA | IDF1 | IDSW | IDs | GT_IDs |
|---|---|---|---|---|---|---|
| seq1 | 0.548 | 54.654 | 56.567 | 152.000 | 230.000 | 83.000 |
| COMBINED | 0.548 | 54.654 | 56.567 | 152.000 | 230.000 | 83.000 |

### Option B — HTML Dashboard (no setup required)

Double-click `dashboard.html` to open it in your browser.

Features:
- Results grouped by algorithm.
- Best value per column highlighted green, worst highlighted red.
- History selector to compare multiple runs of the same configuration over time.
- Summary table with all COMBINED rows side by side.

> **Note:** Firefox may block local file `<script src>` loading. Use Chrome or Edge for the dashboard.

---

## Single-Sequence Quick Validate

To quickly validate MOT_02 only (for fast iteration):

```powershell
uvenv uni_ai
cd projekt2
python source/main.py      # re-runs tracking on MOT_02
python validate.py         # copies result to TrackEval and prints metrics
```

---

## Metric Glossary

| Metric | Description |
|---|---|
| **HOTA(0)** | Higher Order Tracking Accuracy at IoU=0.5 — combined detection + association |
| **MOTA** | Multi-Object Tracking Accuracy — penalizes FP, FN, ID switches |
| **IDF1** | ID F1 score — measures identity consistency over time |
| **IDSW** | ID Switches — number of times a tracked object changes its assigned ID |
| **IDs** | Total unique tracker IDs assigned |
| **GT_IDs** | Total unique ground truth object IDs |

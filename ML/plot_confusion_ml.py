"""
Blue recall-normalized confusion-matrix figures from confusion_matrices.xlsx
(the file produced by the ML pipeline). No need to re-run the 100-repeat CV.

- Reads each 4x4 summed confusion matrix (rows=true, cols=pred).
- Divides by N_REPEATS so the printed counts match the real class sizes
  (443 / 443 / 346 / 441 for 25PDB) instead of the x100 summed totals.
- Colour = row-normalized recall; each cell shows count + recall %.
- Greek class labels, same style as the GNN figure.

Usage:
  * one figure  -> set SHEETS = ["VG2D-AND_SVM"]
  * all figures -> set SHEETS = "ALL"
Requires: pandas, numpy, matplotlib, openpyxl
"""
import os
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

INP       = "results_nonnested/confusion_matrices.xlsx"  # adjust path if needed
OUT_DIR   = "confusion_figures"
N_REPEATS = 100                    # must match N_REPEATS used in the ML run
SHEETS    = ["VG2D-AND_SVM"]       # or "ALL" for every sheet

GLABELS = ["\u03b1", "\u03b2", "\u03b1/\u03b2", "\u03b1+\u03b2"]

def nice_title(sheet):
    # e.g. "VG2D-AND_SVM" -> "VG 2D-AND with SVM"
    fs, clf = sheet.rsplit("_", 1)
    fs = fs.replace("VG1D", "VG 1D").replace("HVG1D", "HVG 1D")
    fs = fs.replace("VG2D", "VG 2D").replace("HVG2D", "HVG 2D")
    clf = clf.replace("(added)", " (added)")
    return f"Confusion matrix, {fs} with {clf}"

def plot_sheet(xls, sheet):
    df = pd.read_excel(xls, sheet_name=sheet, header=0, index_col=0)
    C = df.to_numpy(dtype=float) / N_REPEATS
    row = C.sum(1, keepdims=True); row[row == 0] = 1; Nrm = C / row
    plt.rcParams.update({"font.family": ["DejaVu Sans"],
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(Nrm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(GLABELS); ax.set_yticklabels(GLABELS)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(nice_title(sheet), fontweight="bold", pad=12)
    for i in range(4):
        for j in range(4):
            v = Nrm[i, j]; col = "white" if v > 0.55 else "#222222"
            ax.text(j, i, f"{int(round(C[i, j]))}\n{v*100:.1f}%",
                    ha="center", va="center", color=col, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalized (recall)")
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"conf_{sheet}.png")
    fig.savefig(out); plt.close(fig)
    print("saved", out)

def main():
    xls = pd.ExcelFile(INP)
    sheets = xls.sheet_names if SHEETS == "ALL" else SHEETS
    for s in sheets:
        if s in xls.sheet_names:
            plot_sheet(xls, s)
        else:
            print(f"[skip] sheet not found: {s}")

if __name__ == "__main__":
    main()

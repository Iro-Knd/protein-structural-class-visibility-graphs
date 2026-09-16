"""
============================================================================
 Protein structural-class classification from VG/HVG graph features
 Zervou (2021) protocol  +  Random Forest addition
 -- improved SVM tuning (denser grid, scaler inside the pipeline)
 -- results saved to Excel (no need to re-run every time)
============================================================================

Protocol (Zervou et al., 2021, VG/HVG paper):
    - 10-fold cross-validation, repeated N_REPEATS times
    - z-score normalization (here: inside the pipeline, per fold -> no leakage)
    - Gaussian-kernel SVM  (C, gamma tuned once on a log-scaled grid)
    - FDA (Fisher's / Linear Discriminant Analysis)
    - metrics: per-class sensitivity & specificity + Overall Accuracy

The SVM hyperparameters are tuned once on the full dataset and then frozen for
the repeated 10-fold evaluation. This mirrors the protocol of Zervou et al.
(2021) and keeps the comparison fair while remaining computationally light.

Added (not in the paper), suggested by the supervisor:
    - Random Forest (150 trees, no scaling / no tuning needed)

Labels are read from data.txt (last column, a/b/c/d). data.txt and DATA.mat
carry identical labels in the same order, so either is valid; data.txt keeps
features-source and labels in the same place.

Install once:  pip install scikit-learn pandas scipy openpyxl numpy
============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     GridSearchCV)
from sklearn.metrics import confusion_matrix

BASE_DIR   = "."            # folder with the Excel files and data.txt
LABELS_TXT = "data.txt"     # labels come from the LAST column (a/b/c/d)
OUT_DIR    = "results"      # Excel outputs are written here
N_SPLITS   = 10             # 10-fold CV                (Zervou 2021)
N_REPEATS  = 100            # repeated 100 times        (Zervou 2021)
SEED       = 0

# Start with N_REPEATS = 5 to check it runs, then set it to 100 for finals.

FEATURE_FILES = {
    "VG 1D":      "101VG_1D_Results_Parallel.xlsx",
    "HVG 1D":     "101HVG_1D_Results_Parallel.xlsx",
    "HVG 2D-AND": "104HVG_2D_AND_Results_Parallel.xlsx",
    "HVG 2D-OR":  "104HVG_2D_OR_Results_Parallel.xlsx",
    "VG 2D-AND":  "104VG_2D_AND_Results_Parallel.xlsx",
    "VG 2D-OR":   "104VG_2D_OR_Results_Parallel.xlsx",
}

# Readable class names for the output tables (a/b/c/d -> structural class).
CLASS_NAMES = {"a": "alpha", "b": "beta", "c": "alpha_beta", "d": "alpha+beta"}

# Improved SVM grid: denser and shifted toward smaller gamma than the original
# 7-point grid, so the search does not keep hitting the corner (C=1000, g=1e-3).
C_GRID     = np.logspace(-1, 3, 9)      # 0.1 ... 1000
GAMMA_GRID = np.logspace(-4, 0, 9)      # 0.0001 ... 1

# Which classifiers to run. "svm" and "fda" = paper; "rf" = your addition.
CLASSIFIERS = ("svm", "fda", "rf")
LABEL_NAMES = {"svm": "SVM", "fda": "FDA", "rf": "RF (added)"}

# ----------------------------------------------------------------------------
def load_labels(path):
    """Read labels from the LAST column of data.txt (raw a/b/c/d strings)."""
    df = pd.read_csv(path, header=None)
    return df.iloc[:, -1].astype(str).str.strip().to_numpy()

def load_features(path):
    df = pd.read_excel(path, header=None).apply(pd.to_numeric, errors="coerce")
    return df.to_numpy(float)

def sens_spec(cm):
    """Per-class sensitivity & specificity from a confusion matrix (rows=true)."""
    total = cm.sum()
    se, sp = [], []
    for k in range(cm.shape[0]):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        FP = cm[:, k].sum() - TP
        TN = total - TP - FN - FP
        se.append(TP / (TP + FN) if TP + FN else 0.0)
        sp.append(TN / (TN + FP) if TN + FP else 0.0)
    return np.array(se), np.array(sp)

def make_model(name, bestC, bestG):
    """Fresh pipeline per fold. Scaler sits inside -> fitted on train only."""
    if name == "svm":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", C=bestC, gamma=bestG))])
    if name == "fda":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LinearDiscriminantAnalysis())])
    if name == "rf":
        # a forest does not need scaling
        return RandomForestClassifier(n_estimators=150, random_state=SEED)
    raise ValueError(name)

def evaluate(X, y, classes):
    # --- SVM: tune C, gamma once, with the scaler inside the search ---
    grid = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
        {"clf__C": C_GRID, "clf__gamma": GAMMA_GRID},
        cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED),
        n_jobs=-1)
    grid.fit(X, y)
    bestC = grid.best_params_["clf__C"]
    bestG = grid.best_params_["clf__gamma"]

    # --- repeated 10-fold evaluation with the frozen best params ---
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    res = {c: {"oa": [], "se": [], "sp": []} for c in CLASSIFIERS}
    conf = {c: np.zeros((len(classes), len(classes)), dtype=int)
            for c in CLASSIFIERS}

    for tr, te in cv.split(X, y):
        for name in CLASSIFIERS:
            model = make_model(name, bestC, bestG)
            p  = model.fit(X[tr], y[tr]).predict(X[te])
            cm = confusion_matrix(y[te], p, labels=classes)
            se, sp = sens_spec(cm)
            res[name]["oa"].append((p == y[te]).mean())
            res[name]["se"].append(se)
            res[name]["sp"].append(sp)
            conf[name] += cm

    return res, conf, (bestC, bestG)

def ovr_oa(cm):
    """Zervou-style 'overall accuracy': mean of the four one-vs-rest
    binary accuracies (TP+TN)/N, from the summed confusion matrix."""
    total = cm.sum()
    accs = []
    for k in range(cm.shape[0]):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        FP = cm[:, k].sum() - TP
        TN = total - TP - FN - FP
        accs.append((TP + TN) / total)
    return float(np.mean(accs)) * 100

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    y = load_labels(os.path.join(BASE_DIR, LABELS_TXT))
    classes = sorted(set(y))                       # ['a','b','c','d']
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"Loaded {len(y)} proteins. Class counts: {counts}")
    print(f"Protocol: {N_SPLITS}-fold CV x {N_REPEATS} repeats "
          f"(Zervou 2021) + RF, improved tuning\n")

    summary_rows = []   # one row per feature_set x classifier x class
    cm_sheets = {}       # summed confusion matrix per feature_set x classifier
    best_params_lines = []

    for fs_name, fname in FEATURE_FILES.items():
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fs_name}: file not found ({fname})")
            continue
        X = load_features(path)
        if X.shape[0] != len(y):
            raise ValueError(f"{fname}: {X.shape[0]} rows but {len(y)} labels. "
                             "Row order must match data.txt.")
        res, conf, (bestC, bestG) = evaluate(X, y, classes)
        print(f"=== {fs_name}   (SVM best C={bestC:g}, gamma={bestG:g}) ===")
        best_params_lines.append(f"{fs_name} | SVM best C={bestC:g}, gamma={bestG:g}")

        for clf in CLASSIFIERS:
            oa = np.mean(res[clf]["oa"]) * 100 # standard multiclass
            oa_ovr = ovr_oa(conf[clf])  # Zervou-style OvR
            se = np.mean(res[clf]["se"], axis=0) * 100
            sp = np.mean(res[clf]["sp"], axis=0) * 100
            print(f"  {LABEL_NAMES[clf]:11s}  OA={oa:5.2f}%")
            for i, c in enumerate(classes):
                cname = CLASS_NAMES.get(c, c)
                print(f"        {cname:11s}:  Sens={se[i]:5.2f}%  Spec={sp[i]:5.2f}%")
                summary_rows.append({
                    "feature_set": fs_name,
                    "classifier": LABEL_NAMES[clf],
                    "class": cname,
                    "sensitivity": round(se[i], 2),
                    "specificity": round(sp[i], 2),
                    "OA": round(oa, 2),
                    "OA_ovr": round(oa_ovr, 2),
                })
            # summed confusion matrix as a labelled DataFrame
            cnames = [CLASS_NAMES.get(c, c) for c in classes]
            cm_df = pd.DataFrame(conf[clf],
                                 index=[f"true_{n}" for n in cnames],
                                 columns=[f"pred_{n}" for n in cnames])
            sheet = f"{fs_name}_{LABEL_NAMES[clf]}".replace(" ", "").replace("/", "")[:31]
            cm_sheets[sheet] = cm_df
        print()

    # ---- save everything to Excel ------------------------------------------
    summary = pd.DataFrame(summary_rows)
    summary.to_excel(os.path.join(OUT_DIR, "results_summary.xlsx"), index=False)

    with pd.ExcelWriter(os.path.join(OUT_DIR, "confusion_matrices.xlsx")) as w:
        for sheet, cm_df in cm_sheets.items():
            cm_df.to_excel(w, sheet_name=sheet)

    with open(os.path.join(OUT_DIR, "best_params.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(best_params_lines))

    print("Done. Results written to:", OUT_DIR)
    print("  results_summary.xlsx     -> OA / sensitivity / specificity")
    print("  confusion_matrices.xlsx  -> one sheet per feature set x classifier")
    print("  best_params.txt          -> SVM best C, gamma per feature set")

if __name__ == "__main__":
    main()
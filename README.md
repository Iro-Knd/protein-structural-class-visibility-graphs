# Protein structural class classification with visibility graphs

Python reimplementation and extension of the methodology of Zervou et al. (2021)
for protein structural class prediction (all-α, all-β, α/β, α+β) on the 25PDB
dataset, using Chaos Game Representation and visibility graphs.

Diploma thesis, School of ECE, Technical University of Crete.

## Pipeline

PSI-PRED secondary structure (H/C/E) → CGR → visibility graphs →
9 graph features → classification (SVM, FDA, Random Forest)

## Structure

**Shared**
- `chaos_game_representation.py` — CGR mapping onto the triangle, plus plotting
  of the trajectory and of the x/y coordinate signals
- `assortativity.py` — Pearson degree correlation, matching the MATLAB
  Brain Connectivity Toolbox convention

**Graph construction and feature extraction** (one folder per representation)
- `VG1D/` — one-dimensional visibility graph
- `HVG1D/` — one-dimensional horizontal visibility graph
- `VG2D/` — two-dimensional VG, AND and OR variants
- `HVG2D/` — two-dimensional HVG, AND and OR variants

Each folder contains the graph construction (`*_all.py`), a parallel runner
over the full dataset (`run_*_parallel2.py`), and two validation scripts:
`Compare_*.py` (per-feature MSE against the reference values) and
`Find_Exceptions_Which_Line_printout_*.py` (per-protein comparison,
reporting which proteins and which features differ).

**Classification**
- `ML/ML_classification_improved.py` — SVM, FDA and Random Forest under
  stratified 10-fold cross-validation repeated 100 times; reports per-class
  sensitivity and specificity, multiclass accuracy and one-vs-rest accuracy
- `ML/PCA.py` — two-dimensional PCA projection of the feature space
- `ML/plot_confusion_ml.py` — confusion matrix figures

**Graph neural network (exploratory extension)**
- `GNN_Colab.ipynb` — ASAPooling GNN on the 2D HVG-AND graphs, trained on a
  Colab GPU runtime

## Data

The 25PDB dataset is available at
http://biomine.cs.vcu.edu/datasets/SCPRED/SCPRED.html

`ML/data.txt` holds the class labels (last column: a/b/c/d).

## Requirements

numpy, pandas, networkx, scikit-learn, matplotlib, openpyxl, joblib,
torch, torch_geometric

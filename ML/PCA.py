import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = pd.read_excel("104VG_2D_AND_Results_Parallel.xlsx", header=None).to_numpy(float)
y = pd.read_csv("data.txt", header=None).iloc[:, -1].astype(str).str.strip().to_numpy()

Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
proj = pca.fit_transform(Xs)

ev = pca.explained_variance_ratio_ * 100
print(f"PC1 explains {ev[0]:.1f}%, PC2 explains {ev[1]:.1f}%  (total {ev.sum():.1f}%)")

names = {"a": "all-α", "b": "all-β", "c": "α/β", "d": "α+β"}
colors = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868", "d": "#C44E52"}

plt.figure(figsize=(7, 6))
for cls in ["a", "b", "c", "d"]:
    m = y == cls
    plt.scatter(proj[m, 0], proj[m, 1], s=10, alpha=0.5,
                c=colors[cls], label=names[cls])

plt.xlabel(f"PC1 ({ev[0]:.1f}%)")
plt.ylabel(f"PC2 ({ev[1]:.1f}%)")
plt.legend(title="Structural class")
plt.tight_layout()
plt.savefig("pca_vg2d_and.png", dpi=200)
plt.show()
print("Saved -> pca_vg2d_and.png")
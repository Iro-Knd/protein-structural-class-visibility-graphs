"""
============================================================================
 GNN for protein structural-class classification (ASAPooling)
 -- in the spirit of the HGP-SL pooling used by Zervou et al. (EUSIPCO 2022)
 -- exactly 3 node features [state, x, y], as in the paper
============================================================================

WHAT THIS DOES (the deep-learning idea, in plain words)
-------------------------------------------------------
Instead of (graph -> 9 hand-made metrics -> SVM/FDA), here we feed the GRAPH
ITSELF into a Graph Neural Network. The network looks at the nodes and how
they are connected and learns on its own what matters, then predicts the class.
This follows Zervou's 2022 direction: a GNN on the mdHVG graph representation,
using only 3 node features [secondary-structure state, x, y].

NODE FEATURES (exactly 3, matching the paper)
---------------------------------------------
  [state, x, y]  where state is the secondary-structure element encoded as a
  single integer (H -> 0, C -> 1, E -> 2), and x, y are the CGR coordinates.

DATA SOURCES (both in the same order, verified 1673 proteins each)
------------------------------------------------------------------
  - sequences : PSI_PRED.txt  (one H/C/E secondary-structure string per line)
  - labels    : data.txt      (LAST column, a/b/c/d)
No DATA.mat is needed.

POOLING
-------
ASAPooling is a hierarchical, structure-aware pooling operator from the same
family as HGP-SL, and -- unlike the original HGP-SL code -- it is built into
modern PyTorch Geometric, so it needs no extra libraries / old versions.

OUTPUT
------
  gnn_results.xlsx
    - sheet "per_split"  : test accuracy of every random split
    - sheet "summary"    : mean +/- std across splits

Reference points (25PDB, Zervou 2022): ~87% with HGP-SL pooling, ~78% without.

Requires: torch, torch_geometric, pandas, numpy, openpyxl
============================================================================
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, ASAPooling, JumpingKnowledge, global_mean_pool
from sklearn.model_selection import train_test_split

import time
from datetime import datetime
_t0 = time.time()
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

# --------------------------- settings ---------------------------------------
SEQ_FILE   = "PSI_PRED.txt"   # H/C/E sequences, one per line
LABELS_TXT = "data.txt"       # labels from the LAST column (a/b/c/d)
OUT_XLSX   = "gnn_results.xlsx"

N_SPLITS      = 1            # random splits (Zervou uses 15 for large datasets)
EPOCHS_MAX    = 20
PATIENCE      = 10           # early stopping (no val-loss improvement)
HIDDEN        = 64
NUM_LAYERS    = 3
RATIO         = 0.90         # ASAPooling keep-ratio (Zervou's pooling ratio 90%)
DROPOUT       = 0.40
LR            = 0.001
WEIGHT_DECAY  = 0.001
BATCH         = 16
SEED          = 2
torch.manual_seed(SEED); np.random.seed(SEED)

# state encoding: single integer per node (NOT one-hot) -> matches "3 features"
STATE_CODE = {"H": 0.0, "C": 1.0, "E": 2.0}

# --------------------------- CGR (unit triangle, like Zervou) ---------------
VERT = {"H": (0.0, 0.0), "C": (0.5, np.sqrt(3) / 2), "E": (1.0, 0.0)}
def cgr(seq):
    x, y = 0.5, np.sqrt(3) / 6          # triangle centroid
    xs, ys = [], []
    for ch in seq:
        vx, vy = VERT[ch]
        x, y = (x + vx) / 2, (y + vy) / 2
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

# --------------------------- fast 1D HVG ------------------------------------
def hvg_edges(s):
    """Edges (i<j) of the 1D horizontal visibility graph (strict <)."""
    n = len(s); E = set()
    for i in range(n):
        cur_max = float("-inf")
        for j in range(i + 1, n):
            if cur_max < min(s[i], s[j]):
                E.add((i, j))
            cur_max = max(cur_max, s[j])
            if cur_max >= s[i]:
                break
    return E

# --------------------------- build one protein graph ------------------------
def build_graph(seq, label):
    xs, ys = cgr(seq)
    n = len(seq)
    # mdHVG-AND = edges present in BOTH the x-HVG and the y-HVG
    E = hvg_edges(xs) & hvg_edges(ys)
    # node features: EXACTLY 3 -> [state, x, y]  (state as a single integer)
    state = np.array([STATE_CODE[ch] for ch in seq], dtype=np.float32)
    feat = np.column_stack([state, xs, ys]).astype(np.float32)   # shape (n, 3)
    x = torch.tensor(feat, dtype=torch.float)
    if E:
        ei = np.array(list(E)).T
        edge_index = torch.tensor(np.hstack([ei, ei[::-1]]), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

# --------------------------- load sequences + labels ------------------------
print("Loading sequences and labels ...")
with open(SEQ_FILE, encoding="utf-8") as f:
    seqs = [ln.strip() for ln in f if ln.strip()]

lab_raw = pd.read_csv(LABELS_TXT, header=None).iloc[:, -1].astype(str).str.strip()
classes = sorted(lab_raw.unique())                 # ['a','b','c','d']
class_to_int = {c: i for i, c in enumerate(classes)}
labels = lab_raw.map(class_to_int).to_numpy()      # 0..3

assert len(seqs) == len(labels), \
    f"{len(seqs)} sequences but {len(labels)} labels -- order/counts must match."
print(f"  {len(seqs)} proteins | classes {classes} -> {list(range(len(classes)))}")

print("Building graphs ...")
dataset = [build_graph(s, y) for s, y in zip(seqs, labels)]
IN_DIM = dataset[0].x.shape[1]                     # = 3  ([state, x, y])
print(f"  built {len(dataset)} graphs (node feature dim = {IN_DIM})")

# --------------------------- the GNN model ----------------------------------
class PoolNet(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_classes, num_layers, ratio, p):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        for _ in range(num_layers):
            self.pools.append(ASAPooling(hidden, ratio=ratio))
        self.jump = JumpingKnowledge(mode="cat")
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, n_classes)
        self.p = p

    def forward(self, x, edge_index, batch):
        xs = []
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, _ = pool(x, edge_index, batch=batch)
            xs.append(global_mean_pool(x, batch))
        x = self.jump(xs)                          # concat readouts of all layers
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        return self.lin2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crit = torch.nn.CrossEntropyLoss()

def epoch_pass(model, loader, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot = correct = total = 0
    for data in loader:
        data = data.to(device)
        if train: opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = crit(out, data.y)
        if train:
            loss.backward(); opt.step()
        tot += loss.item() * data.num_graphs
        correct += (out.argmax(1) == data.y).sum().item()
        total += data.num_graphs
    return tot / total, correct / total

# --------------------------- train over splits ------------------------------
print(f"Training over {N_SPLITS} random 80/10/10 splits ...")
y_all = labels
rows = []
for split in range(N_SPLITS):
    tr, tmp = train_test_split(range(len(dataset)), test_size=0.20,
                               stratify=y_all, random_state=split)
    _ts = time.time()
    va, te = train_test_split(tmp, test_size=0.50,
                              stratify=y_all[tmp], random_state=split)
    mk = lambda idx, sh=False: DataLoader([dataset[i] for i in idx],
                                          batch_size=BATCH, shuffle=sh)
    tr_l, va_l, te_l = mk(tr, True), mk(va), mk(te)

    model = PoolNet(IN_DIM, HIDDEN, len(classes), NUM_LAYERS, RATIO, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(1, EPOCHS_MAX + 1):
        epoch_pass(model, tr_l, opt)
        val_loss, _ = epoch_pass(model, va_l)
        if val_loss < best_val - 1e-4:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    _, te_acc = epoch_pass(model, te_l)
    rows.append({"split": split + 1, "test_accuracy_%": round(te_acc * 100, 2),
                 "stopped_epoch": ep})
    print(f"  split {split + 1:2d}/{N_SPLITS}: test acc {te_acc*100:5.2f}%  "
          f"(stopped at epoch {ep}, {time.time()-_ts:.0f}s)")

# --------------------------- save to Excel ----------------------------------
per_split = pd.DataFrame(rows)
acc = per_split["test_accuracy_%"].to_numpy()
summary = pd.DataFrame([{
    "n_splits": N_SPLITS,
    "mean_accuracy_%": round(acc.mean(), 2),
    "std_%": round(acc.std(), 2),
    "worst_%": round(acc.min(), 2),
    "best_%": round(acc.max(), 2),
}])

with pd.ExcelWriter(OUT_XLSX) as w:
    per_split.to_excel(w, sheet_name="per_split", index=False)
    summary.to_excel(w, sheet_name="summary", index=False)

print(f"\nMean test accuracy over {N_SPLITS} splits: "
      f"{acc.mean():.2f}%  +/- {acc.std():.2f}")
print(f"Saved -> {OUT_XLSX}")
print("(Zervou 2022 on 25PDB: ~87% with HGP-SL pooling, ~78% without.)")
print(f"End: {datetime.now().strftime('%H:%M:%S')}")
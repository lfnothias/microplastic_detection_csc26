"""Generate manuscript-style result figures from the recorded evaluation numbers (docs/RESULTS.md).

    .venv/bin/python scripts/make_figures.py
Output: docs/figures/fig_results.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))

# (a) cross-sieve generalization (held-out sieve, leakage-free)
labels_a = ["DSLR\n(atypical)", "B\n(typical)"]
vals_a = [0.142, 0.393]
ax[0].bar(labels_a, vals_a, color=["#9e9e9e", "#4363d8"], width=0.6)
ax[0].set_ylabel("mAP@50 (held-out sieve)")
ax[0].set_title("(a) Cross-sieve generalization")
ax[0].set_ylim(0, 0.75)
for i, v in enumerate(vals_a):
    ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

# (b) effect of AI-assisted annotation enrichment (held-out B, same benchmark)
metrics = ["Precision", "Recall", "mAP@50"]
v6 = [0.463, 0.358, 0.393]
v7 = [0.717, 0.389, 0.372]
x = np.arange(len(metrics)); w = 0.38
ax[1].bar(x - w / 2, v6, w, label="original annot.", color="#9e9e9e")
ax[1].bar(x + w / 2, v7, w, label="enriched annot.", color="#e6194b")
ax[1].set_xticks(x); ax[1].set_xticklabels(metrics)
ax[1].set_title("(b) Effect of annotation enrichment")
ax[1].set_ylim(0, 0.9); ax[1].legend(frameon=False, fontsize=8)
for i, (a, b) in enumerate(zip(v6, v7)):
    ax[1].text(i - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.5)
    ax[1].text(i + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.5)

# (c) plastic vs organic confusion (held-out B, among detected particles)
cm = np.array([[110, 7], [4, 37]], float)
cmn = cm / cm.sum(1, keepdims=True)
ax[2].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["plastic", "organic"])
ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["plastic", "organic"])
ax[2].set_xlabel("predicted"); ax[2].set_ylabel("ground truth")
ax[2].set_title("(c) Plastic vs organic (held-out B)")
for i in range(2):
    for j in range(2):
        ax[2].text(j, i, f"{cmn[i, j] * 100:.0f}%\n(n={int(cm[i, j])})", ha="center", va="center",
                   color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig_results.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT / 'fig_results.png'}")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, x, y, w, h, text, color):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        edgecolor="black",
        linewidth=1.0,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
    )


fig, ax = plt.subplots(figsize=(18, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.95, "CNN Architecture for CIFAR-10", ha="center", va="center", fontsize=16, fontweight="bold")

y = 0.48
h = 0.24
gap = 0.015

blocks = [
    (0.02, 0.10, "Input\n32x32x3", "#dbeafe"),
    (0.135, 0.18, "Data Augmentation\nPad(4) -> Crop(32x32)\nHFlip -> Rot(0.08)", "#dcfce7"),
    (0.33, 0.14, "Block 1\nConv64 -> BN -> ReLU\nConv64 -> BN -> ReLU\nMaxPool -> Dropout(0.2)", "#fde68a"),
    (0.485, 0.14, "Block 2\nConv128 -> BN -> ReLU\nConv128 -> BN -> ReLU\nMaxPool -> Dropout(0.3)", "#fcd34d"),
    (0.64, 0.14, "Block 3\nConv256 -> BN -> ReLU\nConv256 -> BN -> ReLU", "#fca5a5"),
    (0.795, 0.08, "GAP\n256", "#ddd6fe"),
    (0.885, 0.08, "Dropout\n0.4", "#fecaca"),
    (0.975, 0.02, "Dense(10)\nSoftmax", "#bfdbfe"),
]

for x, w, label, color in blocks:
    add_box(ax, x, y, w, h, label, color)

for i in range(len(blocks) - 1):
    x, w, _, _ = blocks[i]
    x2, _, _, _ = blocks[i + 1]
    arrow(ax, x + w, y + h / 2, x2, y + h / 2)

tensor_labels = [
    (0.07, "32x32x3"),
    (0.225, "32x32x3"),
    (0.40, "32x32x64"),
    (0.555, "16x16x128"),
    (0.71, "8x8x256"),
    (0.835, "256"),
    (0.925, "256"),
    (0.985, "10 classes"),
]

for x, t in tensor_labels:
    ax.text(x, 0.38, t, ha="center", va="center", fontsize=8)

legend_text = (
    "Legend: Conv/BN/ReLU blocks (yellow/orange/red), "
    "Pooling+Dropout inside blocks, GAP (purple), Classifier (blue)"
)
ax.text(0.5, 0.12, legend_text, ha="center", va="center", fontsize=9)

out_path = r"d:\Year 3-2\DL\Project Based Exam\Deep Learning\cifar10_cnn_architecture.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

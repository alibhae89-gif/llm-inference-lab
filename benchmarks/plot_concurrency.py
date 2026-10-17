import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(label):
    with open(f"results/concurrency_{label}.json") as f:
        return json.load(f)


series = {"1 slot (-np 1)": load("np1"), "4 slots (-np 4)": load("np4")}
panels = [
    ("ttft_median_ms", "Wait for first word (median, ms)", "lower is better"),
    ("per_user_tps", "Speed each user sees (tokens/sec)", "higher is better"),
    ("total_tps", "Total server output (tokens/sec)", "higher is better"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (key, title, note) in zip(axes, panels):
    for name, rows in series.items():
        ax.plot([r["users"] for r in rows], [r[key] for r in rows], marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("Concurrent users")
    ax.set_xticks([1, 2, 4])
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", fontsize=8, alpha=0.7)
axes[0].legend()
fig.suptitle("Qwen2.5-0.5B Q4_K_M on CPU: queueing (1 slot) vs batching (4 slots)")
fig.tight_layout()
fig.savefig("docs/concurrency.png", dpi=150)
print("saved docs/concurrency.png")

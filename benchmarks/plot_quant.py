
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["fp16", "q8_0", "q5_k_m", "q4_k_m", "q2_k"]
KEYS = [("F16", "fp16"), ("Q8_0", "q8_0"), ("Q5_K", "q5_k_m"), ("Q4_K", "q4_k_m"), ("Q2_K", "q2_k")]


def to_mib(text):
    value, unit = text.split()
    return float(value) * (1024 if unit == "GiB" else 1)


speed = {q: {} for q in ORDER}
size = {}
with open("results/quant_speed.md") as f:
    for line in f:
        if not line.startswith("| qwen2"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        name, size_text, test, ts = cells[0], cells[1], cells[5], cells[6]
        q = next(label for key, label in KEYS if key in name)
        mean, std = [float(x) for x in ts.split("±")]
        speed[q][test] = (mean, std)
        size[q] = to_mib(size_text)

ppl = {}
with open("results/quant_ppl.txt") as f:
    for line in f:
        m = re.match(r"(\S+):.*PPL = ([\d.]+)", line)
        if m:
            ppl[m.group(1)] = float(m.group(2))
rise = {q: 100 * (ppl[q] / ppl["fp16"] - 1) for q in ORDER}

rows = ["| Model | Size (MiB) | Decode tok/s | Prefill tok/s | Perplexity | vs fp16 |", "|---|---:|---:|---:|---:|---:|"]
for q in ORDER:
    tg, pp = speed[q]["tg64"], speed[q]["pp128"]
    rows.append(f"| {q} | {size[q]:.0f} | {tg[0]:.1f} ± {tg[1]:.1f} | {pp[0]:.0f} ± {pp[1]:.0f} | {ppl[q]:.2f} | {rise[q]:+.1f}% |")
summary = "\n".join(rows)
print(summary)
with open("results/quant_summary.md", "w") as f:
    f.write(summary + "\n")

x = list(range(len(ORDER)))
decode = [speed[q]["tg64"] for q in ORDER]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].bar(x, [size[q] for q in ORDER], color="#4C78A8")
axes[0].set_title("File size (MiB)")
axes[1].bar(x, [d[0] for d in decode], yerr=[d[1] for d in decode], capsize=4, color="#59A14F")
axes[1].set_title("Decode speed (tokens/sec, higher is better)")
axes[2].bar(x, [rise[q] for q in ORDER], color="#E15759")
axes[2].set_title("Perplexity vs fp16 (% increase, lower is better)")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Qwen2.5-0.5B-Instruct on CPU: size, speed and quality by quantization")
fig.tight_layout()
fig.savefig("docs/quantization.png", dpi=150)
print("saved docs/quantization.png")

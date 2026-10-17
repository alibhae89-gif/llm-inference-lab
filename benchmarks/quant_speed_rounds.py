import random
import statistics
import subprocess
import time

BENCH = "../llama.cpp/build/bin/llama-bench"
MODELS = ["fp16", "q8_0", "q5_k_m", "q4_k_m", "q2_k"]
ROUNDS = 3
REPS = 5
COOLDOWN_S = 15

samples = {q: {"pp128": [], "tg64": []} for q in MODELS}
meta = {}

for round_no in range(1, ROUNDS + 1):
    order = random.sample(MODELS, len(MODELS))
    print(f"round {round_no}: {order}", flush=True)
    for q in order:
        out = subprocess.run(
            [
                BENCH,
                "-m", f"models/qwen2.5-0.5b-instruct-{q}.gguf",
                "-t", "4", "-p", "128", "-n", "64",
                "-r", str(REPS), "-o", "md",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in out.splitlines():
            if not line.startswith("| qwen2"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            samples[q][cells[5]].append(float(cells[6].split("±")[0]))
            meta[q] = (cells[0], cells[1])
        print(f"  {q} done", flush=True)
        time.sleep(COOLDOWN_S)

lines = [
    "| model | size | params | backend | threads | test | t/s |",
    "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
]
for q in MODELS:
    name, size = meta[q]
    for test in ("pp128", "tg64"):
        vals = samples[q][test]
        lines.append(
            f"| {name} | {size} | - | CPU | 4 | {test} | "
            f"{statistics.median(vals):.2f} ± {statistics.stdev(vals):.2f} |"
        )
    print(q, "decode per round:", [round(v, 1) for v in samples[q]["tg64"]])

with open("results/quant_speed.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

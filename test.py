import pandas as pd
import numpy as np
from collections import Counter, defaultdict


# -----------------------------
# Parse TANE FD file
# -----------------------------
def parse_tane(fd_path, csv_path):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    columns = list(df.columns)

    mapping = {}
    fds = []

    with open(fd_path) as f:
        lines = f.readlines()

    in_columns = False
    in_results = False

    for line in lines:
        line = line.strip()

        if line.startswith("# COLUMN"):
            in_columns = True
            continue

        if line.startswith("# RESULTS"):
            in_columns = False
            in_results = True
            continue

        if line.startswith("#"):
            continue

        # Column mapping
        if in_columns and line:
            parts = line.split("\t")
            if len(parts) == 2:
                idx = int(parts[1]) - 1
                mapping[parts[1]] = columns[idx]

        # FDs
        if in_results and "->" in line:
            lhs, rhs = line.split("->")
            lhs_ids = [x.strip() for x in lhs.split(",") if x.strip()]
            rhs_id = rhs.strip()

            lhs = [mapping[x] for x in lhs_ids]
            if rhs_id not in mapping:
                continue
            rhs = mapping[rhs_id]

            fds.append((lhs, rhs))

    return fds


# -----------------------------
# Task 1 analysis
# -----------------------------
def analyze_fds(fds):
    print("\n==============================")
    print("TASK 1 – FD STRUCTURE ANALYSIS")
    print("==============================\n")

    # Number of FDs
    total = len(fds)
    print("Total FDs:", total)

    # Average LHS size
    lhs_sizes = [len(lhs) for lhs, _ in fds]
    print("Average LHS size:", round(np.mean(lhs_sizes), 2))

    # Attribute frequency
    lhs_freq = Counter()
    rhs_freq = Counter()

    for lhs, rhs in fds:
        for a in lhs:
            lhs_freq[a] += 1
        rhs_freq[rhs] += 1

    print("\nTop LHS attributes:")
    for a, c in lhs_freq.most_common(10):
        print(f"  {a}: {c}")

    print("\nTop RHS attributes:")
    for a, c in rhs_freq.most_common(10):
        print(f"  {a}: {c}")

    # ID-based FDs (single attribute determines many)
    single = defaultdict(list)
    for lhs, rhs in fds:
        if len(lhs) == 1:
            single[lhs[0]].append(rhs)

    print("\nID-based FDs (>=3 dependents):")
    for k, v in single.items():
        if len(v) >= 3:
            print(f"  {k} -> {len(v)} attributes")

    # Large determinants
    print("\nLarge determinants (LHS >= 4):")
    for lhs, rhs in fds:
        if len(lhs) >= 4:
            print(f"  {lhs} -> {rhs}")

    # Suspicious (simple definition: large or unique complex)
    print("\nSuspicious FDs:")

    for lhs, rhs in fds:
        if len(lhs) == 1 and lhs[0].lower() in ["id", "index"]:
            print(f"ID-based : {lhs[0]} → {rhs}")

        elif len(lhs) >= 4:
            print(f"LHS très large : {', '.join(lhs)} → {rhs}")

        elif rhs in lhs:
            print(f"Dégénérée : {', '.join(lhs)} → {rhs}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    fds = parse_tane(
        "data/echocardiogram/echocardiogram_fds.txt",
        "data/echocardiogram/echocardiogram.csv"
    )

    analyze_fds(fds)

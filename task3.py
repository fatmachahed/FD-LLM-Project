import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from itertools import combinations
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

DATASETS = [
    "data/abalone",
    "data/pittsburgh+bridges",
    "data/iris",
    "data/nursery",
    "data/echocardiogram",
    "data/balance+scale",
    "data/breast+cancer+wisconsin+original"
]


def discover_fds(sample_df, dataset):
    csv_text = sample_df.to_csv(index=False)

    prompt = f"""
You are given a 50-row sample from dataset {dataset}.

Infer likely Functional Dependencies in the form:

A,B -> C

Return ONLY a JSON list of strings.

Example:
["A -> B", "C,D -> E"]

Dataset sample:
{csv_text}
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    text = r.json()["choices"][0]["message"]["content"]

    start = text.find("[")
    end = text.rfind("]") + 1

    return json.loads(text[start:end])


def check_fd(df, lhs, rhs):
    grouped = df.groupby(lhs)[rhs]

    total = 0
    valid = 0

    for _, series in grouped:
        total += 1
        if series.nunique() == 1:
            valid += 1

    violations = total - valid
    approx = valid / total if total else 0

    return violations, approx




def is_minimal(df, lhs, rhs):
    if len(lhs) == 1:
        return True

    for i in range(len(lhs)):
        sub = lhs[:i] + lhs[i+1:]
        v, _ = check_fd(df, sub, rhs)
        if v == 0:
            return False
    return True


def run_dataset(folder):
    csvs = list(Path(folder).glob("*.csv"))
    if not csvs:
        print(f"No CSV in {folder}")
        return

    df = pd.read_csv(csvs[0])
    sample = df.sample(min(50, len(df)), random_state=42)

    dataset = Path(folder).name
    print(f"\n=== {dataset} ===")

    fds = discover_fds(sample, dataset)

    results = []

    for fd in fds:
        if "->" not in fd:
            continue

        lhs, rhs = fd.split("->")
        lhs = [c.strip() for c in lhs.split(",")]
        rhs = rhs.strip()

        if rhs not in df.columns or not all(c in df.columns for c in lhs):
            continue

        violations, approx = check_fd(df, lhs, rhs)
        minimal = is_minimal(df, lhs, rhs)

        results.append({
            "FD": fd,
            "violations": violations,
            "approx_validity": approx,
            "minimal": minimal,
            "false_positive": violations > 0,
            "misleading": approx > 0.7 and violations > 0
        })

        print(fd, "viol:", violations, "approx:", round(approx, 2))

    out = Path(folder) / "Llm_discovered_fds.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved:", out)

if __name__ == "__main__":
    for d in DATASETS:
        run_dataset(d)

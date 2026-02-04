import os
import re
import json
import time
import requests
from pathlib import Path

# =====================
# CONFIG
# =====================
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
print("GROQ key loaded:", "Yes" if GROQ_API_KEY else "No")

LABELS = ["meaningful", "accidental", "encoding-based", "degenerate", "unlikely"]

DATASETS = [
    "data/abalone",
    "data/pittsburgh+bridges",
    "data/iris",
    "data/nursery",
    "data/echocardiogram",
    "data/balance+scale",
    "data/breast+cancer+wisconsin+original"
]

N_PER_CATEGORY = 3

def extract_fds(fd_file, k=3):
    with open(fd_file, "r", encoding="utf-8") as f:
        txt = f.read()

    suspicious = re.findall(r"=== SUSPICIOUS FDs ===(.*?)(?:===|$)", txt, re.S)
    plausible = re.findall(r"=== PLAUSIBLE FDs ===(.*)", txt, re.S)

    def clean(block):
        out = []
        for line in block.strip().splitlines():
            # remove prefixes like "ID-based:", "Très large:", etc
            line = re.sub(r"^[^:]+:\s*", "", line)
            if "->" in line:
                out.append(line.strip())
        return out

    susp = clean(suspicious[0]) if suspicious else []
    plau = clean(plausible[0]) if plausible else []

    # take up to k, but if less exist, take all
    susp = susp[:k]
    plau = plau[:k]

    # if dataset has only one FD total, just return it
    if len(susp) + len(plau) == 0:
        return []

    return susp + plau


def query_groq(fd, dataset_name, retries=3):
    prompt = f"""
Dataset: {dataset_name}

Functional Dependency:
{fd}

Question:
Does this dependency make sense in the real world?

Classify it as exactly ONE of:
meaningful
accidental
encoding-based
degenerate
unlikely

Respond ONLY with the label.
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

    for i in range(retries):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].lower()

            for l in LABELS:
                if l in text:
                    return l

            return print("DEBUG response:", r.json())


        except Exception:
            if i == retries - 1:
                return print("DEBUG response:", r.json())

            time.sleep(1)



def human_label(fd):
    print("\nFD:", fd)
    print("Choose one:")
    print("meaningful / accidental / encoding-based / degenerate / unlikely")

    while True:
        lab = input("Your judgment: ").strip().lower()
        if lab in LABELS:
            return lab
        print("Invalid label. Try again.")


def run_dataset(dataset_dir):
    fd_file = Path(dataset_dir) / "fd_results.txt"
    dataset_name = Path(dataset_dir).name

    if not fd_file.exists():
        print(f"Missing {fd_file}")
        return []

    fds = extract_fds(fd_file, N_PER_CATEGORY)

    rows = []

    for fd in fds:
        llm = query_groq(fd, dataset_name)
        human = human_label(fd)
        agreement = llm == human

        rows.append({
            "FD": fd,
            "LLM_judgment": llm,
            "Human_judgment": human,
            "Agreement": agreement
        })

        print(f"LLM: {llm} | You: {human} | {'✓' if agreement else '✗'}")

    return rows


def save_results(rows, dataset_dir):
    out = Path(dataset_dir) / "fd_eval.json"

    agreement_rate = sum(r["Agreement"] for r in rows) / len(rows)

    result = {
        "rows": rows,
        "agreement_rate": agreement_rate
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out}")
    print(f"Agreement rate: {agreement_rate:.2f}")



if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set")
        exit(1)

    for d in DATASETS:
        print(f"\n================ {d} ================")
        rows = run_dataset(d)
        if rows:
            save_results(rows, d)

import csv


CATEGORIES = ["meaningful", "accidental", "encoding-based", "degenerate", "unlikely"]


def make_prompt(fd):
    lhs = ", ".join(fd["lhs"])
    rhs = fd["rhs"]

    return f"""
Functional Dependency:

[{lhs}] -> {rhs}

Question:
Does this dependency make sense in the real world?

Classify into ONE:

- meaningful
- accidental
- encoding-based
- degenerate
- unlikely

Return ONLY the category name.
""".strip()


if __name__ == "__main__":

    dataset = "abalone"

    # -------------------------------------------------
    # Manually selected FDs (3 plausible + 3 suspicious)
    # -------------------------------------------------

    fds = [
        # plausible
        {"lhs": ["Shell weight", "Whole weight"], "rhs": "Length"},
        {"lhs": ["Viscera weight", "Shell weight"], "rhs": "Diameter"},
        {"lhs": ["Whole weight", "Shucked weight"], "rhs": "Height"},

        # suspicious
        {"lhs": ["Height", "Diameter", "Rings"], "rhs": "Sex"},
        {"lhs": ["Length", "Whole weight"], "rhs": "Rings"},
        {"lhs": ["Height", "Diameter", "Viscera weight", "Rings"], "rhs": "Length"},
    ]

    rows = []

    for i, fd in enumerate(fds, 1):
        print("\n" + "=" * 60)
        print(f"FD #{i}")

        prompt = make_prompt(fd)
        print("\nPROMPT TO SEND TO LLM:\n")
        print(prompt)

        llm = input("\nPaste LLM judgment: ").strip()
        human = input("Your judgment: ").strip()

        agreement = "YES" if llm == human else "NO"

        rows.append([
            f"[{', '.join(fd['lhs'])}] -> {fd['rhs']}",
            llm,
            human,
            agreement
        ])

    # -------------------------------------------------
    # Save table
    # -------------------------------------------------

    out = f"{dataset}_task2_results.csv"

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FD", "LLM judgment", "Your judgment", "Agreement"])
        writer.writerows(rows)

    print(f"\nSaved results to {out}")

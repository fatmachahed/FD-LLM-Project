import streamlit as st
import pandas as pd
import requests
import os
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
import os
from dotenv import load_dotenv

# charge le fichier .env à la racine
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # export GROQ_API_KEY=xxxx
GROQ_MODEL = "llama3-70b-8192"

LABELS = [
    "meaningful",
    "accidental",
    "encoding-based",
    "degenerate",
    "unlikely"
]

DATASETS = {
    "iris": {
        "data_file": "data/iris/iris.data",
        "fd_file": "data/iris/iris_fds.txt",
        "names_file": "data/iris/iris.names",
        "has_header": False
    },
    "adult": {
        "data_file": "data/adult/adult.data",
        "fd_file": "data/adult/adult_fds.txt",
        "has_header": False
    },
    "balance_scale": {
        "data_file": "data/balance+scale/balance-scale.data",
        "fd_file": "data/balance+scale/balance-scale_fds.txt",
        "has_header": False
    },
    "breast_cancer_wisconsin": {
        "data_file": "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data",
        "fd_file": "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin_fds.txt",
        "has_header": False
    },
    "nursery": {
        "data_file": "data/nursery/nursery.data",
        "fd_file": "data/nursery/nursery_fds.txt",
        "has_header": False
    }
}

# =========================
# UTILITIES
# =========================
def load_fds(fd_file):
    """Charge les FDs depuis un fichier texte"""
    fds = []
    path = Path(fd_file)
    if not path.exists():
        return fds

    with open(path, "r") as f:
        in_results = False
        for line in f:
            line = line.strip()
            if line.startswith("# RESULTS"):
                in_results = True
                continue
            if in_results and "->" in line:
                lhs, rhs = line.split("->")
                lhs = [c.strip() for c in lhs.split(",")]
                rhs = rhs.strip()
                fds.append((lhs, rhs))
    return fds

def ask_llm(fd_str, dataset_name):
    """Appel API LLM pour juger la plausibilité sémantique"""
    prompt = f"""
You are a data semantics expert.

Dataset: {dataset_name}

Functional Dependency:
{fd_str}

Question:
Does this functional dependency make sense in the real world?

Choose exactly ONE label from:
- meaningful
- accidental
- encoding-based
- degenerate
- unlikely

Return ONLY the label.
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        result = response.json()["choices"][0]["message"]["content"].strip().lower()
        for label in LABELS:
            if label in result:
                return label
    except:
        return "error"

    return "unknown"

# =========================
# STREAMLIT UI
# =========================
def run_task2():
    st.header("🤖 Task 2 — LLM-Assisted Semantic FD Discovery")

    # -------------------------
    # CHOIX DU DATASET
    # -------------------------
    dataset_name = st.selectbox(
        "Choisissez le dataset sur lequel travailler",
        list(DATASETS.keys())
    )
    cfg = DATASETS[dataset_name]

    # -------------------------
    # CHARGEMENT DES FDs
    # -------------------------
    fds = load_fds(cfg["fd_file"])
    if not fds:
        st.warning(f"Aucune FD trouvée pour {dataset_name}")
        return

    st.markdown(f"**Dataset sélectionné :** `{dataset_name}`")
    st.write(f"Nombre de FDs détectées : {len(fds)}")

    # -------------------------
    # SELECTION DES FDs
    # -------------------------
    fd_strings = [f"{', '.join(lhs)} → {rhs}" for lhs, rhs in fds]
    selected_fds = st.multiselect(
        "Sélectionnez au moins 6 FDs (3 plausibles + 3 suspectes)",
        fd_strings
    )
    if len(selected_fds) < 6:
        st.info("Veuillez sélectionner au moins 6 FDs.")
        return

    # -------------------------
    # ANALYSE SEMANTIQUE
    # -------------------------
    st.subheader("🧠 Analyse sémantique (LLM vs Humain)")
    results = []
    disagreement_count = 0

    for fd in selected_fds:
        st.markdown(f"### FD : `{fd}`")
        llm_label = ask_llm(fd, dataset_name)
        st.write(f"🤖 Jugement LLM : `{llm_label}`")

        human_label = st.selectbox(
            "Votre jugement",
            LABELS,
            key=f"human_{fd}"
        )

        agreement = "Yes" if llm_label == human_label else "No"

        explanation = ""
        if agreement == "No":
            disagreement_count += 1
            explanation = st.text_area(
                "Expliquez le désaccord (obligatoire)",
                key=f"explain_{fd}"
            )

        results.append({
            "FD": fd,
            "LLM judgment": llm_label,
            "Your judgment": human_label,
            "Agreement?": agreement,
            "Explanation": explanation
        })
        st.divider()

    # -------------------------
    # TABLE DES RESULTATS
    # -------------------------
    st.subheader("📋 Résultats finaux")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    if disagreement_count >= 2:
        st.success("✅ Exigence respectée : au moins 2 désaccords avec le LLM.")
    else:
        st.warning("⚠️ Il faut au moins 2 désaccords (exigence de l’énoncé).")

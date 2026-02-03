import streamlit as st
import pandas as pd
import requests
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import altair as alt  # Pour les graphiques

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "mixtral-8x7b-32768"

LABELS = [
    "meaningful",
    "accidental",
    "encoding-based",
    "degenerate",
    "unlikely"
]

# =========================
# DATASETS
# =========================
DATASETS = {
    "iris": {
        "data_file": "data/iris/iris.csv",
        "fd_file": "data/iris/iris_fds.txt"
    },
    "adult": {
        "data_file": "data/adult/adult.csv",
        "fd_file": "data/adult/adult_fds.txt"
    },
    "balance_scale": {
        "data_file": "data/balance+scale/balance-scale.csv",
        "fd_file": "data/balance+scale/balance-scale_fds.txt"
    },
    "breast_cancer_wisconsin": {
        "data_file": "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin.csv",
        "fd_file": "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin_fds.txt"
    },
    "nursery": {
        "data_file": "data/nursery/nursery.csv",
        "fd_file": "data/nursery/nursery_fds.txt"
    }
}

# =========================
# UTILITIES
# =========================
def load_fds(fd_file, column_names):
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
            if in_results and ("->" in line or "→" in line):
                if "->" in line:
                    lhs, rhs = line.split("->")
                else:
                    lhs, rhs = line.split("→")
                try:
                    lhs_idx = [int(i.strip()) - 1 for i in lhs.split(",")]
                    rhs_idx = int(rhs.strip()) - 1
                    lhs_cols = [column_names[i] for i in lhs_idx]
                    rhs_col = column_names[rhs_idx]
                    fds.append((lhs_cols, rhs_col))
                except Exception:
                    continue
    return fds

def normalize_fd_string(fd_str):
    return fd_str.replace("→", "->")

def ask_llm(fd_str, dataset_name, use_simulation=False):
    fd_str = normalize_fd_string(fd_str)
    if use_simulation:
        return simulate_llm_response(fd_str, dataset_name)
    if not GROQ_API_KEY:
        st.error("❌ Clé API Groq manquante")
        return "missing_api_key"
    prompt = f"""
You are a data semantics expert analyzing functional dependencies.
Dataset: {dataset_name}
FD: {fd_str}
Question: Does this functional dependency make sense?
Return exactly ONE label: {', '.join(LABELS)}
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 20,
        "stop": ["\n"]
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=payload, timeout=15)
        if response.status_code != 200:
            return "api_error"
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip().lower()
        for label in LABELS:
            if label in content:
                return label
        return "unknown"
    except:
        return "exception"

def simulate_llm_response(fd_str, dataset_name):
    fd_str = normalize_fd_string(fd_str)
    if "->" not in fd_str:
        return random.choice(LABELS)
    lhs, rhs = fd_str.split("->")
    lhs = lhs.strip().lower()
    rhs = rhs.strip().lower()
    if lhs == rhs:
        return "degenerate"
    if "id" in lhs or "number" in lhs:
        return "meaningful"
    return random.choices(LABELS, weights=[0.3,0.25,0.2,0.15,0.1])[0]

def hybrid_fd_pipeline(fds, dataset_name, use_simulation=True):
    hybrid_results = []
    for lhs, rhs in fds:
        fd_str = f"{', '.join(lhs)} -> {rhs}"
        if set(lhs) == set([rhs]):
            continue
        label = ask_llm(fd_str, dataset_name, use_simulation)
        if label in ["degenerate", "unlikely"]:
            continue
        hybrid_results.append((lhs, rhs, label))
    return hybrid_results

# =========================
# STREAMLIT UI
# =========================
def run_task4():
    st.header("🤖 Task 4 — Hybrid FD Discovery")

    st.sidebar.subheader("⚙️ Configuration")
    use_simulation = st.sidebar.checkbox("Utiliser le mode simulation (pas d'API)", value=True)

    dataset_name = st.selectbox("Choisissez le dataset", list(DATASETS.keys()))
    cfg = DATASETS[dataset_name]

    data_path = Path(cfg["data_file"])
    if not data_path.exists():
        st.error(f"CSV introuvable : {cfg['data_file']}")
        return
    df = pd.read_csv(data_path)
    st.markdown(f"**Colonnes :** {df.columns.tolist()}, **Lignes :** {len(df)}")

    fd_path = Path(cfg["fd_file"])
    if fd_path.exists():
        fds = load_fds(fd_path, df.columns.tolist())
    else:
        fds = []
        if len(df.columns) >= 3:
            fds = [([df.columns[0]], df.columns[1]),
                   ([df.columns[0], df.columns[1]], df.columns[2]),
                   ([df.columns[1]], df.columns[0])]
    st.write(f"FDs initiales : {len(fds)}")

    # Affichage FDs initiales
    for lhs, rhs in fds:
        st.write(f"{', '.join(lhs)} -> {rhs}")

    # Pipeline hybride
    st.subheader("🧪 Pipeline Hybride")
    with st.spinner("Analyse en cours..."):
        hybrid_results = hybrid_fd_pipeline(fds, dataset_name, use_simulation)

    st.write(f"FDs retenues après pipeline hybride : {len(hybrid_results)}")
    for lhs, rhs, label in hybrid_results:
        st.write(f"{', '.join(lhs)} -> {rhs}  | LLM: {label}")

    # =========================
    # GRAPHIQUES
    # =========================
    if hybrid_results:
        st.subheader("📊 Analyse graphique des FDs")

        # Histogramme des labels LLM
        df_graph = pd.DataFrame(hybrid_results, columns=["LHS", "RHS", "LLM"])
        label_counts = df_graph["LLM"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        chart = alt.Chart(label_counts).mark_bar().encode(
            x="Label",
            y="Count",
            color="Label"
        ).properties(title="Répartition des labels LLM")
        st.altair_chart(chart, use_container_width=True)

        # Comparaison FDs initiales vs retenues
        comp_df = pd.DataFrame({
            "Type": ["Initiales", "Retenues"],
            "Count": [len(fds), len(hybrid_results)]
        })
        chart2 = alt.Chart(comp_df).mark_bar(color="steelblue").encode(
            x="Type",
            y="Count",
            tooltip=["Type", "Count"]
        ).properties(title="FDs : Initiales vs Retenues")
        st.altair_chart(chart2, use_container_width=True)

        # Histogramme taille LHS
        lhs_sizes = [len(lhs) for lhs, rhs, label in hybrid_results]
        lhs_df = pd.DataFrame({"LHS Size": lhs_sizes})
        chart3 = alt.Chart(lhs_df).mark_bar(color="orange").encode(
            x="LHS Size",
            y="count()"
        ).properties(title="Distribution de la taille du LHS")
        st.altair_chart(chart3, use_container_width=True)

    # Export CSV
    st.subheader("💾 Export")
    if st.button("📥 Exporter les résultats"):
        export_data = [{"FD": f"{', '.join(lhs)} -> {rhs}", "LLM": label}
                       for lhs, rhs, label in hybrid_results]
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False, encoding="utf-8")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hybrid_fd_{dataset_name}_{timestamp}.csv"
        st.download_button("Télécharger CSV", data=csv, file_name=filename, mime="text/csv")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    st.set_page_config(page_title="Task 4 - Hybrid FD", layout="wide")
    run_task4()

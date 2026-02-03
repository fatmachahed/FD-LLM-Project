import streamlit as st
import pandas as pd
from collections import Counter
from pathlib import Path

# =========================
# CONFIGURATION DES DATASETS
# =========================
DATASETS = {
    "abalone": {
        "data_file": "data/abalone/abalone.csv",
        "fd_file": "data/abalone/abalone_fds.txt"
    },
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
    },
    # "echocardiogram": {
    #     "data_file": "data/echocardiogram/echocardiogram.csv",
    #     "fd_file": "data/echocardiogram/echocardiogram_fds.txt"
    # }
}

# =========================
# UTILS (RÉUTILISABLES)
# =========================
def list_datasets():
    """Retourne la liste des datasets disponibles"""
    return list(DATASETS.keys())


def load_dataset(dataset_name: str):
    """Charge un dataset CSV et retourne le DataFrame"""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset inconnu : {dataset_name}")

    path = DATASETS[dataset_name]["data_file"]
    return pd.read_csv(path)


def make_unique(columns):
    """Ensure column names are unique"""
    seen = {}
    result = []

    for c in columns:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")

    return result


def load_fds(fd_file, column_names):
    """
    Load functional dependencies from file.
    Format :
    # RESULTS
    1,2,3->4
    """
    fds = []

    if not Path(fd_file).exists():
        return fds

    with open(fd_file, "r") as f:
        in_results = False

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                if "RESULTS" in line:
                    in_results = True
                continue

            if in_results and "->" in line:
                try:
                    lhs, rhs = line.split("->")
                    lhs_idx = [int(i.strip()) - 1 for i in lhs.split(",")]
                    rhs_idx = int(rhs.strip()) - 1

                    lhs_cols = [column_names[i] for i in lhs_idx]
                    rhs_col = column_names[rhs_idx]

                    fds.append((lhs_cols, rhs_col))

                except Exception:
                    continue

    return fds


# =========================
# STREAMLIT UI
# =========================
def run_task1():
    st.header("🧠 Task 1 — Algorithmic Functional Dependency Analysis")

    dataset_name = st.selectbox(
        "Choisir un dataset",
        list_datasets()
    )

    cfg = DATASETS[dataset_name]

    try:
        df = load_dataset(dataset_name)
    except Exception as e:
        st.error(f"Erreur chargement dataset : {e}")
        return

    columns = make_unique(df.columns.tolist())
    df.columns = columns

    with st.expander("🔍 Colonnes détectées"):
        st.write(columns)

    fds = load_fds(cfg["fd_file"], columns)

    # --- Save for Task 2 ---
    st.session_state["fds"] = fds
    st.session_state["df_columns"] = columns
    st.session_state["dataset_name"] = dataset_name

    # --- Stats ---
    st.subheader("📊 Statistiques générales")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nb lignes", df.shape[0])
    c2.metric("Nb colonnes", df.shape[1])
    c3.metric("Nb FDs", len(fds))

    if fds:
        avg_lhs = sum(len(lhs) for lhs, _ in fds) / len(fds)
        st.write(f"**Taille moyenne du LHS :** {avg_lhs:.2f}")
    else:
        st.write("**Taille moyenne du LHS :** N/A")

    # --- Frequencies ---
    st.subheader("📈 Fréquence des attributs")

    lhs_counter = Counter()
    rhs_counter = Counter()

    for lhs, rhs in fds:
        lhs_counter.update(lhs)
        rhs_counter.update([rhs])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**LHS**")
        st.dataframe(pd.DataFrame(lhs_counter.items(), columns=["Attribut", "Fréquence"]))

    with col2:
        st.markdown("**RHS**")
        st.dataframe(pd.DataFrame(rhs_counter.items(), columns=["Attribut", "Fréquence"]))

    # --- Display FDs ---
    st.subheader("🔗 Dépendances Fonctionnelles")
    for lhs, rhs in fds:
        st.code(f"{', '.join(lhs)} → {rhs}")

    # --- Suspicious ---
    st.subheader("⚠️ FDs suspectes")
    suspicious = False

    for lhs, rhs in fds:
        if len(lhs) == 1 and lhs[0].lower() in ["id", "index"]:
            st.warning(f"ID-based : {lhs[0]} → {rhs}")
            suspicious = True
        elif len(lhs) >= 4:
            st.warning(f"LHS large : {', '.join(lhs)} → {rhs}")
            suspicious = True
        elif rhs in lhs:
            st.warning(f"Dégénérée : {', '.join(lhs)} → {rhs}")
            suspicious = True

    if not suspicious:
        st.success("Aucune FD suspecte détectée")

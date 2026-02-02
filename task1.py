import streamlit as st
import re
import pandas as pd
from collections import Counter
from pathlib import Path

# =========================
# CONFIGURATION DES DATASETS
# =========================
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
        "has_header": False,
        "columns": [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"
        ]
    },

    "balance_scale": {
        "data_file": "data/balance+scale/balance-scale.data",
        "fd_file": "data/balance+scale/balance-scale_fds.txt",
        "has_header": False,
        "columns": [
            "class",
            "left_weight",
            "left_distance",
            "right_weight",
            "right_distance"
        ]
    },

    "breast_cancer_wisconsin": {
        "data_file": "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data",
        "fd_file": "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin_fds.txt",
        "has_header": False,
        "columns": [
            "sample_code_number",
            "clump_thickness",
            "uniformity_cell_size",
            "uniformity_cell_shape",
            "marginal_adhesion",
            "single_epithelial_cell_size",
            "bare_nuclei",
            "bland_chromatin",
            "normal_nucleoli",
            "mitoses",
            "class"
        ]
    },

    "nursery": {
        "data_file": "data/nursery/nursery.data",
        "fd_file": "data/nursery/nursery_fds.txt",
        "names_file": "data/nursery/nursery.names",
        "has_header": False,
        "columns": [
            "parents",
            "has_nurs",
            "form",
            "children",
            "housing",
            "finance",
            "social",
            "health",
            "class"
        ]
    }
}


def make_unique(columns):
    """Ensure column names are unique"""
    seen = {}
    result = []
    for c in columns:
        if c not in seen:
            seen[c] = 0
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
    return result


def extract_columns_from_names(names_file):
    """Extract attribute names from .names file"""
    columns = []
    in_attr_section = False

    with open(names_file, "r") as f:
        for line in f:
            line = line.strip()

            if "Attribute Information" in line:
                in_attr_section = True
                continue

            if in_attr_section:
                if line == "" or line.lower().startswith("summary"):
                    break

                match = re.match(r"^\d+\.\s*([a-zA-Z0-9_\- ]+)", line)
                if match:
                    col = (
                        match.group(1)
                        .lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                    )
                    columns.append(col)

    columns.append("class")
    return make_unique(columns)


def load_fds(fd_file, column_names):
    """Load functional dependencies from file"""
    fds = []

    if not Path(fd_file).exists():
        return fds

    with open(fd_file, "r") as f:
        in_results = False

        for line in f:
            line = line.strip()

            if line.startswith("# RESULTS"):
                in_results = True
                continue

            if in_results and "->" in line:
                lhs, rhs = line.split("->")
                lhs_idx = [int(i) - 1 for i in lhs.split(",")]
                rhs_idx = int(rhs) - 1

                try:
                    fds.append(
                        ([column_names[i] for i in lhs_idx], column_names[rhs_idx])
                    )
                except IndexError:
                    pass  # FD invalide

    return fds


# =========================
# STREAMLIT UI
# =========================
def run_task1():
    st.header("🧠 Task 1 — Algorithmic Functional Dependency Analysis")

    dataset_name = st.selectbox("Choisir un dataset", list(DATASETS.keys()))
    cfg = DATASETS[dataset_name]

    # -------- LOAD COLUMNS --------
    if dataset_name == "iris":
        columns = extract_columns_from_names(cfg["names_file"])
    else:
        columns = make_unique(cfg["columns"])

    with st.expander("🔍 Colonnes détectées"):
        st.write(columns)

    # -------- LOAD DATA --------
    df = pd.read_csv(
        cfg["data_file"],
        header=None if not cfg["has_header"] else "infer",
        names=columns
    )

    # -------- LOAD FDs --------
    fds = load_fds(cfg["fd_file"], columns)

    # Save for Task 2
    st.session_state["fds"] = fds
    st.session_state["df_columns"] = columns
    st.session_state["dataset_name"] = dataset_name

    # -------- STATISTICS --------
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

    # -------- FREQUENCIES --------
    st.subheader("📈 Fréquence des attributs")

    lhs_counter = Counter()
    rhs_counter = Counter()

    for lhs, rhs in fds:
        lhs_counter.update(lhs)
        rhs_counter.update([rhs])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**LHS**")
        st.dataframe(
            pd.DataFrame(lhs_counter.items(), columns=["Attribut", "Fréquence"])
        )

    with col2:
        st.markdown("**RHS**")
        st.dataframe(
            pd.DataFrame(rhs_counter.items(), columns=["Attribut", "Fréquence"])
        )

    # -------- DISPLAY FDs --------
    st.subheader("🔗 Dépendances Fonctionnelles")

    if fds:
        for lhs, rhs in fds:
            st.code(f"{', '.join(lhs)} → {rhs}")
    else:
        st.info("Aucune FD trouvée")

    # -------- SUSPICIOUS FDs --------
    st.subheader("⚠️ FDs suspectes")

    suspicious_found = False

    for lhs, rhs in fds:
        if len(lhs) == 1 and lhs[0].lower() in ["id", "index"]:
            st.warning(f"ID-based : {lhs[0]} → {rhs}")
            suspicious_found = True

        elif len(lhs) >= 4:
            st.warning(f"LHS très large : {', '.join(lhs)} → {rhs}")
            suspicious_found = True

        elif rhs in lhs:
            st.warning(f"Dégénérée : {', '.join(lhs)} → {rhs}")
            suspicious_found = True

    if not suspicious_found:
        st.success("Aucune FD suspecte détectée")

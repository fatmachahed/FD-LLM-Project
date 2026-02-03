import streamlit as st
from task1 import run_task1
from task2 import run_task2
from task4 import run_task4  # <-- Import Task 4

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Functional Dependency Explorer",
    layout="wide"
)

st.title("🔗 Functional Dependency Explorer")
st.markdown("""
**Projet : Functional Dependency Discovery & Semantic Analysis**  
Université Paris-Dauphine
""")

# =========================
# NAVIGATION
# =========================
task = st.sidebar.radio(
    "Choisir une tâche",
    [
        "Task 1 — Algorithmic FD Analysis",
        "Task 2 — Semantic FD Evaluation",
        "Task 4 — Hybrid FD Discovery"  # <-- Ajouter Task 4
    ]
)

# =========================
# TASK ROUTING
# =========================
if task.startswith("Task 1"):
    run_task1()
elif task.startswith("Task 2"):
    run_task2()
else:
    run_task4()  # <-- Lancer Task 4

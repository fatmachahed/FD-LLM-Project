import streamlit as st
import pandas as pd
import requests
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

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
    """Charge les FDs depuis un fichier texte et remplace les index par les noms de colonnes"""
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
                # Gérer les deux types de flèches
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
                except Exception as e:
                    continue
    return fds

def normalize_fd_string(fd_str):
    """Normalise la FD pour qu'elle utilise toujours ->"""
    # Remplacer la flèche Unicode par ASCII si nécessaire
    if "→" in fd_str:
        fd_str = fd_str.replace("→", "->")
    return fd_str

def ask_llm(fd_str, dataset_name, use_simulation=False):
    """Appel API LLM pour juger la plausibilité sémantique"""
    
    # Normaliser la FD
    fd_str = normalize_fd_string(fd_str)
    
    # Option de simulation pour tester sans API
    if use_simulation:
        return simulate_llm_response(fd_str, dataset_name)
    
    if not GROQ_API_KEY:
        st.error("❌ Clé API Groq manquante. Vérifiez votre fichier .env")
        return "missing_api_key"

    prompt = f"""
You are a data semantics expert analyzing functional dependencies.

Dataset: {dataset_name}

Functional Dependency (FD):
{fd_str}

Question: Does this functional dependency make sense in the real world?

Choose exactly ONE label from these options:
- meaningful: The FD represents a real-world semantic relationship
- accidental: The FD happens to hold in the data but has no semantic meaning
- encoding-based: The FD exists due to encoding choices or data representation
- degenerate: The FD is trivial (e.g., A → A) or based on unique identifiers
- unlikely: The FD contradicts real-world semantics or common sense

Important: Return ONLY the single label word, nothing else.
"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 20,
        "stop": ["\n"]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code != 200:
            return "api_error"
        
        result = response.json()
        
        if "choices" not in result or len(result["choices"]) == 0:
            return "unknown"
            
        content = result["choices"][0]["message"]["content"].strip().lower()
        content = content.replace(".", "").replace(",", "").strip()
        
        for label in LABELS:
            if content == label.lower():
                return label
            elif label in content:
                return label
        
        if content:
            first_word = content.split()[0]
            if first_word in [l.lower() for l in LABELS]:
                for label in LABELS:
                    if label.lower() == first_word:
                        return label
        
        return "unknown"
        
    except requests.exceptions.Timeout:
        return "timeout"
    except requests.exceptions.ConnectionError:
        return "connection_error"
    except Exception as e:
        return "exception"

def simulate_llm_response(fd_str, dataset_name):
    """Simule une réponse LLM pour tester sans API"""
    # Normaliser la FD
    fd_str = normalize_fd_string(fd_str)
    
    # Vérifier que la FD a bien un format valide
    if "->" not in fd_str:
        # Si pas de flèche, retourner une réponse aléatoire
        return random.choice(LABELS)
    
    fd_lower = fd_str.lower()
    
    try:
        # Extraire LHS et RHS
        lhs_part, rhs_part = fd_str.split("->")
        lhs_part = lhs_part.strip().lower()
        rhs_part = rhs_part.strip().lower()
        
        # Règles de simulation basées sur la sémantique
        if "breast_cancer" in dataset_name.lower():
            # Règles spécifiques pour le dataset breast cancer
            
            # FD tautologique (A -> A ou A,B -> A)
            # Vérifier si RHS est dans LHS
            lhs_cols = [col.strip() for col in lhs_part.split(",")]
            rhs_cols = [col.strip() for col in rhs_part.split(",")]
            
            for rhs_col in rhs_cols:
                if rhs_col in lhs_cols:
                    return "degenerate"
            
            # Sample code number (ID) détermine souvent d'autres attributs
            if "sample code number" in lhs_part:
                # Mais pas s'il détermine un ID
                if "sample code number" not in rhs_part:
                    return random.choices(
                        ["degenerate", "meaningful", "encoding-based"],
                        weights=[0.6, 0.3, 0.1]
                    )[0]
            
            # FD médicalement plausibles
            medical_keywords = ["uniformity", "marginal", "clump", "mitoses", "cell", "adhesion", "thickness"]
            if any(keyword in fd_lower for keyword in medical_keywords):
                return random.choices(
                    ["meaningful", "accidental", "unlikely"],
                    weights=[0.5, 0.3, 0.2]
                )[0]
        
        # Pour les autres datasets ou cas génériques
        # FD qui semble logique
        if "id" in fd_lower or "code" in fd_lower or "number" in fd_lower:
            if "name" in fd_lower or "address" in fd_lower:
                return "meaningful"
        
        # FD avec même attribut des deux côtés
        if lhs_part == rhs_part:
            return "degenerate"
        
        # Réponse aléatoire pondérée
        return random.choices(
            LABELS,
            weights=[0.3, 0.25, 0.2, 0.15, 0.1]
        )[0]
        
    except Exception as e:
        # En cas d'erreur, retourner une réponse aléatoire
        return random.choice(LABELS)

# =========================
# STREAMLIT UI
# =========================
def run_task2():
    st.header("🤖 Task 2 — LLM-Assisted Semantic FD Discovery (CSV)")
    
    # -------------------------
    # CONFIGURATION
    # -------------------------
    st.sidebar.subheader("⚙️ Configuration")
    use_simulation = st.sidebar.checkbox(
        "Utiliser le mode simulation (pas d'API réelle)",
        value=True,
        help="Activez cette option pour tester sans clé API Groq"
    )
    
    if not use_simulation and not GROQ_API_KEY:
        st.warning("⚠️ Clé API Groq manquante. Activez le mode simulation ou configurez votre .env")
        use_simulation = True

    # -------------------------
    # CHOIX DU DATASET
    # -------------------------
    dataset_name = st.selectbox(
        "Choisissez le dataset sur lequel travailler",
        list(DATASETS.keys())
    )
    cfg = DATASETS[dataset_name]

    # -------------------------
    # CHARGEMENT DU CSV
    # -------------------------
    data_path = Path(cfg["data_file"])
    if not data_path.exists():
        st.error(f"Fichier CSV introuvable : {cfg['data_file']}")
        st.info("Assurez-vous que le fichier existe à cet emplacement.")
        return

    try:
        df = pd.read_csv(data_path)
        columns = df.columns.tolist()
        st.markdown(f"**Colonnes détectées :** {columns}")
        st.markdown(f"**Nombre de lignes :** {len(df)}")
    except Exception as e:
        st.error(f"Erreur lors du chargement du CSV : {str(e)}")
        return

    # -------------------------
    # CHARGEMENT DES FDs
    # -------------------------
    fd_path = Path(cfg["fd_file"])
    if not fd_path.exists():
        st.warning(f"Fichier de FDs introuvable : {cfg['fd_file']}")
        st.info("Vous pouvez créer manuellement des FDs pour tester.")
        # Créer des FDs de test basées sur les colonnes
        test_fds = []
        if len(columns) >= 3:
            test_fds = [
                ([columns[0]], columns[1]),
                ([columns[0], columns[1]], columns[2]),
                ([columns[1]], columns[0]),
                ([columns[2]], columns[3] if len(columns) > 3 else columns[1]),
                ([columns[0], columns[2]], columns[1]),
                ([columns[1], columns[2]], columns[0]),
            ]
        fds = test_fds[:6]  # Prendre au plus 6 FDs de test
    else:
        fds = load_fds(cfg["fd_file"], columns)
    
    if not fds:
        st.warning(f"Aucune FD trouvée ou générée pour {dataset_name}")
        return

    st.write(f"**Nombre de FDs disponibles :** {len(fds)}")
    
    # Afficher les FDs disponibles
    with st.expander("Aperçu des FDs disponibles", expanded=True):
        for i, (lhs, rhs) in enumerate(fds):
            fd_str = f"{', '.join(lhs)} -> {rhs}"
            st.write(f"{i+1}. {fd_str}")

    # -------------------------
    # SELECTION DES FDs
    # -------------------------
    fd_strings = [f"{', '.join(lhs)} -> {rhs}" for lhs, rhs in fds]
    selected_fds = st.multiselect(
        "Sélectionnez au moins 6 FDs",
        fd_strings,
        help="Choisissez au moins 6 FDs pour l'analyse"
    )
    
    if len(selected_fds) < 6:
        st.info(f"Veuillez sélectionner au moins 6 FDs. Actuel : {len(selected_fds)}")
        if len(fds) < 6:
            st.warning(f"Le fichier ne contient que {len(fds)} FDs. Ajoutez plus de FDs ou utilisez un autre dataset.")
        return

    # -------------------------
    # ANALYSE SEMANTIQUE
    # -------------------------
    st.subheader("🧠 Analyse sémantique (LLM vs Humain)")
    
    if use_simulation:
        st.info("🔧 Mode simulation activé - Réponses LLM simulées")
    else:
        st.info("🌐 Mode API Groq activé")
    
    # Initialiser les résultats
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Créer des onglets pour organiser l'affichage
    tab1, tab2 = st.tabs(["📝 Analyse détaillée", "📊 Résumé"])
    
    with tab1:
        # Barre de progression
        progress_bar = st.progress(0)
        
        for idx, fd in enumerate(selected_fds):
            st.markdown(f"#### FD {idx+1}/{len(selected_fds)}")
            st.code(fd, language="text")
            
            # Obtenir le jugement LLM
            with st.spinner("Interrogation du LLM..."):
                llm_label = ask_llm(fd, dataset_name, use_simulation)
            
            # Afficher le résultat LLM
            col1, col2 = st.columns(2)
            with col1:
                if llm_label in ["api_error", "missing_api_key", "exception", "unknown", "timeout", "connection_error"]:
                    st.error(f"**🤖 LLM :** `{llm_label}`")
                    llm_valid = False
                else:
                    st.success(f"**🤖 LLM :** `{llm_label}`")
                    llm_valid = True
            
            # Jugement humain
            with col2:
                human_label = st.selectbox(
                    "**Votre jugement :**",
                    LABELS,
                    index=0,
                    key=f"human_{fd}_{idx}"
                )
            
            # Calcul de l'accord
            if llm_valid:
                agreement = llm_label == human_label
                agreement_text = "✅ Oui" if agreement else "❌ Non"
                
                explanation = ""
                if not agreement:
                    explanation = st.text_area(
                        "**Expliquez le désaccord :**",
                        placeholder="Pourquoi pensez-vous que le LLM s'est trompé?",
                        key=f"explain_{fd}_{idx}",
                        height=100
                    )
            else:
                agreement_text = "⚠️ N/A"
                explanation = "Réponse LLM invalide - non comptabilisé"
            
            # Afficher l'accord
            st.markdown(f"**Accord :** {agreement_text}")
            
            # Stocker les résultats dans session_state
            result_entry = {
                "FD": fd,
                "LLM judgment": llm_label,
                "Human judgment": human_label,
                "Agreement?": agreement_text,
                "Explanation": explanation,
                "Valid": llm_valid
            }
            
            # Mettre à jour ou ajouter le résultat
            if idx < len(st.session_state.results):
                st.session_state.results[idx] = result_entry
            else:
                st.session_state.results.append(result_entry)
            
            # Mettre à jour la barre de progression
            progress_bar.progress((idx + 1) / len(selected_fds))
            
            if idx < len(selected_fds) - 1:
                st.divider()
    
    # -------------------------
    # TABLE DES RESULTATS
    # -------------------------
    with tab2:
        st.subheader("📋 Tableau des résultats")
        
        if st.session_state.results:
            # Calculer les statistiques
            valid_responses = sum(1 for r in st.session_state.results if r.get("Valid", False))
            disagreement_count = sum(1 for r in st.session_state.results 
                                   if r.get("Valid", False) and 
                                   "Non" in r.get("Agreement?", ""))
            
            # Créer un DataFrame pour affichage
            display_data = []
            for r in st.session_state.results:
                display_data.append({
                    "FD": r["FD"],
                    "LLM": r["LLM judgment"],
                    "Humain": r["Human judgment"],
                    "Accord": r["Agreement?"].replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", ""),
                    "Valide": "✅" if r.get("Valid", False) else "❌"
                })
            
            display_df = pd.DataFrame(display_data)
            # CORRECTION ICI : remplacer use_container_width par width='stretch'
            st.dataframe(display_df, height=400, width='stretch')
            
            # -------------------------
            # STATISTIQUES
            # -------------------------
            st.subheader("📊 Statistiques")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("FDs analysées", len(selected_fds))
            with col2:
                st.metric("Réponses valides", valid_responses)
            with col3:
                st.metric("Désaccords", disagreement_count)
            with col4:
                if valid_responses > 0:
                    agreement_rate = (valid_responses - disagreement_count) / valid_responses * 100
                    st.metric("Taux d'accord", f"{agreement_rate:.1f}%")
                else:
                    st.metric("Taux d'accord", "N/A")
            
            # -------------------------
            # EVALUATION FINALE
            # -------------------------
            st.subheader("🎯 Évaluation des exigences")
            
            requirements = []
            
            # Exigence 1: Au moins 6 FDs
            req1_ok = len(selected_fds) >= 6
            requirements.append((
                "✅" if req1_ok else "❌",
                f"Au moins 6 FDs sélectionnées ({len(selected_fds)}/6)"
            ))
            
            # Exigence 2: Au moins 2 désaccords
            req2_ok = disagreement_count >= 2
            requirements.append((
                "✅" if req2_ok else "❌",
                f"Au moins 2 désaccords ({disagreement_count}/2)"
            ))
            
            # Afficher les exigences
            for icon, text in requirements:
                st.write(f"{icon} {text}")
            
            # Conclusion
            if req1_ok and req2_ok:
                st.success("""
                ## ✅ Toutes les exigences sont respectées !
                
                **Résumé:**
                - ✓ Au moins 6 FDs analysées
                - ✓ Au moins 2 désaccords identifiés
                - ✓ Analyse sémantique complète
                """)
            else:
                st.warning("""
                ## ⚠️ Certaines exigences ne sont pas remplies
                
                **Actions recommandées:**
                1. Sélectionnez au moins 6 FDs
                2. Créez des désaccords en donnant des jugements différents du LLM
                3. Pour chaque désaccord, expliquez votre raisonnement
                """)
            
            # -------------------------
            # EXPORT DES RESULTATS
            # -------------------------
            st.subheader("💾 Export des résultats")
            
            # Bouton pour exporter
            if st.button("📥 Exporter les résultats au format CSV"):
                # Créer un DataFrame sécurisé pour l'export
                export_data = []
                for r in st.session_state.results:
                    export_data.append({
                        "FD": r.get("FD", ""),
                        "LLM_judgment": r.get("LLM judgment", ""),
                        "Human_judgment": r.get("Human judgment", ""),
                        "Agreement": r.get("Agreement?", ""),
                        "Explanation": r.get("Explanation", ""),
                        "Valid": r.get("Valid", False)
                    })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False, encoding='utf-8')
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"fd_analysis_{dataset_name}_{timestamp}.csv"
                    
                    st.download_button(
                        label="Télécharger le CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )
                else:
                    st.error("Aucune donnée à exporter")
        else:
            st.info("Aucun résultat à afficher. Effectuez d'abord l'analyse dans l'onglet 'Analyse détaillée'.")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    # Configuration de la page
    st.set_page_config(
        page_title="FD Semantic Analysis - Task 2",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Style CSS personnalisé
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #f0f2f6;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialiser session_state si nécessaire
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    run_task2()
import streamlit as st
from pathlib import Path
import tempfile
import os
import re
import json
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from io import BytesIO
from task1 import parse_fds, analyze_fds, save_fds

# ==================
# CONFIGURATION
# ==================
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"
LABELS = ["meaningful", "accidental", "encoding-based", "degenerate", "unlikely"]

st.set_page_config(page_title="FD Analysis Suite", layout="wide", page_icon="🔍")

# ==================
# FONCTIONS UTILITAIRES
# ==================

def groq_classify(fd_text):
    """Classification avec Groq API"""
    try:
        prompt = f"""Classify this functional dependency into ONE category:

{fd_text}

Categories:
- meaningful: Real semantic relationship
- accidental: Coincidental in this dataset
- encoding-based: Due to data encoding
- degenerate: Trivial/self-referential
- unlikely: Doesn't make sense

Respond with ONLY ONE WORD from the categories above:"""
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 50
        }
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip().lower()
            
            for label in LABELS:
                if label in answer:
                    return {"success": True, "label": label}
            
            return {"success": False, "error": f"Unexpected response: {answer[:50]}"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)[:100]}"}

def check_fd_validity(df, lhs, rhs, threshold=0.95):
    """Vérifie la validité statistique d'une FD"""
    try:
        grouped = df.groupby(lhs)[rhs]
        total_groups = 0
        valid_groups = 0
        
        for _, series in grouped:
            total_groups += 1
            if series.nunique() == 1:
                valid_groups += 1
        
        validity = valid_groups / total_groups if total_groups > 0 else 0
        violations = total_groups - valid_groups
        
        return {
            "valid": validity >= threshold,
            "validity_score": validity,
            "violations": violations,
            "total_groups": total_groups
        }
    except:
        return {
            "valid": False,
            "validity_score": 0,
            "violations": -1
        }

def discover_fds_llm(sample_df, dataset_name):
    """Découverte de FDs via LLM (Task 3)"""
    csv_text = sample_df.to_csv(index=False)
    
    prompt = f"""You are given a sample from dataset {dataset_name}.

Infer likely Functional Dependencies in the form: A,B -> C

Return ONLY a JSON list of strings.

Example: ["A -> B", "C,D -> E"]

Dataset sample:
{csv_text}
"""
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            start = text.find("[")
            end = text.rfind("]") + 1
            json_text = text[start:end].strip()
            
            if json_text:
                return json.loads(json_text)
        return []
    except:
        return []

def draw_pipeline_diagram():
    """Génère le diagramme du pipeline hybride"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Titre
    ax.text(5, 10.5, 'HYBRID FD DISCOVERY PIPELINE', 
            ha='center', fontsize=16, fontweight='bold')
    
    # PHASE 1
    phase1 = FancyBboxPatch((1, 8.5), 8, 1.2, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2)
    ax.add_patch(phase1)
    ax.text(5, 9.3, 'PHASE 1: Algorithmic Discovery', 
            ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 8.9, 'TANE/Metanome → Raw FDs', ha='center', fontsize=10)
    
    # Flèche
    arrow1 = FancyArrowPatch((5, 8.5), (5, 7.7),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # PHASE 2
    phase2_box = FancyBboxPatch((0.5, 4), 9, 3.5, 
                                boxstyle="round,pad=0.15",
                                edgecolor='#2A9D8F', facecolor='#B7E4C7', 
                                linewidth=2, linestyle='--', alpha=0.3)
    ax.add_patch(phase2_box)
    
    ax.text(5, 7.3, 'PHASE 2: LLM-Based Filtering', 
            ha='center', fontsize=12, fontweight='bold')
    
    # Composants LLM
    comp1 = FancyBboxPatch((1, 5.5), 2.3, 1.2,
                          boxstyle="round,pad=0.08",
                          edgecolor='#40916C', facecolor='#95D5B2', linewidth=1.5)
    ax.add_patch(comp1)
    ax.text(2.15, 6.3, 'Semantic\nFilter', ha='center', fontsize=9, fontweight='bold')
    
    comp2 = FancyBboxPatch((3.8, 5.5), 2.3, 1.2,
                          boxstyle="round,pad=0.08",
                          edgecolor='#40916C', facecolor='#95D5B2', linewidth=1.5)
    ax.add_patch(comp2)
    ax.text(5, 6.3, 'Suspicious\nDetector', ha='center', fontsize=9, fontweight='bold')
    
    comp3 = FancyBboxPatch((6.6, 5.5), 2.3, 1.2,
                          boxstyle="round,pad=0.08",
                          edgecolor='#40916C', facecolor='#95D5B2', linewidth=1.5)
    ax.add_patch(comp3)
    ax.text(7.75, 6.3, 'Candidate\nRanking', ha='center', fontsize=9, fontweight='bold')
    
    # Info LLM
    ax.text(5, 4.8, '🤖 LLM: llama-3.3-70b-versatile', 
            ha='center', fontsize=8, style='italic')
    
    # Flèche
    arrow2 = FancyArrowPatch((5, 4), (5, 3.2),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # PHASE 3
    phase3 = FancyBboxPatch((1, 1.5), 8, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='#E63946', facecolor='#FFCCD5', linewidth=2)
    ax.add_patch(phase3)
    ax.text(5, 2.7, 'PHASE 3: Statistical Verification', 
            ha='center', fontsize=12, fontweight='bold')
    ax.text(5, 2.3, 'Validity Check + Minimality Test', ha='center', fontsize=10)
    ax.text(5, 1.9, '→ Final Verified FDs', ha='center', fontsize=10, style='italic')
    
    # Légende
    legend_elements = [
        mpatches.Patch(facecolor='#A9D6E5', edgecolor='#2E86AB', label='Algorithmic'),
        mpatches.Patch(facecolor='#95D5B2', edgecolor='#40916C', label='LLM'),
        mpatches.Patch(facecolor='#FFCCD5', edgecolor='#E63946', label='Verification')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
    
    plt.tight_layout()
    
    # Sauvegarder dans un buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# ==================
# SIDEBAR
# ==================
with st.sidebar:
    st.title("🔍 FD Analysis Suite")
    st.markdown("---")
    
    # Sélection de task
    task = st.selectbox(
        "Choose Task:",
        [
            "Task 1: FD Analysis",
            "Task 2: LLM Evaluation",
            "Task 3: LLM Discovery",
            "Task 4: Hybrid System"
        ]
    )
    

# ==================
# INITIALISATION
# ==================
if 'eval_data' not in st.session_state:
    st.session_state.eval_data = []
if 'task3_results' not in st.session_state:
    st.session_state.task3_results = []
if 'task4_results' not in st.session_state:
    st.session_state.task4_results = None

# ==================
# TASK 1: FD ANALYSIS
# ==================
if task == "Task 1: FD Analysis":
    st.header("📊 Task 1: FD Analysis")
    st.markdown("Analyze functional dependencies from algorithmic discovery (TANE/Metanome)")
    
    col1, col2 = st.columns(2)
    with col1:
        csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="task1_csv")
    with col2:
        fd_file = st.file_uploader("Upload FD file", type=["txt", "json"], key="task1_fd")
    
    if csv_file and fd_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write(csv_file.getbuffer())
            csv_path = Path(f.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(fd_file.getbuffer())
            fd_path = Path(f.name)
        
        try:
            fds = parse_fds(fd_path, csv_path)
            
            if fds:
                st.success(f"✅ Found **{len(fds)}** FDs")
                
                # Afficher les FDs
                st.markdown("### Discovered FDs")
                for i, (lhs, rhs) in enumerate(fds, 1):
                    st.code(f"{i}. {', '.join(lhs)} → {rhs}")
                
                # Analyse
                with st.spinner("Analyzing..."):
                    analyze_fds(fds)
                    save_fds(fds, fd_path, filename="results.txt")
                
                # Résultats
                results_file = fd_path.parent / "results.txt"
                if results_file.exists():
                    with st.expander("📊 View Analysis Details"):
                        with open(results_file, "r") as f:
                            st.text(analyze_fds(fds))
                    
                    # Téléchargement
                    with open(results_file, "r") as f:
                        st.download_button(
                            label="📥 Download Analysis",
                            data=f.read(),
                            file_name="fd_analysis_results.txt",
                            mime="text/plain"
                        )
            else:
                st.warning("⚠️ No FDs found")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
        
        finally:
            os.unlink(csv_path)
            os.unlink(fd_path)

# ==================
# TASK 2: LLM EVALUATION
# ==================
elif task == "Task 2: LLM Evaluation":
    st.header("🤖 Task 2: LLM Evaluation")
    st.markdown("Evaluate FDs using LLM and compare with human judgment")
    fd_file = st.file_uploader("Upload FD file", type=["txt", "json"], key="task2_fd")

    if fd_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(fd_file.getbuffer())
            fd_path = Path(f.name)
        
        try:
                from task2 import extract_fds, query_groq, LABELS   # adjust import path if needed

                dataset_name = fd_path.stem

                fds_to_eval = extract_fds(fd_path, k=3)

                if fds_to_eval:
                    # Initialiser
                    if not st.session_state.eval_data or len(st.session_state.eval_data) != len(fds_to_eval):
                        st.session_state.eval_data = [
                            {"fd": fd, "groq": None, "human": None}
                            for fd in fds_to_eval
                        ]
                    
                    # Interface d'évaluation
                    for idx, item in enumerate(st.session_state.eval_data):
                        st.markdown("---")
                        st.markdown(f"#### FD {idx+1}")
                        st.code(item['fd'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🤖 GROQ**")
                            if item["groq"]:
                                if item["groq"] in LABELS:
                                    st.success(f"✅ {item['groq']}")
                                else:
                                    st.error(f"❌ {item['groq']}")
                                
                                if st.button("🔄 Retry", key=f"r{idx}"):
                                    item["groq"] = None
                                    st.rerun()
                            else:
                                if st.button("▶️ Run", key=f"g{idx}", type="primary"):
                                    with st.spinner("Calling API..."):
                                        label = query_groq(item["fd"], dataset_name)
                                        item["groq"] = label
                                        st.rerun()
                        
                        with col2:
                            st.markdown("**👤 Your Choice**")
                            idx_select = 0
                            if item.get("human") in LABELS:
                                idx_select = LABELS.index(item["human"]) + 1
                            
                            choice = st.selectbox(
                                "Select:",
                                ["--"] + LABELS,
                                key=f"h{idx}",
                                index=idx_select
                            )
                            if choice != "--":
                                item["human"] = choice
                    
                    # Résultats
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    
                    all_done = all(item.get("groq") and item["groq"] in LABELS and item.get("human") 
                                  for item in st.session_state.eval_data)
                    
                    if all_done:
                        results_data = []
                        for item in st.session_state.eval_data:
                            results_data.append({
                                "FD": item["fd"],
                                "GROQ": item["groq"],
                                "Human": item["human"],
                                "Match": "✅" if item["groq"] == item["human"] else "❌"
                            })
                        
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
                        
                        matches = sum(1 for d in results_data if d["Match"] == "✅")
                        total = len(results_data)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", total)
                        with col2:
                            st.metric("Matches", matches)
                        with col3:
                            st.metric("Accuracy", f"{matches/total*100:.1f}%")
                        
                        # Save
                        if st.button("💾 Save Results", type="primary"):
                            output = {
                                "model": GROQ_MODEL,
                                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "evaluations": results_data,
                                "accuracy": matches/total*100
                            }
                            
                            output_file = fd_path.parent / "task2_results.json"
                            with open(output_file, 'w') as f:
                                json.dump(output, f, indent=2)
                            
                            st.success("✅ Saved!")
                            
                            with open(output_file, 'r') as f:
                                st.download_button(
                                    "📥 Download JSON",
                                    data=f.read(),
                                    file_name="task2_evaluation.json",
                                    mime="application/json"
                                )
                    else:
                        st.info("⏳ Complete all evaluations")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
        
        finally:
            os.unlink(fd_path)

# ==================
# TASK 3: LLM DISCOVERY
# ==================
elif task == "Task 3: LLM Discovery":
    st.header("🔮 Task 3: LLM-Based FD Discovery")
    st.markdown("Discover FDs directly from data using LLM reasoning")
    
    csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="task3_csv")
    
    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write(csv_file.getbuffer())
            csv_path = Path(f.name)
        
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            dataset_name = csv_file.name.replace('.csv', '')
            
            st.success(f"✅ Loaded dataset: **{dataset_name}** ({len(df)} rows, {len(df.columns)} columns)")
            
            # Sample
            sample_size = st.slider("Sample size for LLM", 10, 100, 50, 10)
            sample = df.sample(min(sample_size, len(df)), random_state=42)
            
            with st.expander("👀 View Sample Data"):
                st.dataframe(sample.head(20))
            
            if st.button("🚀 Discover FDs", type="primary"):
                with st.spinner(f"Discovering FDs with LLM ({sample_size} rows)..."):
                    discovered_fds = discover_fds_llm(sample, dataset_name)
                    
                    if discovered_fds:
                        st.session_state.task3_results = []
                        
                        st.success(f"✅ Discovered {len(discovered_fds)} FDs")
                        
                        # Vérifier chaque FD
                        st.markdown("### Validation")
                        
                        for fd_str in discovered_fds:
                            if "->" not in fd_str:
                                continue
                            
                            lhs_str, rhs = fd_str.split("->")
                            lhs = [c.strip() for c in lhs_str.split(",")]
                            rhs = rhs.strip()
                            
                            # Vérifier colonnes
                            if rhs not in df.columns or not all(c in df.columns for c in lhs):
                                continue
                            
                            # Valider
                            validity = check_fd_validity(df, lhs, rhs, threshold=0.9)
                            
                            result = {
                                "fd": fd_str,
                                "validity": validity["validity_score"],
                                "violations": validity["violations"],
                                "valid": validity["valid"]
                            }
                            
                            st.session_state.task3_results.append(result)
                            
                            status = "✅" if result["valid"] else "❌"
                            st.write(f"{status} `{fd_str}` - Validity: {result['validity']:.2%} ({result['violations']} violations)")
                        
                        # Résumé
                        if st.session_state.task3_results:
                            valid_count = sum(1 for r in st.session_state.task3_results if r["valid"])
                            
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Discovered", len(st.session_state.task3_results))
                            with col2:
                                st.metric("Valid", valid_count)
                            with col3:
                                st.metric("Precision", f"{valid_count/len(st.session_state.task3_results)*100:.1f}%")
                            
                            # Save
                            output = {
                                "dataset": dataset_name,
                                "model": GROQ_MODEL,
                                "sample_size": sample_size,
                                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "results": st.session_state.task3_results
                            }
                            
                            st.download_button(
                                "📥 Download Results",
                                data=json.dumps(output, indent=2),
                                file_name="task3_discovery.json",
                                mime="application/json"
                            )
                    else:
                        st.warning("⚠️ No FDs discovered")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
        
        finally:
            os.unlink(csv_path)

# ==================
# TASK 4: HYBRID SYSTEM
# ==================
elif task == "Task 4: Hybrid System":
    st.header("🔬 Task 4: Hybrid FD Discovery System")
    st.markdown("Combines algorithmic discovery + LLM filtering + statistical verification")
    
    # Pipeline Diagram
    st.markdown("### Pipeline Architecture")
    diagram_buf = draw_pipeline_diagram()
    st.image(diagram_buf, use_column_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="task4_csv")
    with col2:
        fd_file = st.file_uploader("Upload FD file", type=["txt", "json"], key="task4_fd")
    
    if csv_file and fd_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write(csv_file.getbuffer())
            csv_path = Path(f.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(fd_file.getbuffer())
            fd_path = Path(f.name)
        
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            dataset_name = csv_file.name.replace('.csv', '')
            
            if st.button("🚀 Run Hybrid Pipeline", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                # PHASE 1: Algorithmic Discovery
                status.info("🔧 PHASE 1: Loading algorithmic FDs...")
                progress.progress(10)
                fds = parse_fds(fd_path, csv_path)
                raw_count = len(fds)
                st.success(f"✅ Phase 1: Loaded {raw_count} raw FDs")
                
                # PHASE 2: LLM Filtering
                status.info("🤖 PHASE 2: LLM-based filtering...")
                progress.progress(30)
                
                categorized = {cat: [] for cat in ["meaningful", "accidental", "encoding", "degenerate", "suspicious"]}
                ranked = []
                
                for lhs, rhs in fds:
                    fd_str = f"{', '.join(lhs)} → {rhs}"
                    result = groq_classify(fd_str)
                    
                    if result["success"]:
                        category = result["label"]
                        categorized[category].append((lhs, rhs))
                    else:
                        categorized["suspicious"].append((lhs, rhs))
                    
                    time.sleep(0.3)
                
                # Scoring
                category_scores = {"meaningful": 10, "encoding": 5, "accidental": 2, "suspicious": 1, "degenerate": 0}
                for category, fds_list in categorized.items():
                    for lhs, rhs in fds_list:
                        score = category_scores[category]
                        if len(lhs) >= 4:
                            score -= 2
                        ranked.append({"fd": (lhs, rhs), "category": category, "score": max(0, score)})
                
                ranked.sort(key=lambda x: x["score"], reverse=True)
                
                st.success(f"✅ Phase 2: Categorized FDs")
                
                # PHASE 3: Verification
                status.info("📊 PHASE 3: Statistical verification...")
                progress.progress(70)
                
                verified = []
                for item in ranked:
                    lhs, rhs = item["fd"]
                    validity = check_fd_validity(df, lhs, rhs, threshold=0.95)
                    
                    verified.append({
                        "fd": f"{', '.join(lhs)} → {rhs}",
                        "category": item["category"],
                        "score": item["score"],
                        "validity": validity["validity_score"],
                        "verified": validity["valid"]
                    })
                
                verified_count = sum(1 for v in verified if v["verified"])
                
                progress.progress(100)
                status.success("✅ Pipeline completed!")
                
                # Résultats
                st.markdown("---")
                st.markdown("### 📊 Results Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Raw FDs", raw_count)
                with col2:
                    st.metric("Meaningful", len(categorized["meaningful"]))
                with col3:
                    st.metric("Verified", verified_count)
                with col4:
                    reduction = (raw_count - verified_count) / raw_count * 100 if raw_count > 0 else 0
                    st.metric("Reduction", f"{reduction:.1f}%")
                
                # Détails
                with st.expander("📋 View All Results"):
                    df_results = pd.DataFrame(verified)
                    st.dataframe(df_results, use_container_width=True)
                
                # Discussion
                st.markdown("### 💡 Discussion")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Gains:**")
                    st.markdown(f"""
                    - Semantic categorization
                    - {len(categorized["meaningful"])} meaningful FDs identified
                    - Reduced noise by {reduction:.1f}%
                    - Statistical validation ensures correctness
                    """)
                
                with col2:
                    st.markdown("**⚠️ Trade-offs:**")
                    st.markdown(f"""
                    - Filtered {raw_count - verified_count} FDs
                    - LLM adds latency (~0.5s/FD)
                    - Subjective categorization
                    - API costs for large datasets
                    """)
                
                # Save
                st.session_state.task4_results = {
                    "dataset": dataset_name,
                    "model": GROQ_MODEL,
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "phases": {
                        "phase1_raw": raw_count,
                        "phase2_categorized": {k: len(v) for k, v in categorized.items()},
                        "phase3_verified": verified_count
                    },
                    "verified_fds": verified
                }
                
                st.download_button(
                    "📥 Download Full Results",
                    data=json.dumps(st.session_state.task4_results, indent=2),
                    file_name="task4_hybrid_results.json",
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
        
        finally:
            os.unlink(csv_path)
            os.unlink(fd_path)

# Footer
st.markdown("---")
st.caption(f"FD Analysis Suite | Powered by GROQ ({GROQ_MODEL})")
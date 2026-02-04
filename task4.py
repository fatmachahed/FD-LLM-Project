# t4task.py - VERSION CORRIGÉE
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from dotenv import load_dotenv
from t1task import parse_fds, analyze_fds

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL = "llama-3.3-70b-versatile"

# =============================
# PHASE 1: ALGORITHMIC DISCOVERY
# =============================
def load_algorithmic_fds(fd_path, csv_path):
    """Charge les FDs découvertes par l'algorithme"""
    print("\n=== PHASE 1: Algorithmic Discovery ===")
    fds = parse_fds(fd_path, csv_path)
    print(f"✅ Loaded {len(fds)} FDs from algorithmic discovery")
    return fds

# =============================
# PHASE 2: LLM-BASED FILTERING
# =============================

class LLMFilter:
    """Composant LLM pour filtrer et classer les FDs"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, on_bad_lines='skip')
        self.columns = list(self.df.columns)
        self.sample = self.df.head(20)
        
    def call_llm(self, prompt, max_retries=2):
        """Appel générique à l'API Groq"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
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
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"⚠️ API error {response.status_code}, attempt {attempt+1}/{max_retries}")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Exception: {e}, attempt {attempt+1}/{max_retries}")
                time.sleep(2)
        
        return None
    
    def semantic_filter(self, fds):
        """Filtre sémantique : détecte les FDs qui ont du sens"""
        print("\n--- LLM Component 1: Semantic Filter ---")
        
        sample_data = self.sample.to_string(max_rows=10)
        
        categorized = {
            "meaningful": [],
            "accidental": [],
            "encoding": [],
            "degenerate": [],
            "suspicious": []
        }
        
        for lhs, rhs in fds:
            fd_str = f"{', '.join(lhs)} → {rhs}"
            
            prompt = f"""Given this dataset sample:

{sample_data}

Classify this functional dependency: {fd_str}

Categories:
- meaningful: Real-world semantic relationship
- accidental: Statistical coincidence
- encoding: Due to data encoding/format
- degenerate: Trivial (self-reference)
- suspicious: Likely incorrect

Respond with ONLY the category name (one word):"""
            
            response = self.call_llm(prompt)
            
            if response:
                category = response.lower().strip()
                
                if "meaningful" in category:
                    categorized["meaningful"].append((lhs, rhs))
                elif "accidental" in category:
                    categorized["accidental"].append((lhs, rhs))
                elif "encoding" in category:
                    categorized["encoding"].append((lhs, rhs))
                elif "degenerate" in category:
                    categorized["degenerate"].append((lhs, rhs))
                else:
                    categorized["suspicious"].append((lhs, rhs))
                
                print(f"  {fd_str}: {category}")
            else:
                categorized["suspicious"].append((lhs, rhs))
                print(f"  {fd_str}: API_ERROR")
            
            time.sleep(0.5)
        
        return categorized
    
    def detect_suspicious(self, fds):
        """Détecteur de FDs suspectes"""
        print("\n--- LLM Component 2: Suspicious Detector ---")
        
        suspicious_patterns = []
        
        for lhs, rhs in fds:
            reasons = []
            
            if len(lhs) >= 4:
                reasons.append(f"Large LHS ({len(lhs)} attributes)")
            
            if len(lhs) == 1 and any(id_term in lhs[0].lower() for id_term in ['id', 'index', 'key']):
                reasons.append("ID-based attribute")
            
            if rhs in lhs:
                reasons.append("Self-referential")
            
            if reasons:
                suspicious_patterns.append({
                    "fd": (lhs, rhs),
                    "reasons": reasons
                })
        
        print(f"  Found {len(suspicious_patterns)} suspicious FDs")
        return suspicious_patterns
    
    def rank_candidates(self, categorized):
        """Classement des FDs par utilité"""
        print("\n--- LLM Component 3: Candidate Ranking ---")
        
        category_scores = {
            "meaningful": 10,
            "encoding": 5,
            "accidental": 2,
            "suspicious": 1,
            "degenerate": 0
        }
        
        ranked = []
        
        for category, fds in categorized.items():
            base_score = category_scores.get(category, 0)
            
            for lhs, rhs in fds:
                score = base_score
                
                if len(lhs) >= 4:
                    score -= 2
                
                if len(lhs) == 1:
                    score += 1
                
                ranked.append({
                    "fd": (lhs, rhs),
                    "category": category,
                    "score": max(0, score)
                })
        
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"  Ranked {len(ranked)} FDs")
        return ranked

# =============================
# PHASE 3: VERIFICATION AMÉLIORÉE
# =============================

class StatisticalVerifier:
    """Vérification statistique des FDs"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    def check_fd_validity(self, lhs, rhs, threshold=0.95):
        """Vérifie la validité statistique d'une FD"""
        try:
            grouped = self.df.groupby(lhs)[rhs]
            
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
        except Exception as e:
            return {
                "valid": False,
                "validity_score": 0,
                "violations": -1,
                "error": str(e)
            }
    
    def check_minimality_relaxed(self, lhs, rhs, threshold=0.95):
        """
        Vérification de minimalité RELAXÉE
        Une FD est considérée minimale si aucun sous-ensemble strict
        de LHS ne maintient une validité >= threshold
        """
        if len(lhs) <= 1:
            return True  # Toujours minimale avec 1 attribut
        
        # Vérifier tous les sous-ensembles de taille n-1
        for i in range(len(lhs)):
            sub_lhs = lhs[:i] + lhs[i+1:]
            result = self.check_fd_validity(sub_lhs, rhs, threshold)
            
            # Si un sous-ensemble a la même validité, pas minimale
            if result["validity_score"] >= threshold:
                return False
        
        return True
    
    def verify_fds(self, ranked_fds, validity_threshold=0.95, use_relaxed_minimality=True):
        """Vérifie toutes les FDs classées"""
        print("\n=== PHASE 3: Statistical Verification ===")
        print(f"  Validity threshold: {validity_threshold}")
        print(f"  Minimality check: {'RELAXED' if use_relaxed_minimality else 'STRICT'}")
        
        verified = []
        
        for item in ranked_fds:
            lhs, rhs = item["fd"]
            
            # Validité
            validity = self.check_fd_validity(lhs, rhs, validity_threshold)
            
            # Minimalité (relaxée ou stricte)
            if use_relaxed_minimality:
                minimal = self.check_minimality_relaxed(lhs, rhs, validity_threshold)
            else:
                minimal = len(lhs) == 1  # Version très stricte
            
            # Critères de vérification
            is_verified = validity["valid"]  # On n'exige PLUS la minimalité stricte
            
            result = {
                "fd": f"{', '.join(lhs)} → {rhs}",
                "category": item["category"],
                "score": item["score"],
                "validity_score": validity["validity_score"],
                "violations": validity["violations"],
                "minimal": minimal,
                "verified": is_verified
            }
            
            verified.append(result)
            
            status = "✅" if result["verified"] else "❌"
            minimal_str = "✓" if minimal else "✗"
            print(f"  {status} {result['fd']}: validity={validity['validity_score']:.2f}, minimal={minimal_str}")
        
        return verified

# =============================
# PIPELINE PRINCIPAL
# =============================

class HybridFDDiscovery:
    """Système hybride complet"""
    
    def __init__(self, csv_path, fd_path):
        self.csv_path = csv_path
        self.fd_path = fd_path
        self.dataset_name = Path(csv_path).stem
        
    def run(self, validity_threshold=0.95, use_relaxed_minimality=True):
        """Exécute le pipeline complet"""
        print(f"\n{'='*60}")
        print(f"HYBRID FD DISCOVERY: {self.dataset_name}")
        print(f"{'='*60}")
        
        # PHASE 1
        raw_fds = load_algorithmic_fds(self.fd_path, self.csv_path)
        
        if not raw_fds:
            print("❌ No FDs to process")
            return None
        
        # PHASE 2
        print("\n=== PHASE 2: LLM-Based Filtering ===")
        llm_filter = LLMFilter(self.csv_path)
        
        categorized = llm_filter.semantic_filter(raw_fds)
        suspicious = llm_filter.detect_suspicious(raw_fds)
        ranked = llm_filter.rank_candidates(categorized)
        
        # PHASE 3
        verifier = StatisticalVerifier(self.csv_path)
        verified = verifier.verify_fds(ranked, validity_threshold, use_relaxed_minimality)
        
        # Résultats
        results = {
            "dataset": self.dataset_name,
            "config": {
                "validity_threshold": validity_threshold,
                "relaxed_minimality": use_relaxed_minimality
            },
            "phases": {
                "phase1_raw_fds": len(raw_fds),
                "phase2_categorized": {k: len(v) for k, v in categorized.items()},
                "phase2_suspicious": len(suspicious),
                "phase3_verified": sum(1 for v in verified if v["verified"]),
                "phase3_valid_but_not_minimal": sum(1 for v in verified if v["verified"] and not v["minimal"])
            },
            "verified_fds": verified,
            "suspicious_patterns": suspicious
        }
        
        # Sauvegarde
        output_file = Path(self.csv_path).parent / f"{self.dataset_name}_hybrid_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Raw FDs (algorithmic): {results['phases']['phase1_raw_fds']}")
        print(f"Categorized by LLM:")
        for cat, count in results['phases']['phase2_categorized'].items():
            print(f"  - {cat}: {count}")
        print(f"Suspicious detected: {results['phases']['phase2_suspicious']}")
        print(f"Verified FDs (valid): {results['phases']['phase3_verified']}")
        print(f"  - Minimal: {results['phases']['phase3_verified'] - results['phases']['phase3_valid_but_not_minimal']}")
        print(f"  - Non-minimal but valid: {results['phases']['phase3_valid_but_not_minimal']}")
        print(f"\n✅ Results saved to: {output_file}")
        
        return results

# =============================
# ÉVALUATION
# =============================

def evaluate_hybrid_method(results):
    """Évalue les gains et pertes du système hybride"""
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    initial = results['phases']['phase1_raw_fds']
    verified = results['phases']['phase3_verified']
    
    reduction_rate = (initial - verified) / initial * 100 if initial > 0 else 0
    
    meaningful = results['phases']['phase2_categorized'].get('meaningful', 0)
    suspicious = results['phases']['phase2_suspicious']
    non_minimal = results['phases']['phase3_valid_but_not_minimal']
    
    print(f"\n📊 METRICS:")
    print(f"  Initial FDs: {initial}")
    print(f"  Verified FDs: {verified}")
    print(f"  Reduction: {reduction_rate:.1f}%")
    print(f"  Meaningful FDs: {meaningful}")
    print(f"  Suspicious FDs: {suspicious}")
    print(f"  Valid but non-minimal: {non_minimal}")
    
    print(f"\n✅ GAINS:")
    print(f"  - Kept {verified} valid FDs out of {initial}")
    print(f"  - Semantic categorization for prioritization")
    print(f"  - Detected {suspicious} suspicious patterns")
    print(f"  - Flexibility with minimality (kept {non_minimal} non-minimal but valid FDs)")
    
    print(f"\n⚠️ TRADE-OFFS:")
    print(f"  - Filtered {initial - verified} FDs ({reduction_rate:.1f}%)")
    print(f"  - Some non-minimal FDs retained for usefulness")
    print(f"  - LLM categorization is subjective but adds semantic value")

# =============================
# MAIN
# =============================

if __name__ == "__main__":
    csv_path = "data/iris/iris.csv"
    fd_path = "data/iris/iris_fds.txt"
    
    # Exécuter avec minimalité RELAXÉE
    hybrid = HybridFDDiscovery(csv_path, fd_path)
    results = hybrid.run(
        validity_threshold=0.95,
        use_relaxed_minimality=True  # CHANGEMENT CLÉ
    )
    
    if results:
        evaluate_hybrid_method(results)
import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict
from pathlib import Path

def parse_tane(fd_path, csv_path):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    columns = list(df.columns)

    mapping = {}
    fds = []

    with open(fd_path) as f:
        lines = f.readlines()

    in_columns = False
    in_results = False

    for line in lines:
        line = line.strip()

        if line.startswith("# COLUMN"):
            in_columns = True
            continue

        if line.startswith("# RESULTS"):
            in_columns = False
            in_results = True
            continue

        if line.startswith("#"):
            continue

        # Column mapping
        if in_columns and line:
            parts = line.split("\t")
            if len(parts) == 2:
                idx = int(parts[1]) - 1
                mapping[parts[1]] = columns[idx]

        # FDs
        if in_results and "->" in line:
            lhs, rhs = line.split("->")
            lhs_ids = [x.strip() for x in lhs.split(",") if x.strip()]
            rhs_id = rhs.strip()

            lhs = [mapping[x] for x in lhs_ids]
            if rhs_id not in mapping:
                continue
            rhs = mapping[rhs_id]

            fds.append((lhs, rhs))

    return fds

def build_value_to_column_mapping(csv_path):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    
    value_to_column = {}
    
    for col in df.columns:
        unique_vals = df[col].astype(str).unique()
        
        for val in unique_vals:
            if pd.isna(val) or val == 'nan' or val == '':
                continue
            
            if val not in value_to_column:
                value_to_column[val] = col
    return value_to_column



def parse_json_fd(fd_path, csv_path):

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    columns = list(df.columns)
    
    # Build mapping from values to column names
    print("  Building value-to-column mapping...")
    value_to_column = build_value_to_column_mapping(csv_path)
    print(f"  Found {len(value_to_column)} unique values across all columns")
    
    # Also create standard mappings
    col_mapping = {}
    
    # Map by index (0-based and 1-based)
    for i, col in enumerate(columns):
        col_mapping[str(i)] = col  # 0-based
        col_mapping[str(i + 1)] = col  # 1-based
        col_mapping[col] = col  # by name
    
    fds = []
    
    try:
        with open(fd_path, 'r') as f:
            content = f.read().strip()
            
        if not content:
            print("Warning: Empty FD file")
            return []
            
        # Handle both single JSON object and JSON lines format
        if content.startswith('['):
            # Array of FDs
            fd_list = json.loads(content)
        elif content.startswith('{'):
            # Could be single FD or JSONL (one per line)
            lines = content.split('\n')
            fd_list = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        fd_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Try as single object
                        try:
                            fd_list = [json.loads(content)]
                            break
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse JSON: {e}")
                            return []
        else:
            print(f"Warning: Unrecognized format (doesn't start with [ or {{)")
            return []
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    # Parse each FD
    for fd_obj in fd_list:
        if not isinstance(fd_obj, dict):
            continue
            
        if fd_obj.get('type') != 'FunctionalDependency':
            continue
        
        # Parse determinant (LHS)
        determinant = fd_obj.get('determinant', {})
        lhs_cols = determinant.get('columnIdentifiers', [])
        
        lhs_attrs = []
        for col_id in lhs_cols:
            if not isinstance(col_id, dict):
                continue
                
            col_identifier = col_id.get('columnIdentifier', '')
            
            # Strategy 1: Try as value -> find which column it belongs to
            if col_identifier in value_to_column:
                lhs_attrs.append(value_to_column[col_identifier])
            # Strategy 2: Try as column name directly
            elif col_identifier in columns:
                lhs_attrs.append(col_identifier)
            # Strategy 3: Try as index
            elif col_identifier in col_mapping:
                lhs_attrs.append(col_mapping[col_identifier])
            # Strategy 4: Case-insensitive column name match
            else:
                col_lower = col_identifier.lower()
                found = False
                for col in columns:
                    if col.lower() == col_lower:
                        lhs_attrs.append(col)
                        found = True
                        break
                if not found:
                    print(f"Warning: Could not map LHS identifier '{col_identifier}'")
        
        # Parse dependant (RHS)
        dependant = fd_obj.get('dependant', {})
        if not isinstance(dependant, dict):
            continue
            
        rhs_identifier = dependant.get('columnIdentifier', '')
        
        # Try to find matching column for RHS (same strategies)
        rhs_attr = None
        
        # Strategy 1: Try as value
        if rhs_identifier in value_to_column:
            rhs_attr = value_to_column[rhs_identifier]
        # Strategy 2: Try as column name
        elif rhs_identifier in columns:
            rhs_attr = rhs_identifier
        # Strategy 3: Try as index
        elif rhs_identifier in col_mapping:
            rhs_attr = col_mapping[rhs_identifier]
        # Strategy 4: Case-insensitive
        else:
            rhs_lower = rhs_identifier.lower()
            for col in columns:
                if col.lower() == rhs_lower:
                    rhs_attr = col
                    break
            if not rhs_attr:
                print(f"Warning: Could not map RHS identifier '{rhs_identifier}'")
        
        if lhs_attrs and rhs_attr:
            fds.append((lhs_attrs, rhs_attr))
    
    return fds

def detect_fd_format(fd_path):
    with open(fd_path, 'r') as f:
        first_line = f.readline().strip()
        
        # Check if it's JSON
        if first_line.startswith('{') or first_line.startswith('['):
            return 'json'
        
        # Check if it's TANE format
        if first_line.startswith('#'):
            return 'tane'
        
        return 'tane'

def parse_fds(fd_path, csv_path):
    format_type = detect_fd_format(fd_path)    
    if format_type == 'json':
        return parse_json_fd(fd_path, csv_path)
    else:
        return parse_tane(fd_path, csv_path)
    
def save_fds(fds, fd_path, filename="fd_results.txt"):
    out_dir = Path(fd_path).parent
    output_path = out_dir / filename

    suspicious = []
    plausible = []

    for lhs, rhs in fds:
        if len(lhs) == 1 and lhs[0].lower() in ["id", "index"]:
            suspicious.append(f"ID-based : {lhs[0]} -> {rhs}")

        elif len(lhs) >= 4:
            suspicious.append(f"LHS très large : {', '.join(lhs)} -> {rhs}")

        elif rhs in lhs:
            suspicious.append(f"Dégénérée : {', '.join(lhs)} -> {rhs}")

        else:
            plausible.append(f"{', '.join(lhs)} -> {rhs}")

    with open(output_path, "w") as f:
        f.write("=== SUSPICIOUS FDs ===\n")
        for fd in suspicious:
            f.write(fd + "\n")

        f.write("\n=== PLAUSIBLE FDs ===\n")
        for fd in plausible:
            f.write(fd + "\n")

    print(f"\nFDs saved to {output_path}")


def analyze_fds(fds):
    # Number of FDs
    total = len(fds)
    print("Total FDs:", total)

    # Average LHS size
    lhs_sizes = [len(lhs) for lhs, _ in fds]
    print("Average LHS size:", round(np.mean(lhs_sizes), 2))

    # Attribute frequency
    lhs_freq = Counter()
    rhs_freq = Counter()

    for lhs, rhs in fds:
        for a in lhs:
            lhs_freq[a] += 1
        rhs_freq[rhs] += 1

    print("\nTop LHS attributes:")
    for a, c in lhs_freq.most_common(10):
        print(f"  {a}: {c}")

    print("\nTop RHS attributes:")
    for a, c in rhs_freq.most_common(10):
        print(f"  {a}: {c}")

    # ID-based FDs (single attribute determines many)
    single = defaultdict(list)
    for lhs, rhs in fds:
        if len(lhs) == 1:
            single[lhs[0]].append(rhs)

    print("\nID-based FDs (>=3 dependents):")
    for k, v in single.items():
        if len(v) >= 3:
            print(f"  {k} -> {len(v)} attributes")

    # Large determinants
    print("\nLarge determinants (LHS >= 4):")
    for lhs, rhs in fds:
        if len(lhs) >= 4:
            print(f"  {lhs} -> {rhs}")

    # Suspicious FDs
    print("\nSuspicious FDs:")

    for lhs, rhs in fds:
        if len(lhs) == 1 and lhs[0].lower() in ["id", "index"]:
            print(f"ID-based : {lhs[0]} → {rhs}")

        elif len(lhs) >= 4:
            print(f"LHS très large : {', '.join(lhs)} → {rhs}")

        elif rhs in lhs:
            print(f"Dégénérée : {', '.join(lhs)} → {rhs}")


if __name__ == "__main__":

    fd_path = "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin_fds.txt"
    csv_path = "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin.csv"

    fds = parse_fds(fd_path, csv_path)

    """    
        fds=parse_fds(
        "data/nursery/nursery_fds.txt",
        "data/nursery/nursery.csv"
    )
    fds= (
        "data/pittsburgh+bridges/bridges_fds.txt",
        "data/pittsburgh+bridges/bridges.csv"
    )
    fds=(
        "data/abalone/abalone_fds.txt",
        "data/abalone/abalone.csv"
    )
    fds=(
        "data/adult/adult_fds.txt",
        "data/adult/adult.csv"
    )
    fds=(
        "data/nursery/nursery_fds.txt",
        "data/nursery/nursery.csv"
    )
    fds=(
        "data/iris/iris_fds.txt",
        "data/iris/iris.csv"
    )"""

    analyze_fds(fds)
    save_fds(fds, fd_path)

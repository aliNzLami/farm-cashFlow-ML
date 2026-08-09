# ============================================================
# Context-Aware Model Recommendation Framework
# Adapted for Corn Dataset - Using Calibrated Coefficients
# ============================================================
# Changes from original:
#   1. Dataset changed from Lending Club to integrated_corn_weekly.csv
#   2. Coefficients updated based on Corn dataset characteristics:
#      - Interpretability: 0.45 (lowered, as black-box models perform better)
#      - Robustness: 0.50 (maintained, tree-based models are naturally robust)
#      - Scalability: 0.40 (lowered, dataset is small, no need for high scalability)
#      - Representation Capacity: 0.85 (increased, non-linear patterns are key)
#   3. Model profiles updated to Corn-specific models:
#      - ARIMA, Random Forest, XGBoost, LightGBM, WaveletANN
# ============================================================

import sys
import os
import warnings
import numpy as np
import pandas as pd
import hashlib
import time
import gzip

warnings.filterwarnings('ignore')


# ============================================================
# SECTION 1: CORN DATASET LOADING (LOCAL)
# ============================================================
def load_corn_data():
    """Load Corn dataset from local path."""
    print("=" * 60)
    print("Loading Corn Dataset")
    print("=" * 60)
    
    local_paths = [
        'dataset/integrated_corn_weekly.csv',
        'integrated_corn_weekly.csv',
        'data/integrated_corn_weekly.csv'
    ]
    
    file_path = None
    for path in local_paths:
        if os.path.exists(path):
            file_path = path
            print(f"Dataset found locally at: {path}")
            break
    
    if file_path is None:
        raise FileNotFoundError("Corn dataset not found. Please ensure integrated_corn_weekly.csv is in the dataset/ directory.")
    
    return file_path


# ============================================================
# SECTION 2: COUNT TOTAL ROWS
# ============================================================
def count_rows_in_csv(file_path):
    """Count total rows in CSV without loading into memory."""
    print("   Counting total rows...")
    try:
        with open(file_path, 'r') as f:
            total = sum(1 for _ in f) - 1
            print(f"   Total rows: {total:,}")
            return total
    except Exception as e:
        print(f"   Could not count rows: {e}")
        return None


# ============================================================
# SECTION 3: INCREMENTAL INDICATOR COMPUTATION (STREAMING)
# ============================================================
def compute_indicators_streaming(file_path, chunk_size=1000):
    """Compute indicators by streaming the dataset in chunks."""
    print(f"\nProcessing full dataset using chunk-based streaming...")
    print(f"   Chunk size: {chunk_size:,} rows per chunk")
    
    total_rows = count_rows_in_csv(file_path)
    
    processed_rows = 0
    total_missing = 0
    total_cells = 0
    duplicate_hashes = set()
    duplicate_count = 0
    outlier_counts = {}
    numeric_cols = None
    categorical_cols = None
    n_numeric = 0
    n_categorical = 0
    total_features = 0
    col_cardinalities = {}
    corr_sum = 0
    corr_count = 0
    
    start_time = time.time()
    chunk_count = 0
    
    print(f"\n   Processing chunks...\n")
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        chunk_count += 1
        n_rows_chunk = len(chunk)
        n_cols_chunk = len(chunk.columns)
        processed_rows += n_rows_chunk
        
        if numeric_cols is None:
            for col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='ignore')
            
            numeric_cols = chunk.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = chunk.select_dtypes(include=['object', 'category']).columns.tolist()
            
            id_cols = [c for c in numeric_cols + categorical_cols if 'id' in c.lower() or 'date' in c.lower()]
            numeric_cols = [c for c in numeric_cols if c not in id_cols]
            categorical_cols = [c for c in categorical_cols if c not in id_cols]
            
            for col in numeric_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            n_numeric = len(numeric_cols)
            n_categorical = len(categorical_cols)
            total_features = n_numeric + n_categorical
            
            for col in numeric_cols:
                outlier_counts[col] = 0
            
            for col in categorical_cols:
                col_cardinalities[col] = set()
        else:
            for col in numeric_cols:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        
        total_missing += chunk.isnull().sum().sum()
        total_cells += n_rows_chunk * n_cols_chunk
        
        for idx, row in chunk.iterrows():
            row_str = ''.join(str(v) for v in row.values)
            row_hash = hashlib.md5(row_str.encode()).hexdigest()
            if row_hash in duplicate_hashes:
                duplicate_count += 1
            else:
                duplicate_hashes.add(row_hash)
        
        for col in numeric_cols:
            if col in chunk.columns and n_rows_chunk > 0:
                col_data = chunk[col].dropna()
                if len(col_data) > 0:
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_counts[col] += ((col_data < lower) | (col_data > upper)).sum()
        
        for col in categorical_cols:
            if col in chunk.columns:
                col_cardinalities[col].update(chunk[col].dropna().unique())
        
        if len(numeric_cols) >= 2 and n_rows_chunk > 1:
            valid_numeric = [col for col in numeric_cols if col in chunk.columns and chunk[col].dtype in ['int64', 'float64']]
            if len(valid_numeric) >= 2:
                sub_df = chunk[valid_numeric].dropna()
                if len(sub_df) > 1:
                    corr_matrix_chunk = sub_df.corr().abs()
                    upper_tri = corr_matrix_chunk.where(np.triu(np.ones(corr_matrix_chunk.shape), k=1).astype(bool))
                    corr_sum += upper_tri.mean().mean() * (len(sub_df) - 1)
                    corr_count += (len(sub_df) - 1)
        
        if total_rows is not None and total_rows > 0:
            progress = processed_rows / total_rows
            percent = progress * 100
            
            elapsed = time.time() - start_time
            if progress > 0.01:
                eta_seconds = (elapsed / progress) * (1 - progress)
                if eta_seconds > 3600:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                elif eta_seconds > 60:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"
            else:
                eta_str = "estimating..."
            
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            sys.stdout.write(f"\r   [{bar}] {percent:5.1f}%  |  {processed_rows:,} / {total_rows:,} rows  |  ETA: {eta_str}  ")
            sys.stdout.flush()
    
    print()
    print(f"   Processed {processed_rows:,} rows in {chunk_count} chunks")
    
    missing_ratio = total_missing / total_cells if total_cells > 0 else 0
    duplicate_ratio = duplicate_count / processed_rows if processed_rows > 0 else 0
    total_outliers = sum(outlier_counts.values())
    total_numeric_values = processed_rows * n_numeric
    outlier_ratio = total_outliers / total_numeric_values if total_numeric_values > 0 else 0
    feature_diversity = n_numeric / total_features if total_features > 0 else 0
    
    if categorical_cols:
        avg_cardinality = np.mean([len(col_cardinalities[col]) for col in categorical_cols])
    else:
        avg_cardinality = 0
    
    avg_correlation = corr_sum / corr_count if corr_count > 0 else 0
    rho = total_features / processed_rows if processed_rows > 0 else 0
    
    indicators = {
        'n_rows': processed_rows,
        'n_cols': n_cols_chunk,
        'n_numeric': n_numeric,
        'n_categorical': n_categorical,
        'total_features': total_features,
        'missing_ratio': missing_ratio,
        'duplicate_ratio': duplicate_ratio,
        'outlier_ratio': outlier_ratio,
        'feature_diversity': feature_diversity,
        'avg_cardinality': avg_cardinality,
        'avg_correlation': avg_correlation,
        'rho': rho,
        'dataset_size': processed_rows
    }
    
    return indicators


# ============================================================
# SECTION 4: CONTEXT VARIABLES (SCALED 0-1)
# ============================================================
def compute_context_variables(indicators, manager_expertise):
    n_rows = indicators['dataset_size']
    
    # Volume: logarithmic scaling to avoid dominance
    log_rows = np.log10(n_rows + 1)
    max_log = np.log10(10_000_000 + 1)
    V = min(1.0, max(0.0, log_rows / max_log))
    
    # Noise: combination of missing, duplicate, outlier ratios
    N_raw = (0.4 * indicators['missing_ratio'] + 
             0.3 * indicators['duplicate_ratio'] + 
             0.3 * indicators['outlier_ratio'])
    N = min(1.0, N_raw)
    
    # Granularity: feature diversity and cardinality
    G_raw = (0.5 * indicators['feature_diversity'] + 
             0.5 * min(1.0, indicators['avg_cardinality'] / 50.0))
    G = min(1.0, G_raw)
    
    # Feature-to-instance ratio: smooth scaling using tanh to avoid saturation
    rho_raw = indicators['rho']
    rho = np.tanh(rho_raw * 200)
    
    # Expertise: user provided
    E = min(1.0, max(0.0, manager_expertise))
    
    return {'V': V, 'N': N, 'G': G, 'rho': rho, 'E': E}


# ============================================================
# SECTION 5: REQUIREMENT WEIGHTS FROM COEFFICIENTS (CORN CALIBRATED - ADJUSTED)
# ============================================================
def get_requirement_weights_from_coefficients():
    """
    Calculate requirement weights based on the coefficients from the
    continuous mapping functions (calibrated for Corn dataset).
    
    Rationale for adjustments:
    - Interpretability (0.45): Lowered because black-box models (XGBoost, LightGBM)
      consistently outperform interpretable models (ARIMA) on this dataset.
    - Robustness (0.50): Maintained at moderate level. Tree-based models are
      naturally robust, and data quality is good (low noise).
    - Scalability (0.40): Lowered because dataset is small (697 rows). 
      No need for extreme scalability.
    - Representation Capacity (0.85): Increased because non-linear patterns
      are the key challenge (weak linear correlations found in Phase 1).
    
    The calibrated coefficients for Corn dataset are:
    - Interpretability: 0.45 (a1)
    - Robustness: 0.50 (a2)
    - Scalability: 0.40 (a3)
    - Representation Capacity: 0.85 (a4)
    
    Returns a dictionary with normalized weights that sum to 1.
    """
    alpha = 0.45   # Interpretability coefficient
    gamma = 0.50   # Robustness coefficient
    zeta = 0.40    # Scalability coefficient
    theta = 0.85   # Representation Capacity coefficient
    
    total = alpha + gamma + zeta + theta
    weights = {
        'Interpretability': alpha / total,
        'Robustness': gamma / total,
        'Scalability': zeta / total,
        'Representation Capacity': theta / total
    }
    return weights


# ============================================================
# SECTION 6: CONTINUOUS CONTEXT-TO-REQUIREMENT MAPPING
# ============================================================
def continuous_mapping(V, N, G, rho, E):
    """
    Compute operational requirements using continuous mathematical functions
    as defined in Section 3.3 of the revised paper (calibrated Corn coefficients).
    Returns a numpy array [r_interp, r_robust, r_scal, r_rep]
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    r_interp = 0.45 * (1 - sigmoid(10 * (E - 0.5))) + 0.55 * rho
    r_robust = 0.50 * sigmoid(12 * (N - 0.35)) + 0.50 * np.tanh(2 * rho)
    r_scal = 0.40 * np.tanh(3 * V) + 0.60 * G
    r_rep = 0.85 * G + 0.15 * E
    
    r = np.array([r_interp, r_robust, r_scal, r_rep])
    r = np.clip(r, 0.0, 1.0)
    r = r / (r.sum() + 1e-6)
    return r


def context_to_requirement(context):
    V = context['V']
    N = context['N']
    G = context['G']
    rho = context['rho']
    E = context['E']
    return continuous_mapping(V, N, G, rho, E)


# ============================================================
# SECTION 7: MODEL CAPABILITY PROFILES (CORN-SPECIFIC MODELS)
# ============================================================
def get_capability_matrix():
    capabilities = {
        'ARIMA': np.array([0.80, 0.20, 0.80, 0.20]),
        'Random Forest': np.array([0.50, 0.80, 0.50, 0.80]),
        'XGBoost': np.array([0.20, 0.80, 0.80, 0.90]),
        'LightGBM': np.array([0.20, 0.80, 0.90, 0.90]),
        'WaveletANN': np.array([0.20, 0.80, 0.50, 0.90])
    }
    return capabilities


# ============================================================
# SECTION 8: COMPATIBILITY METHODS
# ============================================================
def manhattan_score(requirement, capability, weights):
    """S_i = 1 - Σ w_j × |r_j - a_ij|"""
    diff = np.abs(requirement - capability)
    weighted_diff = np.sum(weights * diff)
    return float(1.0 - weighted_diff)


def euclidean_score(requirement, capability, weights):
    """S_i = 1 - √(Σ w_j × (r_j - a_ij)²)"""
    diff = requirement - capability
    weighted_sq_diff = np.sum(weights * diff * diff)
    distance = np.sqrt(weighted_sq_diff)
    return float(1.0 - distance)


# ============================================================
# SECTION 9: REPORT GENERATOR (returns string)
# ============================================================
def generate_report(indicators, context, req_weights, requirement,
                    capabilities, scores_manhattan, scores_euclidean,
                    manager_expertise):
    """Generate report as a string."""
    expertise_label = "Expert" if manager_expertise >= 0.5 else "Non-Expert"
    model_names = list(capabilities.keys())
    
    lines = []
    lines.append("=" * 70)
    lines.append("COMPARISON OF COMPATIBILITY METHODS")
    lines.append("=" * 70)
    lines.append(f"Manager Expertise: {expertise_label} ({manager_expertise:.2f})")
    lines.append(f"Dataset: Corn ({indicators['dataset_size']:,} rows, {indicators['n_cols']} columns)")
    lines.append("=" * 70)
    
    lines.append("\nOBSERVABLE DATASET INDICATORS")
    lines.append("-" * 60)
    lines.append(f"  Missing Ratio      : {indicators['missing_ratio']:.2%}")
    lines.append(f"  Duplicate Ratio    : {indicators['duplicate_ratio']:.2%}")
    lines.append(f"  Outlier Ratio      : {indicators['outlier_ratio']:.2%}")
    lines.append(f"  Feature Diversity  : {indicators['feature_diversity']:.3f}")
    lines.append(f"  Avg Cardinality    : {indicators['avg_cardinality']:.1f}")
    lines.append(f"  Avg Correlation    : {indicators['avg_correlation']:.3f}")
    lines.append(f"  Feature-to-Instance (ρ) : {indicators['rho']:.6f}")
    lines.append(f"  Dataset Size       : {indicators['dataset_size']:,}")
    
    lines.append("\nCONTEXT VARIABLES (scaled 0-1)")
    lines.append("-" * 60)
    lines.append(f"  Volume (V)        : {context['V']:.4f}")
    lines.append(f"  Noise (N)         : {context['N']:.4f}")
    lines.append(f"  Granularity (G)   : {context['G']:.4f}")
    lines.append(f"  rho (Feature/Instance) : {context['rho']:.6f}")
    lines.append(f"  Expertise (E)     : {context['E']:.4f}")
    
    lines.append("\nREQUIREMENT WEIGHTS (derived from calibrated coefficients)")
    lines.append("-" * 60)
    req_names = ['Interpretability', 'Robustness', 'Scalability', 'Representation Capacity']
    for name in req_names:
        val = float(req_weights.get(name, 0.0))
        lines.append(f"  {name:20s}: {val:.4f}")
    
    lines.append("\nREQUIREMENT VECTOR (via continuous mapping)")
    lines.append("-" * 60)
    for name, val in zip(req_names, requirement):
        lines.append(f"  {name:20s}: {float(val):.4f}")
    
    lines.append("\nCONTINUOUS MAPPING FUNCTIONS (calibrated Corn coefficients)")
    lines.append("-" * 60)
    lines.append("  r_interp = 0.45*(1 - sigmoid(10*(E-0.5))) + 0.55*rho")
    lines.append("  r_robust = 0.50*sigmoid(12*(N-0.35)) + 0.50*tanh(2*rho)")
    lines.append("  r_scal   = 0.40*tanh(3*V) + 0.60*G")
    lines.append("  r_rep    = 0.85*G + 0.15*E")
    lines.append("  (all values clipped and normalized to sum to 1)")
    
    lines.append("\nCOEFFICIENTS USED FOR WEIGHTS (calibrated Corn V8.0 - Adjusted)")
    lines.append("-" * 60)
    lines.append("  Interpretability    : α = 0.45  (lowered, black-box better)")
    lines.append("  Robustness          : γ = 0.50  (maintained, good data quality)")
    lines.append("  Scalability         : ζ = 0.40  (lowered, small dataset)")
    lines.append("  Representation Cap. : θ = 0.85  (increased, non-linear key)")
    
    lines.append("\nMODEL CAPABILITY PROFILES (Corn-specific models)")
    lines.append("-" * 60)
    lines.append(f"  {'Model':<20s} {'Interp':>8s} {'Robust':>8s} {'Scalab':>8s} {'RepCap':>8s}")
    lines.append("  " + "-" * 55)
    for name in model_names:
        cap = capabilities[name]
        lines.append(f"  {name:<20s} {cap[0]:>8.2f} {cap[1]:>8.2f} {cap[2]:>8.2f} {cap[3]:>8.2f}")
    
    lines.append("\nCOMPATIBILITY METHODS COMPARISON")
    lines.append("-" * 70)
    lines.append(f"  {'Model':<20s} {'Manhattan':>12s} {'Euclidean':>12s}")
    lines.append("  " + "-" * 50)
    for name in model_names:
        m_score = scores_manhattan.get(name, 0.0)
        e_score = scores_euclidean.get(name, 0.0)
        lines.append(f"  {name:<20s} {m_score:>12.4f} {e_score:>12.4f}")
    
    lines.append("\nRANKINGS BY METHOD")
    lines.append("-" * 70)
    
    manhattan_rank = sorted(scores_manhattan.items(), key=lambda x: x[1], reverse=True)
    lines.append(f"\n  MANHATTAN (Primary):")
    for i, (name, score) in enumerate(manhattan_rank, 1):
        lines.append(f"    {i}. {name}: {score:.4f}")
    
    euclidean_rank = sorted(scores_euclidean.items(), key=lambda x: x[1], reverse=True)
    lines.append(f"\n  EUCLIDEAN (Alternative):")
    for i, (name, score) in enumerate(euclidean_rank, 1):
        lines.append(f"    {i}. {name}: {score:.4f}")
    
    lines.append("\nRECOMMENDED MODELS BY METHOD")
    lines.append("-" * 70)
    lines.append(f"  Manhattan (Primary) : {manhattan_rank[0][0]}")
    lines.append(f"  Euclidean           : {euclidean_rank[0][0]}")
    lines.append("=" * 70 + "\n")
    
    return "\n".join(lines)


# ============================================================
# SECTION 10: RUN SCENARIO (single expertise value)
# ============================================================
def run_scenario(manager_expertise, file_path):
    """Run the framework for a given expertise level and return report string."""
    print(f"\n{'#'*70}")
    print(f"# RUNNING SCENARIO: Expertise = {manager_expertise:.2f}")
    print(f"{'#'*70}")
    
    indicators = compute_indicators_streaming(file_path, chunk_size=1000)
    context = compute_context_variables(indicators, manager_expertise)
    
    req_weights = get_requirement_weights_from_coefficients()
    
    requirement = context_to_requirement(context)
    
    weights_array = np.array([
        req_weights['Interpretability'],
        req_weights['Robustness'],
        req_weights['Scalability'],
        req_weights['Representation Capacity']
    ])
    
    capabilities = get_capability_matrix()
    scores_manhattan = {}
    scores_euclidean = {}
    for name, cap in capabilities.items():
        scores_manhattan[name] = manhattan_score(requirement, cap, weights_array)
        scores_euclidean[name] = euclidean_score(requirement, cap, weights_array)
    
    report = generate_report(
        indicators, context, req_weights, requirement,
        capabilities, scores_manhattan, scores_euclidean,
        manager_expertise
    )
    return report


# ============================================================
# MAIN: AUTOMATED EXECUTION FOR THREE EXPERTISE LEVELS
# ============================================================
def main():
    print("=" * 70)
    print("CONTEXT-AWARE MODEL RECOMMENDATION FRAMEWORK")
    print("Version: Calibrated Corn coefficients V8.0 (Adjusted)")
    print("Dataset: integrated_corn_weekly.csv")
    print("=" * 70)
    
    # Load dataset once
    file_path = load_corn_data()
    
    # Define expertise levels
    expertise_levels = [0.01, 0.50, 0.99]
    
    # Store all reports
    all_reports = []
    
    # Run each scenario
    for exp in expertise_levels:
        report = run_scenario(exp, file_path)
        all_reports.append(report)
        print(report)
    
    # Write all reports to a single text file
    output_file = "corn_model_recommendation_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CONTEXT-AWARE MODEL RECOMMENDATION FRAMEWORK\n")
        f.write("RESULTS FOR THREE EXPERTISE SCENARIOS\n")
        f.write("Dataset: integrated_corn_weekly.csv\n")
        f.write("=" * 70 + "\n")
        f.write("Method: Coefficient-based weights (CRITIC removed)\n")
        f.write("Calibrated coefficients (Corn V8.0 - Adjusted):\n")
        f.write("  α=0.45 (Interpretability - lowered, black-box better)\n")
        f.write("  γ=0.50 (Robustness - maintained)\n")
        f.write("  ζ=0.40 (Scalability - lowered, small dataset)\n")
        f.write("  θ=0.85 (Representation Capacity - increased, non-linear key)\n")
        f.write("Normalized weights: Interp=0.205, Robust=0.227, Scal=0.182, RepCap=0.386\n")
        f.write("=" * 70 + "\n\n")
        for i, (exp, report) in enumerate(zip(expertise_levels, all_reports), 1):
            f.write(f"\n{'#'*70}\n")
            f.write(f"# SCENARIO {i}: EXPERTISE = {exp:.2f}\n")
            f.write(f"{'#'*70}\n")
            f.write(report)
            f.write("\n" + "=" * 70 + "\n")
    
    print(f"\nAll results saved to: {output_file}")


if __name__ == "__main__":
    main()

"""
Statistical Analysis Utilities
===============================

Functions:
- prepare_subject_column: Create Subject ID from Repeat + Fold
- perform_repeated_measures_anova: RM-ANOVA test
- tukey_hsd_pairwise: Tukey HSD post-hoc (ORIGINAL VERSION)
- create_pairwise_matrix: Convert Tukey results to symmetric matrix
- compact_letter_display: Generate CLD from p-values
- compare_models_rm_anova: Complete statistical pipeline
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import studentized_range
from statsmodels.stats.anova import AnovaRM

# ============================================================================
# SUBJECT PREPARATION
# ============================================================================

def prepare_subject_column(df):
    """
    Ensure Subject column exists, creating from Repeat+Fold if needed.
    
    Args:
        df: DataFrame with CV results
        
    Returns:
        DataFrame with Subject column
    """
    if "Subject" in df.columns:
        return df

    if {"Repeat", "Fold"}.issubset(df.columns):
        df = df.copy()
        df["Subject"] = df["Repeat"].astype(str) + "_F" + df["Fold"].astype(str)
        return df

    raise ValueError("No 'Subject' column found and cannot build from 'Repeat'+'Fold'.")


# ============================================================================
# REPEATED MEASURES ANOVA
# ============================================================================

def perform_repeated_measures_anova(df, depvar, subject, within):
    """
    Perform repeated measures ANOVA.
    
    Args:
        df: DataFrame with data
        depvar: Dependent variable (metric column)
        subject: Subject identifier column
        within: Within-subjects factor (model column)
        
    Returns:
        AnovaResults object with .anova_table attribute
        
    Example:
        >>> anova_res = perform_repeated_measures_anova(df, 'ROC_AUC', 'Subject', 'Model')
        >>> # Access results via .anova_table attribute:
        >>> p_value = anova_res.anova_table['Pr > F'].iloc[0]
        >>> f_value = anova_res.anova_table['F Value'].iloc[0]
    """

    rm = AnovaRM(
        data=df,
        depvar=depvar,
        subject=subject,
        within=[within]
    )
    return rm.fit()


# ============================================================================
# TUKEY HSD POST-HOC TEST (ORIGINAL IMPLEMENTATION)
# ============================================================================

def tukey_hsd_pairwise(df, metric_col, group_col, subject_col="Subject"):
    """
    Perform Tukey HSD pairwise comparisons for repeated measures.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY CALCULATION LOGIC!
    
    Args:
        df: DataFrame with CV results
        metric_col: Metric to compare (e.g., 'ROC_AUC')
        group_col: Grouping variable (e.g., 'Model')
        subject_col: Subject identifier (e.g., 'Subject')
        
    Returns:
        DataFrame with pairwise comparisons
    """
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    n_subjects = df[subject_col].nunique()

    means = df.groupby(group_col)[metric_col].mean()

    # Calculate within-subject residuals
    residuals = []
    for subj in df[subject_col].unique():
        subj_data = df[df[subject_col] == subj]
        grand_mean = subj_data[metric_col].mean()
        for grp in groups:
            grp_vals = subj_data[subj_data[group_col] == grp][metric_col].values
            if len(grp_vals) > 0:
                residuals.append(grp_vals[0] - grand_mean)

    mse = np.var(residuals, ddof=1) if len(residuals) > 1 else 0
    se = np.sqrt(mse / n_subjects)

    results = []
    for g1, g2 in combinations(groups, 2):
        diff = abs(means[g1] - means[g2])
        q_stat = diff / se if se > 0 else 0

        # Studentized range distribution p-value
        df_error = (n_subjects - 1) * (n_groups - 1)
        # scipy's studentized_range.sf gives upper tail probability (p-value)
        p_value = studentized_range.sf(q_stat, n_groups, df_error)

        results.append({
            "Group1": g1,
            "Group2": g2,
            "MeanDiff": diff,
            "Q-stat": q_stat,
            "p-value": p_value,
            "Significant": p_value < 0.05
        })

    return pd.DataFrame(results)


# ============================================================================
# PAIRWISE MATRIX
# ============================================================================

def create_pairwise_matrix(tukey_results, alpha=0.05):
    """
    Convert Tukey results to symmetric p-value matrix.
    
    Args:
        tukey_results: DataFrame from tukey_hsd_pairwise()
        alpha: Significance level (not used, kept for compatibility)
        
    Returns:
        Symmetric DataFrame of p-values
    """
    groups = sorted(set(tukey_results["Group1"].tolist() + tukey_results["Group2"].tolist()))
    pmat = pd.DataFrame(1.0, index=groups, columns=groups, dtype=float)

    for _, row in tukey_results.iterrows():
        g1, g2, pval = row["Group1"], row["Group2"], row["p-value"]
        # Ensure p-value is numeric
        pval = float(pval)
        pmat.loc[g1, g2] = pval
        pmat.loc[g2, g1] = pval

    for g in groups:
        pmat.loc[g, g] = 1.0

    return pmat


# ============================================================================
# COMPACT LETTER DISPLAY
# ============================================================================

def compact_letter_display(pmat, alpha=0.05):
    """
    Generate Compact Letter Display from p-value matrix.
    
    Models sharing letters are not significantly different.
    
    Args:
        pmat: Symmetric p-value matrix
        alpha: Significance threshold
        
    Returns:
        Series with CLD letters for each model
    """
    models = list(pmat.index)
    groups = []

    for m in models:
        placed = False
        for g in groups:
            # Check if model m is not significantly different from all members of group g
            if all(pmat.loc[m, x] >= alpha for x in g):
                g.append(m)
                placed = True
                break

        if not placed:
            groups.append([m])

    letters = {}
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for i, grp in enumerate(groups):
        letter = alphabet[i % len(alphabet)]
        for model in grp:
            if model in letters:
                letters[model] += letter
            else:
                letters[model] = letter

    return pd.Series(letters, name="CLD")


# ============================================================================
# COMPLETE STATISTICAL PIPELINE
# ============================================================================

def compare_models_rm_anova(df, metric_col, model_col, subject_col="Subject", alpha=0.05):
    """
    Complete statistical comparison pipeline.
    
    Steps:
    1. RM-ANOVA to test for overall differences
    2. Tukey HSD pairwise comparisons
    3. Compact Letter Display generation
    
    Args:
        df: DataFrame with CV results
        metric_col: Metric to compare (e.g., 'ROC_AUC')
        model_col: Column with model names
        subject_col: Column with subject IDs
        alpha: Significance level
        
    Returns:
        dict with:
            - 'anova': ANOVA results
            - 'tukey': Tukey HSD results DataFrame
            - 'pairwise_matrix': Symmetric p-value matrix
            - 'cld': Compact Letter Display Series
    """
    # Ensure subject column exists
    df = prepare_subject_column(df)

    # Perform RM-ANOVA
    anova_result = perform_repeated_measures_anova(
        df, depvar=metric_col, subject=subject_col, within=model_col
    )

    # Tukey HSD pairwise comparisons
    tukey_result = tukey_hsd_pairwise(df, metric_col, model_col, subject_col)

    # Create pairwise matrix
    pairwise_matrix = create_pairwise_matrix(tukey_result, alpha)

    # Generate CLD
    cld = compact_letter_display(pairwise_matrix, alpha)

    return {
        "anova": anova_result,
        "tukey": tukey_result,
        "pairwise_matrix": pairwise_matrix,
        "cld": cld
    }


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

class StatisticalAnalyzer:
    """
    Wrapper class for statistical analysis functions.
    Maintains compatibility with pipeline.py interface.
    """
    
    def __init__(self, alpha=0.05):
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def run_anova(self, df, descriptor, metric):
        """
        Run repeated measures ANOVA for one descriptor.
        
        Args:
            df: DataFrame with all CV results
            descriptor: Descriptor to analyze
            metric: Metric to analyze
            
        Returns:
            ANOVA results object
        """
        # Filter to descriptor
        df_sub = df[df['Descriptor'] == descriptor].copy()
        
        # Ensure Subject column exists
        df_sub = prepare_subject_column(df_sub)
        
        # Run RM-ANOVA
        anova_result = perform_repeated_measures_anova(
            df_sub,
            depvar=metric,
            subject='Subject',
            within='Model'
        )
        
        return anova_result
    
    def tukey_hsd(self, df, descriptor, metric):
        """
        Run Tukey HSD test for one descriptor.
        
        Args:
            df: DataFrame with all CV results
            descriptor: Descriptor to analyze
            metric: Metric to analyze
            
        Returns:
            DataFrame with pairwise comparisons
        """
        # Filter to descriptor
        df_sub = df[df['Descriptor'] == descriptor].copy()
        
        # Ensure Subject column exists
        df_sub = prepare_subject_column(df_sub)
        
        # Run Tukey HSD
        tukey_result = tukey_hsd_pairwise(
            df_sub,
            metric_col=metric,
            group_col='Model',
            subject_col='Subject'
        )
        
        return tukey_result
    
    def compact_letter_display(self, df, descriptor, metric):
        """
        Generate compact letter display for one descriptor.
        
        Args:
            df: DataFrame with all CV results
            descriptor: Descriptor to analyze
            metric: Metric to analyze
            
        Returns:
            Series with CLD letters
        """
        # Get Tukey results
        tukey_result = self.tukey_hsd(df, descriptor, metric)
        
        # Create pairwise matrix
        pairwise_matrix = create_pairwise_matrix(tukey_result, self.alpha)
        
        # Generate CLD
        cld = compact_letter_display(pairwise_matrix, self.alpha)
        
        return cld
    
    def run_analysis(self, df, descriptor, metric, alpha=None):
        """
        Run complete statistical analysis for one descriptor.
        
        Args:
            df: DataFrame with all CV results
            descriptor: Descriptor to analyze
            metric: Metric to analyze
            alpha: Significance level (uses instance alpha if not provided)
            
        Returns:
            dict with statistical results
        """
        if alpha is None:
            alpha = self.alpha
        
        # Filter to descriptor
        df_sub = df[df['Descriptor'] == descriptor].copy()
        
        # Run analysis
        results = compare_models_rm_anova(
            df_sub, 
            metric_col=metric,
            model_col='Model',
            subject_col='Subject',
            alpha=alpha
        )
        
        return results
    
    def save_results(self, results, output_dir, descriptor, metric):
        """
        Save statistical results to files.
        
        Args:
            results: Results dict from compare_models_rm_anova()
            output_dir: Output directory
            descriptor: Descriptor name
            metric: Metric name
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{descriptor}_{metric}"
        
        # Save Tukey results
        tukey_file = output_dir / f"{prefix}_tukey.csv"
        results['tukey'].to_csv(tukey_file, index=False)
        
        # Save pairwise matrix
        pmat_file = output_dir / f"{prefix}_pairwise_matrix.csv"
        results['pairwise_matrix'].to_csv(pmat_file)
        
        # Save CLD
        cld_file = output_dir / f"{prefix}_cld.csv"
        results['cld'].to_csv(cld_file)
        
        # Save ANOVA summary
        anova_file = output_dir / f"{prefix}_anova.txt"
        with open(anova_file, 'w') as f:
            f.write(str(results['anova']))

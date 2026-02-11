"""
Plotting Utilities
==================

Functions:
- set_publication_style: Configure matplotlib for publication figures
- plot_pvalue_heatmap: P-value heatmap with custom binning
- plot_heatmap: General heatmap function
- plot_boxplots: Box plot function
- create_dashboard: Multi-panel dashboard visualization
- grouped_minmax_heatmap: Min-max scaled heatmap grouped by descriptor
- create_combined_boxplot: Combined boxplot across all descriptors
- create_external_bar_plot: Bar plot for external test results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# ============================================================================
# P-VALUE HEATMAP (ORIGINAL)
# ============================================================================

def plot_pvalue_heatmap(pvalue_matrix, title="P-value Heatmap",
                        alpha_levels=[0.001, 0.01, 0.05, 1.0],
                        colors=["#00441b", "#238b45", "#99d8c9", "#fee0d2"],
                        figsize=(10, 8), save_path=None):
    """
    Create p-value heatmap with custom binning.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY!
    
    Args:
        pvalue_matrix: Symmetric matrix of p-values
        title: Plot title
        alpha_levels: Significance level boundaries
        colors: Colors for each bin
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    # Ensure values are numeric
    pvalue_matrix = pvalue_matrix.copy()
    pvalue_matrix = pvalue_matrix.apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values with 1.0 (non-significant)
    pvalue_matrix = pvalue_matrix.fillna(1.0).astype(np.float64)
    
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(alpha_levels, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pvalue_matrix.values, cmap=cmap, norm=norm, aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(pvalue_matrix.columns)))
    ax.set_yticks(range(len(pvalue_matrix.index)))
    ax.set_xticklabels(pvalue_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(pvalue_matrix.index)

    # Annotate cells
    for i in range(len(pvalue_matrix.index)):
        for j in range(len(pvalue_matrix.columns)):
            val = pvalue_matrix.iloc[i, j]
            text_color = 'white' if val < 0.05 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=text_color, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, boundaries=alpha_levels, ticks=alpha_levels)
    cbar.set_label('p-value', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, ax


# ============================================================================
# GENERAL HEATMAP (ORIGINAL)
# ============================================================================

def plot_heatmap(data, title="Heatmap", xlabel=None, ylabel=None,
                 cmap="RdYlGn", annot=True, fmt=".3f", figsize=(10, 8),
                 vmin=None, vmax=None, cbar_label=None, save_path=None):
    """
    Create general heatmap.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY!
    
    Args:
        data: DataFrame or 2D array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap
        annot: Show annotations
        fmt: Annotation format
        figsize: Figure size
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cbar_label: Colorbar label
        save_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cbar_label} if cbar_label else None,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, ax


# ============================================================================
# BOX PLOTS (ORIGINAL)
# ============================================================================

def plot_boxplots(data, x, y, hue=None, title="Box Plot",
                 xlabel=None, ylabel=None, figsize=(12, 6),
                 palette="Set2", order=None, save_path=None):
    """
    Create box plots.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY!
    
    Args:
        data: DataFrame
        x: X-axis column
        y: Y-axis column
        hue: Grouping variable
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        palette: Color palette
        order: Order of categories
        save_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Only pass palette if hue is provided (avoids seaborn warning)
    boxplot_kwargs = {
        'data': data,
        'x': x,
        'y': y,
        'hue': hue,
        'order': order,
        'ax': ax
    }
    
    # Add palette only when hue is provided
    if hue is not None:
        boxplot_kwargs['palette'] = palette
    
    sns.boxplot(**boxplot_kwargs)

    ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    plt.xticks(rotation=45, ha='right')
    if hue:
        plt.legend(title=hue)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, ax


# ============================================================================
# DASHBOARD (ORIGINAL)
# ============================================================================

def create_dashboard(master_df, save_path=None):
    """
    Create multi-panel dashboard visualization.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY!
    
    Args:
        master_df: Master table with mean metrics
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_publication_style()

    fig = plt.figure(figsize=(18, 12))

    # ROC-AUC heatmap
    roc_pivot = master_df.pivot(index='Model', columns='Descriptor', values='ROC_AUC_mean')
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(roc_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, ax=ax1, cbar_kws={'label': 'ROC-AUC'})
    ax1.set_title("ROC-AUC", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Descriptor")
    ax1.set_ylabel("Model")

    # MCC heatmap
    if 'MCC_mean' in master_df.columns:
        mcc_pivot = master_df.pivot(index='Model', columns='Descriptor', values='MCC_mean')
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(mcc_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, ax=ax2, cbar_kws={'label': 'MCC'})
        ax2.set_title("MCC", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Descriptor")
        ax2.set_ylabel("Model")

    # GMean heatmap
    if 'GMean_mean' in master_df.columns:
        gmean_pivot = master_df.pivot(index='Model', columns='Descriptor', values='GMean_mean')
        ax3 = plt.subplot(2, 3, 3)
        sns.heatmap(gmean_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0.5, vmax=1.0, ax=ax3, cbar_kws={'label': 'G-Mean'})
        ax3.set_title("G-Mean", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Descriptor")
        ax3.set_ylabel("Model")

    # Top 10 by ROC-AUC
    ax4 = plt.subplot(2, 3, 4)
    top_models = master_df.nlargest(10, 'ROC_AUC_mean')[['Model', 'Descriptor', 'ROC_AUC_mean']]
    top_models['Label'] = top_models['Model'] + '\n(' + top_models['Descriptor'] + ')'
    ax4.barh(range(len(top_models)), top_models['ROC_AUC_mean'], color='steelblue')
    ax4.set_yticks(range(len(top_models)))
    ax4.set_yticklabels(top_models['Label'], fontsize=9)
    ax4.set_xlabel('ROC-AUC', fontsize=11)
    ax4.set_title('Top 10 Models by ROC-AUC', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()

    # Top 10 by MCC
    if 'MCC_mean' in master_df.columns:
        ax5 = plt.subplot(2, 3, 5)
        top_mcc = master_df.nlargest(10, 'MCC_mean')[['Model', 'Descriptor', 'MCC_mean']]
        top_mcc['Label'] = top_mcc['Model'] + '\n(' + top_mcc['Descriptor'] + ')'
        ax5.barh(range(len(top_mcc)), top_mcc['MCC_mean'], color='coral')
        ax5.set_yticks(range(len(top_mcc)))
        ax5.set_yticklabels(top_mcc['Label'], fontsize=9)
        ax5.set_xlabel('MCC', fontsize=11)
        ax5.set_title('Top 10 Models by MCC', fontsize=14, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        ax5.invert_yaxis()

    # Overall metric averages
    ax6 = plt.subplot(2, 3, 6)
    metric_means = {
        'ROC-AUC': master_df['ROC_AUC_mean'].mean(),
    }
    if 'MCC_mean' in master_df.columns:
        metric_means['MCC'] = master_df['MCC_mean'].mean()
    if 'GMean_mean' in master_df.columns:
        metric_means['G-Mean'] = master_df['GMean_mean'].mean()
    if 'F1_mean' in master_df.columns:
        metric_means['F1'] = master_df['F1_mean'].mean()

    ax6.bar(metric_means.keys(), metric_means.values(), 
            color=['steelblue', 'coral', 'seagreen', 'mediumpurple'])
    ax6.set_ylabel('Mean Value', fontsize=11)
    ax6.set_title('Overall Metric Averages', fontsize=14, fontweight='bold')
    ax6.set_ylim(0, 1.0)
    ax6.grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(metric_means.items()):
        ax6.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('ML Benchmark Dashboard',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig


# ============================================================================
# CONVENIENCE WRAPPER CLASS
# ============================================================================

class Visualizer:
    """
    Wrapper class for plotting functions.
    Maintains compatibility with pipeline.py interface.
    """
    
    def __init__(self, output_dir, dpi=300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set publication style on initialization
        set_publication_style()
        if self.dpi != 300:
            plt.rcParams['savefig.dpi'] = self.dpi
    
    def plot_tukey_heatmap(self, pvalue_matrix, metric, descriptor):
        """
        Create and save Tukey p-value heatmap.
        
        Args:
            pvalue_matrix: Symmetric p-value matrix
            metric: Metric name
            descriptor: Descriptor name
            
        Returns:
            Path to saved figure
        """
        save_path = self.output_dir / f"{descriptor}_{metric}_tukey_pvalue_heatmap.png"
        
        plot_pvalue_heatmap(
            pvalue_matrix,
            title=f"Tukey HSD P-values: {descriptor} - {metric}",
            save_path=save_path
        )
        
        return save_path
    
    def plot_heatmap(self, df, metric, descriptor_order=None, model_order=None):
        """
        Create heatmap of metric across descriptors and models.
        
        Args:
            df: Master table DataFrame
            metric: Metric name (e.g., 'ROC_AUC')
            descriptor_order: Order of descriptors
            model_order: Order of models
            
        Returns:
            Path to saved figure or None if metric not found
        """
        # Try with _mean suffix first, then without
        metric_col = None
        if f"{metric}_mean" in df.columns:
            metric_col = f"{metric}_mean"
        elif metric in df.columns:
            metric_col = metric
        else:
            print(f"Warning: Neither '{metric}_mean' nor '{metric}' found in DataFrame")
            print(f"Available columns: {df.columns.tolist()}")
            # Return a dummy path to avoid NoneType error
            save_path = self.output_dir / f"{metric}_heatmap_NOT_CREATED.png"
            return save_path
        
        pivot_data = df.pivot(
            index='Model',
            columns='Descriptor',
            values=metric_col
        )
        
        # Reorder if specified
        if descriptor_order:
            pivot_data = pivot_data.reindex(columns=descriptor_order)
        if model_order:
            pivot_data = pivot_data.reindex(index=model_order)
        
        save_path = self.output_dir / f"{metric}_heatmap.png"
        
        plot_heatmap(
            pivot_data,
            title=f"{metric} Comparison",
            xlabel="Descriptor",
            ylabel="Model",
            cmap="RdYlGn",
            annot=True,
            fmt=".3f",
            figsize=(10, 8),
            vmin=pivot_data.min().min(),
            vmax=pivot_data.max().max(),
            cbar_label=metric,
            save_path=save_path
        )
        
        return save_path
        return save_path
    
    def plot_boxplot(self, df, descriptor, metric, model_order=None):
        """
        Create and save box plot for descriptor-metric combination.
        
        Args:
            df: DataFrame with per-fold results
            descriptor: Descriptor name
            metric: Metric name
            model_order: Order of models on x-axis
            
        Returns:
            Path to saved figure
        """
        # Filter to descriptor
        df_sub = df[df['Descriptor'] == descriptor].copy()
        
        if metric not in df_sub.columns:
            print(f"Warning: {metric} not in DataFrame")
            return None
        
        save_path = self.output_dir / f"{descriptor}_{metric}_boxplot.png"
        
        plot_boxplots(
            data=df_sub,
            x='Model',
            y=metric,
            title=f"{descriptor} - {metric} Distribution",
            xlabel="Model",
            ylabel=metric,
            figsize=(14, 6),
            palette="Set2",
            order=model_order,
            save_path=save_path
        )
        
        return save_path
    
    def plot_comparison(self, df, metrics=None, descriptor_order=None, model_order=None):
        """
        Create comparison plot (grouped minmax heatmap).
        
        Args:
            df: Master table DataFrame
            metrics: List of metrics to include
            descriptor_order: Order of descriptors
            model_order: Order of models
            
        Returns:
            Path to saved figure
        """
        if metrics is None:
            metrics = ['ROC_AUC', 'MCC', 'GMean']
        
        save_path = self.output_dir / "comparison_minmax_heatmap.png"
        
        grouped_minmax_heatmap(
            df,
            metrics=metrics,
            descriptor_order=descriptor_order if descriptor_order else df['Descriptor'].unique().tolist(),
            model_order=model_order,
            save_path=save_path
        )
        
        return save_path
    
    def create_combined_boxplot(self, df, metric, descriptor_order=None, model_order=None):
        """
        Create combined boxplot showing all descriptors together.
        
        Args:
            df: DataFrame with per-fold results
            metric: Metric to plot
            descriptor_order: Order of descriptors
            model_order: Order of models
            
        Returns:
            Path to saved figure
        """
        save_path = self.output_dir / f"all_descriptors_{metric.lower()}_boxplot.png"
        
        return create_combined_boxplot(
            df,
            metric,
            descriptor_order=descriptor_order,
            model_order=model_order,
            save_path=save_path
        )
    
    def create_dashboard(self, master_df):
        """
        Create and save dashboard.
        
        Args:
            master_df: Master table DataFrame
            
        Returns:
            Path to saved figure
        """
        save_path = self.output_dir / "dashboard.png"
        
        create_dashboard(master_df, save_path=save_path)
        
        return save_path


# ============================================================================
# GROUPED MIN-MAX HEATMAP (ORIGINAL)
# ============================================================================

def grouped_minmax_heatmap(
    df,
    metrics,
    descriptor_order=None,
    model_order=None,
    add_separators=True,
    cmap_name="viridis",
    save_path=None,
):
    """
    Create grouped min-max scaled heatmap.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY!
    
    Args:
        df: DataFrame with columns ['Descriptor', 'Model'] + metrics
        metrics: list of metric column names to include as columns
        descriptor_order: list of descriptor names in desired block order
        model_order: list of model names in desired row order (same for each block)
        add_separators: if True, insert blank rows between descriptor blocks
        cmap_name: Colormap name
        save_path: Path to save figure
        
    Returns:
        None (displays and optionally saves figure)
    """
    # Remove '_mean' suffix from column names if present
    df = df.copy()
    df.columns = df.columns.str.removesuffix('_mean')
    
    # Subset to what we need
    subset = df[["Descriptor", "Model"] + metrics].copy()

    all_rows = []
    row_labels = []
    block_bounds = []  # (desc, start_idx, end_idx) *before* separator row

    for i, desc in enumerate(descriptor_order):
        block = subset[subset["Descriptor"] == desc].copy()
        if block.empty:
            continue

        if model_order is not None:
            block["Model"] = pd.Categorical(block["Model"], categories=model_order, ordered=True)
            block = block.sort_values("Model")

        block = block.dropna(subset=["Model"])

        start_idx = len(all_rows)
        for _, row in block.iterrows():
            all_rows.append(row[metrics].values)
            row_labels.append(row["Model"])
        end_idx = len(all_rows)
        block_bounds.append((desc, start_idx, end_idx))

        if add_separators and i < len(descriptor_order) - 1:
            all_rows.append([np.nan] * len(metrics))
            row_labels.append("")  # blank label for separator

    mm = pd.DataFrame(all_rows, columns=metrics)

    mins = mm.min(skipna=True)
    ranges = mm.max(skipna=True) - mins
    ranges[ranges == 0] = 1.0
    mm_scaled = (mm - mins) / ranges

    data = np.ma.masked_invalid(mm_scaled.values)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.grid(False)          # turn off grid if it was enabled
    
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")

    for desc, start, end in block_bounds:
        mid = (start + end - 1) / 2.0
        ax.text(
            -2.6,
            mid,
            desc,
            va="center",
            ha="right",
            fontsize=12,
            rotation=90,
        )
    fig.subplots_adjust(left=0.22)   # tweak 0.2–0.28 as needed
    # Optional: thicker horizontal lines at block boundaries (just above each block)
    # (Use start index; separator row gives additional white space)
    for desc, start, end in block_bounds[1:]:
        ax.axhline(start - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Min-max scaled metric")
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    plt.tight_layout()
    plt.show()


# ============================================================================
# COMBINED BOXPLOT (ORIGINAL)
# ============================================================================

def create_combined_boxplot(df, metric, descriptor_order=None, model_order=None, save_path=None):
    """
    Create combined boxplot showing all descriptors together.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY!
    
    Args:
        df: DataFrame with per-fold results (must have 'Descriptor' column)
        metric: Metric to plot
        descriptor_order: Order of descriptors (for legend)
        model_order: Order of models on x-axis
        save_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    set_publication_style()
    
    if save_path is None:
        save_path = Path(f"all_descriptors_{metric.lower()}_boxplot.png")
    
    plot_boxplots(
        data=df,
        x='Model',
        y=metric,
        hue='Descriptor',
        title=f"{metric} Across All Descriptors",
        xlabel="Model",
        ylabel=metric,
        figsize=(16, 7),
        palette="Set2",
        order=model_order,
        save_path=save_path
    )
    
    print(f"Saved combined boxplot to: {save_path}")
    return save_path


# ============================================================================
# EXTERNAL TEST BAR PLOT (ORIGINAL)
# ============================================================================

def create_external_bar_plot(model_name="Morgan_LogisticRegression",
                             metrics=["MCC", "GMean", "ROC_AUC", "PR_AUC"],
                             metric_labels=["MCC", "G-mean", "ROC-AUC", "PR-AUC"],
                             external_file=None):
    """
    Create bar plot for external test results.
    
    ORIGINAL IMPLEMENTATION - DO NOT MODIFY!
    
    Args:
        model_name: Model identifier (e.g., 'Morgan_LogisticRegression')
        metrics: List of metric column names
        metric_labels: Display labels for metrics
        external_file: Path to external test metrics CSV
        
    Returns:
        List of metric values
    """
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # Find external test file
    if external_file is None:
        possible_files = [
            Path("external_test_metrics.csv"),
            Path("champions_external_metrics.csv"),
            Path("TRPV1_external_test_results.csv")
        ]
        
        stats_csv = None
        for f in possible_files:
            if f.exists():
                stats_csv = f
                break
        
        if stats_csv is None:
            print("ERROR: No external test metrics file found")
            print("Please run external test evaluation first")
            return None
    else:
        stats_csv = Path(external_file)
        if not stats_csv.exists():
            print(f"ERROR: File not found: {stats_csv}")
            return None

    print(f"Loading results from: {stats_csv}")
    df = pd.read_csv(stats_csv)

    # Find the row for this model
    if 'Method' in df.columns:
        row = df[df["Method"] == model_name]
        if row.empty:
            print(f"Warning: Model {model_name} not found, using first row")
            row = df.iloc[0:1]
    elif 'Model' in df.columns and 'Descriptor' in df.columns:
        fp_type, model_type = model_name.split("_", 1)
        row = df[(df["Descriptor"] == fp_type) & (df["Model"] == model_type)]
        if row.empty:
            print(f"Warning: Model {model_name} not found, using first row")
            row = df.iloc[0:1]
    else:
        row = df.iloc[0:1]

    row = row.iloc[0]

    # Extract metric values
    values = []
    for m in metrics:
        if m in row:
            values.append(float(row[m]))
        else:
            values.append(0.0)
            print(f"Warning: Metric {m} not found in data")

    # Create plot
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    x = range(len(metrics))
    bars = ax.bar(
        x,
        values,
        width=0.6,
        color="#4C72B0",
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"External Test - {model_name.replace('_', ' ')}",
                 fontsize=13, fontweight='bold')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()

    # Save
    png_path = Path(f"external_{model_name}_bars.png")
    pdf_path = Path(f"external_{model_name}_bars.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved bar plots to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")

    return values

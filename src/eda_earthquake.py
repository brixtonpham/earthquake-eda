"""
EDA Earthquake Analysis
=======================
Comprehensive Exploratory Data Analysis for Earthquake Dataset (2002-2025)

Author: AI Agent
Date: December 2024
"""

# ============================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set global plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# File paths
DATA_PATH = './data/processed/final_earthquake_data_2002_2025.csv'
OUTPUT_DIR = './outputs/figures/'

# Figure settings
FIGSIZE_SMALL = (8, 6)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_LARGE = (16, 10)
FIGSIZE_XLARGE = (20, 12)

# Column groups for analysis
NUMERICAL_COLS = ['Latitude', 'Longitude', 'Depth', 'Magnitude',
                  'nst', 'gap', 'rms', 'horizontalError', 'depthError',
                  'year', 'month', 'day', 'hour', 'energy']
CATEGORICAL_COLS = ['magnitude_category', 'depth_category', 'magType',
                    'type', 'status', 'net']

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SECTION 2: DATA LOADING FUNCTIONS
# ============================================================

def load_data(filepath):
    """
    Load CSV file with proper datetime parsing.

    Parameters:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    print(f"Loading data from: {filepath}")

    df = pd.read_csv(filepath, parse_dates=['Timestamp', 'updated'])

    print(f"✓ Data loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def get_data_overview(df):
    """
    Generate comprehensive data overview.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Dictionary with overview statistics
    """
    overview = {
        'shape': df.shape,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'numerical_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_cols': df.select_dtypes(include=['datetime64']).columns.tolist()
    }

    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print(f"Total Records: {overview['rows']:,}")
    print(f"Total Columns: {overview['columns']}")
    print(f"Memory Usage: {overview['memory_usage_mb']:.2f} MB")
    print(f"\nColumn Types:")
    for dtype, count in overview['dtypes'].items():
        print(f"  - {dtype}: {count}")
    print(f"\nNumerical Columns ({len(overview['numerical_cols'])}):")
    print(f"  {overview['numerical_cols']}")
    print(f"\nCategorical Columns ({len(overview['categorical_cols'])}):")
    print(f"  {overview['categorical_cols']}")

    return overview


def get_descriptive_stats(df):
    """
    Generate descriptive statistics.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        tuple: (numerical_stats_df, categorical_stats_df)
    """
    numerical_stats = df.describe()
    categorical_stats = df.describe(include=['object', 'category'])

    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS - NUMERICAL")
    print("="*60)
    print(numerical_stats.round(2).to_string())

    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS - CATEGORICAL")
    print("="*60)
    print(categorical_stats.to_string())

    return numerical_stats, categorical_stats


# ============================================================
# SECTION 3: DATA QUALITY FUNCTIONS
# ============================================================

def analyze_missing_values(df):
    """
    Analyze and visualize missing values.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with missing value statistics
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing Percentage': missing_pct.values
    }).sort_values('Missing Percentage', ascending=False)

    # Filter to show only columns with missing values
    missing_df_filtered = missing_df[missing_df['Missing Count'] > 0].reset_index(drop=True)

    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    if len(missing_df_filtered) > 0:
        print(missing_df_filtered.to_string(index=False))
    else:
        print("No missing values found in the dataset!")

    # Create visualization
    if len(missing_df_filtered) > 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
        bars = ax.barh(missing_df_filtered['Column'],
                       missing_df_filtered['Missing Percentage'],
                       color='coral')
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_ylabel('Column')
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')

        # Add percentage labels
        for bar, pct in zip(bars, missing_df_filtered['Missing Percentage']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{pct:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}missing_values.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {OUTPUT_DIR}missing_values.png")

    return missing_df


def check_duplicates(df):
    """
    Check for duplicate records.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Dictionary with duplicate counts
    """
    exact_duplicates = df.duplicated().sum()
    key_duplicates = df.duplicated(subset=['Timestamp', 'Latitude', 'Longitude', 'Magnitude']).sum()

    duplicates_info = {
        'exact_duplicates': exact_duplicates,
        'key_field_duplicates': key_duplicates
    }

    print("\n" + "="*60)
    print("DUPLICATE RECORDS CHECK")
    print("="*60)
    print(f"Exact Duplicates: {exact_duplicates:,}")
    print(f"Duplicates by Key Fields (Timestamp, Lat, Lon, Mag): {key_duplicates:,}")

    return duplicates_info


def validate_data_consistency(df):
    """
    Validate data ranges and consistency.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Dictionary with validation results
    """
    validation = {}

    # Latitude check (-90 to 90)
    lat_valid = df['Latitude'].between(-90, 90).all()
    lat_outliers = df[~df['Latitude'].between(-90, 90)].shape[0]
    validation['latitude_valid'] = lat_valid
    validation['latitude_outliers'] = lat_outliers

    # Longitude check (-180 to 180)
    lon_valid = df['Longitude'].between(-180, 180).all()
    lon_outliers = df[~df['Longitude'].between(-180, 180)].shape[0]
    validation['longitude_valid'] = lon_valid
    validation['longitude_outliers'] = lon_outliers

    # Depth check (>= 0)
    depth_valid = (df['Depth'] >= 0).all()
    depth_negative = (df['Depth'] < 0).sum()
    validation['depth_valid'] = depth_valid
    validation['depth_negative'] = depth_negative

    # Magnitude check (-2 to 10)
    mag_valid = df['Magnitude'].between(-2, 10).all()
    mag_outliers = df[~df['Magnitude'].between(-2, 10)].shape[0]
    validation['magnitude_valid'] = mag_valid
    validation['magnitude_outliers'] = mag_outliers

    # Year check (2002 to 2025)
    year_valid = df['year'].between(2002, 2025).all()
    year_outliers = df[~df['year'].between(2002, 2025)].shape[0]
    validation['year_valid'] = year_valid
    validation['year_outliers'] = year_outliers

    print("\n" + "="*60)
    print("DATA CONSISTENCY VALIDATION")
    print("="*60)
    print(f"Latitude Range [-90, 90]: {'✓ Valid' if lat_valid else f'✗ {lat_outliers} outliers'}")
    print(f"Longitude Range [-180, 180]: {'✓ Valid' if lon_valid else f'✗ {lon_outliers} outliers'}")
    print(f"Depth >= 0: {'✓ Valid' if depth_valid else f'✗ {depth_negative} negative values'}")
    print(f"Magnitude Range [-2, 10]: {'✓ Valid' if mag_valid else f'✗ {mag_outliers} outliers'}")
    print(f"Year Range [2002, 2025]: {'✓ Valid' if year_valid else f'✗ {year_outliers} outliers'}")

    return validation


def detect_outliers(df, columns, method='iqr'):
    """
    Detect outliers using IQR or Z-score method.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check
        method (str): 'iqr' or 'zscore'

    Returns:
        dict: Dictionary with outlier information per column
    """
    outlier_info = {}

    for col in columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        else:  # zscore
            z_scores = np.abs(stats.zscore(data))
            outliers = (z_scores > 3).sum()

        outlier_info[col] = {
            'outlier_count': outliers,
            'outlier_percentage': (outliers / len(data)) * 100
        }

    print("\n" + "="*60)
    print(f"OUTLIER DETECTION ({method.upper()} Method)")
    print("="*60)
    for col, info in outlier_info.items():
        print(f"{col}: {info['outlier_count']:,} outliers ({info['outlier_percentage']:.2f}%)")

    # Create boxplot visualization
    valid_cols = [c for c in columns if c in df.columns]
    if valid_cols:
        n_cols = len(valid_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 6))
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, valid_cols):
            df.boxplot(column=col, ax=ax)
            ax.set_title(f'{col}', fontsize=12)
            ax.set_ylabel('Value')

        plt.suptitle('Outlier Detection - Box Plots', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}outliers_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {OUTPUT_DIR}outliers_boxplot.png")

    return outlier_info


# ============================================================
# SECTION 4: UNIVARIATE ANALYSIS FUNCTIONS
# ============================================================

def plot_magnitude_distribution(df):
    """
    ★ KEY DELIVERABLE: Create comprehensive magnitude distribution visualization.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Statistics dictionary
    """
    print("\n★ Creating Magnitude Distribution (Key Deliverable)...")

    mag_data = df['Magnitude'].dropna()

    # Calculate statistics
    stats_dict = {
        'mean': mag_data.mean(),
        'median': mag_data.median(),
        'std': mag_data.std(),
        'min': mag_data.min(),
        'max': mag_data.max(),
        'skewness': stats.skew(mag_data),
        'kurtosis': stats.kurtosis(mag_data)
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_LARGE)

    # Subplot 1: Histogram with KDE
    ax1 = axes[0]
    ax1.hist(mag_data, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    mag_data.plot(kind='kde', ax=ax1, color='red', linewidth=2, label='KDE')
    ax1.axvline(stats_dict['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean: {stats_dict['mean']:.2f}")
    ax1.axvline(stats_dict['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats_dict['median']:.2f}")
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Density')
    ax1.set_title('Magnitude Distribution (Histogram + KDE)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

    # Subplot 2: Boxplot
    ax2 = axes[1]
    bp = ax2.boxplot(mag_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Magnitude Boxplot', fontsize=12, fontweight='bold')

    # Add quartile annotations
    quartiles = mag_data.quantile([0.25, 0.5, 0.75])
    ax2.annotate(f"Q1: {quartiles[0.25]:.2f}", xy=(1.1, quartiles[0.25]), fontsize=9)
    ax2.annotate(f"Q2: {quartiles[0.5]:.2f}", xy=(1.1, quartiles[0.5]), fontsize=9)
    ax2.annotate(f"Q3: {quartiles[0.75]:.2f}", xy=(1.1, quartiles[0.75]), fontsize=9)

    # Subplot 3: Category bar chart
    ax3 = axes[2]
    if 'magnitude_category' in df.columns:
        cat_order = ['Micro', 'Minor', 'Small', 'Light', 'Moderate', 'Strong', 'Major', 'Great']
        cat_counts = df['magnitude_category'].value_counts()
        # Reorder based on severity
        cat_counts = cat_counts.reindex([c for c in cat_order if c in cat_counts.index])
        bars = ax3.bar(cat_counts.index, cat_counts.values, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cat_counts))))
        ax3.set_xlabel('Magnitude Category')
        ax3.set_ylabel('Count')
        ax3.set_title('Earthquake Counts by Category', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=8)

    # Add overall statistics text box
    stats_text = (f"Statistics:\n"
                  f"Mean: {stats_dict['mean']:.3f}\n"
                  f"Median: {stats_dict['median']:.3f}\n"
                  f"Std Dev: {stats_dict['std']:.3f}\n"
                  f"Skewness: {stats_dict['skewness']:.3f}\n"
                  f"Kurtosis: {stats_dict['kurtosis']:.3f}")

    plt.suptitle('★ Magnitude Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}magnitude_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}magnitude_distribution.png")
    print(f"  Mean: {stats_dict['mean']:.3f}, Median: {stats_dict['median']:.3f}, Skewness: {stats_dict['skewness']:.3f}")

    return stats_dict


def plot_depth_distribution(df):
    """
    Create comprehensive depth distribution visualization.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Statistics dictionary
    """
    print("\nCreating Depth Distribution...")

    depth_data = df['Depth'].dropna()

    # Calculate statistics
    stats_dict = {
        'mean': depth_data.mean(),
        'median': depth_data.median(),
        'std': depth_data.std(),
        'min': depth_data.min(),
        'max': depth_data.max(),
        'skewness': stats.skew(depth_data),
        'kurtosis': stats.kurtosis(depth_data)
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_LARGE)

    # Subplot 1: Histogram with KDE
    ax1 = axes[0]
    ax1.hist(depth_data, bins=50, density=True, alpha=0.7, color='teal', edgecolor='black')
    depth_data.plot(kind='kde', ax=ax1, color='red', linewidth=2, label='KDE')
    ax1.axvline(stats_dict['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean: {stats_dict['mean']:.1f}")
    ax1.set_xlabel('Depth (km)')
    ax1.set_ylabel('Density')
    ax1.set_title('Depth Distribution (Histogram + KDE)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

    # Subplot 2: Boxplot
    ax2 = axes[1]
    bp = ax2.boxplot(depth_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    ax2.set_ylabel('Depth (km)')
    ax2.set_title('Depth Boxplot', fontsize=12, fontweight='bold')

    # Subplot 3: Category bar chart
    ax3 = axes[2]
    if 'depth_category' in df.columns:
        cat_counts = df['depth_category'].value_counts()
        cat_order = ['Shallow', 'Intermediate', 'Deep', 'Very Deep']
        cat_counts = cat_counts.reindex([c for c in cat_order if c in cat_counts.index])
        colors = ['#90EE90', '#FFD700', '#FFA500', '#FF4500'][:len(cat_counts)]
        bars = ax3.bar(cat_counts.index, cat_counts.values, color=colors)
        ax3.set_xlabel('Depth Category')
        ax3.set_ylabel('Count')
        ax3.set_title('Earthquake Counts by Depth Category', fontsize=12, fontweight='bold')

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle('Depth Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}depth_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}depth_distribution.png")

    return stats_dict


def plot_energy_distribution(df):
    """
    Analyze energy distribution (log-scale).

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nCreating Energy Distribution...")

    if 'energy' not in df.columns:
        print("  Warning: 'energy' column not found")
        return None

    energy_data = df['energy'].dropna()
    log_energy = np.log10(energy_data + 1)  # Add 1 to handle zeros

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Subplot 1: Log-transformed histogram
    ax1 = axes[0]
    ax1.hist(log_energy, bins=40, alpha=0.7, color='purple', edgecolor='black')
    ax1.set_xlabel('Log10(Energy + 1)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Energy Distribution (Log Scale)', fontsize=12, fontweight='bold')

    # Subplot 2: Energy by magnitude category
    ax2 = axes[1]
    if 'magnitude_category' in df.columns:
        cat_order = ['Micro', 'Minor', 'Small', 'Light', 'Moderate', 'Strong', 'Major', 'Great']
        df_plot = df[df['magnitude_category'].isin(cat_order)].copy()
        df_plot['magnitude_category'] = pd.Categorical(df_plot['magnitude_category'], categories=cat_order, ordered=True)
        df_plot.boxplot(column='energy', by='magnitude_category', ax=ax2, showfliers=False)
        ax2.set_yscale('log')
        ax2.set_xlabel('Magnitude Category')
        ax2.set_ylabel('Energy (log scale)')
        ax2.set_title('Energy by Magnitude Category', fontsize=12, fontweight='bold')
        plt.suptitle('')  # Remove automatic title
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}energy_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}energy_distribution.png")

    return fig


def plot_categorical_distributions(df):
    """
    Create bar charts for all categorical variables.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nCreating Categorical Distributions...")

    cat_cols = ['magnitude_category', 'depth_category', 'magType', 'type', 'status', 'net']
    available_cols = [col for col in cat_cols if col in df.columns]

    n_cols = len(available_cols)
    if n_cols == 0:
        print("  No categorical columns found")
        return None

    n_rows = (n_cols + 2) // 3  # 3 columns per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for idx, col in enumerate(available_cols):
        ax = axes[idx]
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        bars = ax.barh(value_counts.index, value_counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))
        ax.set_xlabel('Count')
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')

        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width):,}',
                   ha='left', va='center', fontsize=8)

    # Hide unused axes
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Categorical Variable Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}categorical_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}categorical_distributions.png")

    return fig


def analyze_quality_metrics(df):
    """
    Analyze distribution of quality metrics.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nAnalyzing Quality Metrics...")

    quality_cols = ['nst', 'gap', 'rms', 'horizontalError', 'depthError']
    available_cols = [col for col in quality_cols if col in df.columns]

    if len(available_cols) == 0:
        print("  No quality metric columns found")
        return None

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

    for idx, col in enumerate(available_cols):
        ax = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=40, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Quality Metrics Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}quality_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}quality_metrics.png")

    return fig


# ============================================================
# SECTION 5: BIVARIATE ANALYSIS FUNCTIONS
# ============================================================

def plot_depth_vs_magnitude(df):
    """
    ★ KEY DELIVERABLE: Create scatter plot of Depth vs Magnitude.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Correlation coefficients
    """
    print("\n★ Creating Depth vs Magnitude Scatter Plot (Key Deliverable)...")

    # Remove missing values
    df_clean = df[['Depth', 'Magnitude', 'magnitude_category']].dropna()

    # Calculate correlations
    pearson_corr, pearson_p = stats.pearsonr(df_clean['Depth'], df_clean['Magnitude'])
    spearman_corr, spearman_p = stats.spearmanr(df_clean['Depth'], df_clean['Magnitude'])

    correlations = {
        'pearson': pearson_corr,
        'pearson_pvalue': pearson_p,
        'spearman': spearman_corr,
        'spearman_pvalue': spearman_p
    }

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # Subplot 1: Scatter plot colored by magnitude category
    ax1 = axes[0]

    # Sample data if too large (for better visualization)
    if len(df_clean) > 50000:
        df_sample = df_clean.sample(n=50000, random_state=42)
    else:
        df_sample = df_clean

    # Create color mapping for categories
    cat_order = ['Micro', 'Minor', 'Small', 'Light', 'Moderate', 'Strong', 'Major', 'Great']
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(cat_order)))
    color_map = {cat: colors[i] for i, cat in enumerate(cat_order)}

    for cat in cat_order:
        if cat in df_sample['magnitude_category'].unique():
            subset = df_sample[df_sample['magnitude_category'] == cat]
            ax1.scatter(subset['Depth'], subset['Magnitude'],
                       alpha=0.5, s=10, label=cat, color=color_map.get(cat, 'gray'))

    ax1.set_xlabel('Depth (km)', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.set_title('Depth vs Magnitude (by Category)', fontsize=12, fontweight='bold')
    ax1.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Add correlation text
    corr_text = f"Pearson r = {pearson_corr:.4f}\nSpearman ρ = {spearman_corr:.4f}"
    ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 2: Hexbin for density visualization
    ax2 = axes[1]
    hb = ax2.hexbin(df_clean['Depth'], df_clean['Magnitude'],
                    gridsize=50, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('Depth (km)', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Depth vs Magnitude (Density)', fontsize=12, fontweight='bold')
    cb = plt.colorbar(hb, ax=ax2)
    cb.set_label('Count')

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['Depth'], df_clean['Magnitude'])
    x_line = np.array([df_clean['Depth'].min(), df_clean['Depth'].max()])
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'b--', linewidth=2, label=f'Regression: y={slope:.4f}x+{intercept:.2f}')
    ax2.legend(fontsize=9)

    plt.suptitle('★ Depth vs Magnitude Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}depth_vs_magnitude_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}depth_vs_magnitude_scatter.png")
    print(f"  Pearson r: {pearson_corr:.4f}, Spearman ρ: {spearman_corr:.4f}")

    return correlations


def analyze_temporal_patterns(df):
    """
    Analyze earthquake patterns over time.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nAnalyzing Temporal Patterns...")

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_XLARGE)

    # Subplot 1: Yearly earthquake counts
    ax1 = axes[0, 0]
    yearly_counts = df.groupby('year').size()
    ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=4)
    ax1.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Earthquake Count')
    ax1.set_title('Earthquakes per Year', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Subplot 2: Monthly pattern (aggregated)
    ax2 = axes[0, 1]
    monthly_counts = df.groupby('month').size()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2.bar(monthly_counts.index, monthly_counts.values, color='steelblue')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Earthquake Count')
    ax2.set_title('Earthquakes by Month (All Years)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names, rotation=45)

    # Subplot 3: Day of week distribution
    ax3 = axes[0, 2]
    if 'dayofweek' in df.columns:
        dow_counts = df.groupby('dayofweek').size()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax3.bar(dow_counts.index, dow_counts.values, color='teal')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Earthquake Count')
        ax3.set_title('Earthquakes by Day of Week', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(day_names)

    # Subplot 4: Hourly distribution
    ax4 = axes[1, 0]
    hourly_counts = df.groupby('hour').size()
    ax4.bar(hourly_counts.index, hourly_counts.values, color='coral')
    ax4.set_xlabel('Hour (UTC)')
    ax4.set_ylabel('Earthquake Count')
    ax4.set_title('Earthquakes by Hour', fontsize=12, fontweight='bold')

    # Subplot 5: Year × Month heatmap
    ax5 = axes[1, 1]
    pivot_table = df.groupby(['year', 'month']).size().unstack(fill_value=0)
    sns.heatmap(pivot_table, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Count'})
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Year')
    ax5.set_title('Year × Month Heatmap', fontsize=12, fontweight='bold')

    # Subplot 6: Average magnitude by year
    ax6 = axes[1, 2]
    yearly_avg_mag = df.groupby('year')['Magnitude'].mean()
    ax6.plot(yearly_avg_mag.index, yearly_avg_mag.values, marker='s', linewidth=2, color='red')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Average Magnitude')
    ax6.set_title('Average Magnitude by Year', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)

    plt.suptitle('Temporal Pattern Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}temporal_patterns.png")

    return fig


def plot_magnitude_vs_energy(df):
    """
    Visualize magnitude-energy relationship.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nCreating Magnitude vs Energy plot...")

    if 'energy' not in df.columns:
        print("  Warning: 'energy' column not found")
        return None

    df_clean = df[['Magnitude', 'energy']].dropna()

    # Calculate correlation
    correlation = df_clean['Magnitude'].corr(df_clean['energy'])
    log_correlation = df_clean['Magnitude'].corr(np.log10(df_clean['energy'] + 1))

    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    # Sample if needed
    if len(df_clean) > 20000:
        df_sample = df_clean.sample(n=20000, random_state=42)
    else:
        df_sample = df_clean

    ax.scatter(df_sample['Magnitude'], df_sample['energy'], alpha=0.3, s=5, color='purple')
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude', fontsize=12)
    ax.set_ylabel('Energy (log scale)', fontsize=12)
    ax.set_title('Magnitude vs Energy (Exponential Relationship)', fontsize=14, fontweight='bold')

    # Add correlation text
    corr_text = f"Log-Correlation: {log_correlation:.4f}"
    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}magnitude_vs_energy.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}magnitude_vs_energy.png")

    return fig


def analyze_quality_relationships(df):
    """
    Analyze relationships between quality metrics.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nAnalyzing Quality Metric Relationships...")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Subplot 1: nst vs horizontalError
    ax1 = axes[0]
    if 'nst' in df.columns and 'horizontalError' in df.columns:
        df_clean = df[['nst', 'horizontalError']].dropna()
        if len(df_clean) > 10000:
            df_clean = df_clean.sample(n=10000, random_state=42)
        ax1.scatter(df_clean['nst'], df_clean['horizontalError'], alpha=0.3, s=5, color='blue')
        corr = df_clean['nst'].corr(df_clean['horizontalError'])
        ax1.set_xlabel('Number of Stations (nst)')
        ax1.set_ylabel('Horizontal Error')
        ax1.set_title(f'nst vs Horizontal Error\n(r = {corr:.3f})', fontsize=11, fontweight='bold')

    # Subplot 2: gap vs depthError
    ax2 = axes[1]
    if 'gap' in df.columns and 'depthError' in df.columns:
        df_clean = df[['gap', 'depthError']].dropna()
        if len(df_clean) > 10000:
            df_clean = df_clean.sample(n=10000, random_state=42)
        ax2.scatter(df_clean['gap'], df_clean['depthError'], alpha=0.3, s=5, color='green')
        corr = df_clean['gap'].corr(df_clean['depthError'])
        ax2.set_xlabel('Azimuthal Gap')
        ax2.set_ylabel('Depth Error')
        ax2.set_title(f'Gap vs Depth Error\n(r = {corr:.3f})', fontsize=11, fontweight='bold')

    plt.suptitle('Quality Metrics Relationships', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}quality_relationships.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}quality_relationships.png")

    return fig


# ============================================================
# SECTION 6: MULTIVARIATE ANALYSIS FUNCTIONS
# ============================================================

def create_correlation_matrix(df):
    """
    ★ KEY DELIVERABLE: Create and visualize correlation matrix.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Correlation matrix
    """
    print("\n★ Creating Correlation Matrix (Key Deliverable)...")

    # Select numerical columns
    num_cols = ['Latitude', 'Longitude', 'Depth', 'Magnitude',
                'nst', 'gap', 'rms', 'horizontalError', 'depthError',
                'year', 'month', 'hour', 'energy']
    available_cols = [col for col in num_cols if col in df.columns]

    df_numeric = df[available_cols].dropna()

    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Draw heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8}, ax=ax)

    ax.set_title('★ Correlation Matrix - Numerical Variables', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}correlation_matrix.png")

    # Find strongest correlations (excluding self-correlation)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    top_corrs = corr_df.nlargest(10, 'abs_corr')

    print("\n  Top 10 Strongest Correlations:")
    for _, row in top_corrs.iterrows():
        print(f"    {row['var1']} ↔ {row['var2']}: {row['correlation']:.3f}")

    return corr_matrix


def create_pair_plot(df, columns, hue_col):
    """
    Create pair plot for selected variables.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to include
        hue_col (str): Column for coloring

    Returns:
        matplotlib.figure.Figure
    """
    print("\nCreating Pair Plot...")

    available_cols = [col for col in columns if col in df.columns]
    if hue_col not in df.columns:
        hue_col = None

    # Sample data for pair plot (too slow with large data)
    if len(df) > 5000:
        df_sample = df.sample(n=5000, random_state=42)
    else:
        df_sample = df

    plot_cols = available_cols + ([hue_col] if hue_col else [])
    df_plot = df_sample[plot_cols].dropna()

    g = sns.pairplot(df_plot, hue=hue_col, diag_kind='kde',
                     plot_kws={'alpha': 0.5, 's': 20},
                     palette='husl')
    g.fig.suptitle('Pair Plot - Key Variables', fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(f'{OUTPUT_DIR}pair_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}pair_plot.png")

    return g


def perform_pca_analysis(df):
    """
    ★ KEY DELIVERABLE: Perform PCA and visualize results.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        tuple: (PCA object, transformed data, Figure)
    """
    print("\n★ Performing PCA Analysis (Key Deliverable)...")

    # Select numerical columns for PCA (exclude temporal columns that are ordinal)
    pca_cols = ['Latitude', 'Longitude', 'Depth', 'Magnitude',
                'nst', 'gap', 'rms', 'horizontalError', 'depthError', 'energy']
    available_cols = [col for col in pca_cols if col in df.columns]

    # Prepare data
    df_pca = df[available_cols].dropna()

    # Also keep magnitude_category for coloring
    if 'magnitude_category' in df.columns:
        idx = df_pca.index
        mag_cat = df.loc[idx, 'magnitude_category']
    else:
        mag_cat = None

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca)

    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(var_explained)

    # Find n_components for 95% variance
    n_components_95 = np.argmax(cumulative_var >= 0.95) + 1

    # Create figure with 4 subplots
    fig = plt.figure(figsize=FIGSIZE_XLARGE)

    # Subplot 1: Scree plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(range(1, len(var_explained) + 1), var_explained * 100,
            alpha=0.7, color='steelblue', label='Individual')
    ax1.plot(range(1, len(var_explained) + 1), cumulative_var * 100,
             'ro-', linewidth=2, label='Cumulative')
    ax1.axhline(y=95, color='gray', linestyle='--', label='95% threshold')
    ax1.axvline(x=n_components_95, color='green', linestyle='--',
                label=f'{n_components_95} components for 95%')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)

    # Subplot 2: Cumulative variance
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.fill_between(range(1, len(cumulative_var) + 1), cumulative_var * 100, alpha=0.3)
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, 'b-o', linewidth=2)
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')

    # Subplot 3: 2D scatter of PC1 vs PC2
    ax3 = fig.add_subplot(2, 2, 3)

    # Sample for visualization if needed
    if len(X_pca) > 10000:
        sample_idx = np.random.choice(len(X_pca), 10000, replace=False)
        X_pca_sample = X_pca[sample_idx]
        mag_cat_sample = mag_cat.iloc[sample_idx] if mag_cat is not None else None
    else:
        X_pca_sample = X_pca
        mag_cat_sample = mag_cat

    if mag_cat_sample is not None:
        cat_order = ['Micro', 'Minor', 'Small', 'Light', 'Moderate', 'Strong', 'Major', 'Great']
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(cat_order)))
        color_map = {cat: colors[i] for i, cat in enumerate(cat_order)}

        for cat in cat_order:
            mask = mag_cat_sample == cat
            if mask.sum() > 0:
                ax3.scatter(X_pca_sample[mask, 0], X_pca_sample[mask, 1],
                           alpha=0.5, s=10, label=cat, color=color_map.get(cat, 'gray'))
        ax3.legend(title='Category', fontsize=8, bbox_to_anchor=(1.02, 1))
    else:
        ax3.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], alpha=0.3, s=5)

    ax3.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
    ax3.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
    ax3.set_title('PCA - First Two Components', fontsize=12, fontweight='bold')

    # Subplot 4: Loadings heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=available_cols
    )
    # Show only first 5 components
    loadings_display = loadings.iloc[:, :5]
    sns.heatmap(loadings_display, cmap='RdBu_r', center=0, annot=True,
                fmt='.2f', ax=ax4, annot_kws={'size': 8})
    ax4.set_title('Feature Loadings (First 5 PCs)', fontsize=12, fontweight='bold')

    plt.suptitle('★ PCA Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}pca_analysis.png")
    print(f"  Components for 95% variance: {n_components_95}")
    print(f"  First 3 PCs explain: {cumulative_var[2]*100:.1f}% variance")

    return pca, X_pca, loadings


def perform_grouped_analysis(df):
    """
    Compare statistics across different groups.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        dict: Dictionary of grouped statistics
    """
    print("\nPerforming Grouped Analysis...")

    grouped_stats = {}

    # Group by magnitude_category
    if 'magnitude_category' in df.columns:
        mag_grouped = df.groupby('magnitude_category').agg({
            'Depth': ['mean', 'std'],
            'energy': ['mean', 'median'],
            'Magnitude': 'count'
        }).round(2)
        grouped_stats['by_magnitude_category'] = mag_grouped

    # Group by depth_category
    if 'depth_category' in df.columns:
        depth_grouped = df.groupby('depth_category').agg({
            'Magnitude': ['mean', 'std', 'count'],
            'energy': ['mean', 'median']
        }).round(2)
        grouped_stats['by_depth_category'] = depth_grouped

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    # Subplot 1: Mean depth by magnitude category
    ax1 = axes[0, 0]
    if 'magnitude_category' in df.columns:
        cat_order = ['Micro', 'Minor', 'Small', 'Light', 'Moderate', 'Strong', 'Major', 'Great']
        df_plot = df[df['magnitude_category'].isin(cat_order)].copy()
        means = df_plot.groupby('magnitude_category')['Depth'].mean()
        means = means.reindex([c for c in cat_order if c in means.index])
        ax1.bar(means.index, means.values, color='teal')
        ax1.set_xlabel('Magnitude Category')
        ax1.set_ylabel('Mean Depth (km)')
        ax1.set_title('Mean Depth by Magnitude Category', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

    # Subplot 2: Mean magnitude by depth category
    ax2 = axes[0, 1]
    if 'depth_category' in df.columns:
        cat_order = ['Shallow', 'Intermediate', 'Deep', 'Very Deep']
        df_plot = df[df['depth_category'].isin(cat_order)].copy()
        means = df_plot.groupby('depth_category')['Magnitude'].mean()
        means = means.reindex([c for c in cat_order if c in means.index])
        ax2.bar(means.index, means.values, color='coral')
        ax2.set_xlabel('Depth Category')
        ax2.set_ylabel('Mean Magnitude')
        ax2.set_title('Mean Magnitude by Depth Category', fontsize=11, fontweight='bold')

    # Subplot 3: Boxplot of magnitude by depth category
    ax3 = axes[1, 0]
    if 'depth_category' in df.columns:
        cat_order = ['Shallow', 'Intermediate', 'Deep', 'Very Deep']
        df_plot = df[df['depth_category'].isin(cat_order)].copy()
        df_plot['depth_category'] = pd.Categorical(df_plot['depth_category'], categories=cat_order, ordered=True)
        df_plot.boxplot(column='Magnitude', by='depth_category', ax=ax3)
        ax3.set_xlabel('Depth Category')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Magnitude Distribution by Depth Category', fontsize=11, fontweight='bold')
        plt.suptitle('')  # Remove automatic title

    # Subplot 4: Count by status
    ax4 = axes[1, 1]
    if 'status' in df.columns:
        status_counts = df['status'].value_counts()
        ax4.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
               colors=plt.cm.Pastel1(np.linspace(0, 1, len(status_counts))))
        ax4.set_title('Events by Review Status', fontsize=11, fontweight='bold')

    plt.suptitle('Grouped Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}grouped_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}grouped_analysis.png")

    return grouped_stats


# ============================================================
# SECTION 7: GEOSPATIAL ANALYSIS FUNCTIONS
# ============================================================

def plot_earthquake_locations(df):
    """
    Create scatter plot of earthquake locations.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nCreating Earthquake Location Map...")

    # Sample for visualization
    if len(df) > 50000:
        df_sample = df.sample(n=50000, random_state=42)
    else:
        df_sample = df

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    # Color and size by magnitude
    if 'magnitude_category' in df_sample.columns:
        cat_order = ['Micro', 'Minor', 'Small', 'Light', 'Moderate', 'Strong', 'Major', 'Great']
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(cat_order)))
        color_map = {cat: colors[i] for i, cat in enumerate(cat_order)}

        for cat in cat_order:
            subset = df_sample[df_sample['magnitude_category'] == cat]
            if len(subset) > 0:
                # Scale point size by magnitude category
                size_scale = (cat_order.index(cat) + 1) * 2
                ax.scatter(subset['Longitude'], subset['Latitude'],
                          alpha=0.4, s=size_scale, label=cat,
                          color=color_map.get(cat, 'gray'))

        ax.legend(title='Magnitude', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    else:
        scatter = ax.scatter(df_sample['Longitude'], df_sample['Latitude'],
                            c=df_sample['Magnitude'], cmap='YlOrRd',
                            alpha=0.4, s=5)
        plt.colorbar(scatter, ax=ax, label='Magnitude')

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Global Earthquake Distribution', fontsize=14, fontweight='bold')

    # Add reference lines for equator and prime meridian
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}earthquake_locations.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}earthquake_locations.png")

    return fig


def create_density_heatmap(df):
    """
    Create heatmap showing earthquake density.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        matplotlib.figure.Figure
    """
    print("\nCreating Earthquake Density Heatmap...")

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    # Create 2D histogram
    h = ax.hist2d(df['Longitude'], df['Latitude'],
                  bins=[180, 90], cmap='hot_r',
                  cmin=1)  # Minimum count to show

    plt.colorbar(h[3], ax=ax, label='Earthquake Count')

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Earthquake Density Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}earthquake_density.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}earthquake_density.png")

    return fig


def analyze_regional_patterns(df):
    """
    Analyze patterns by geographic region.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Summary statistics by region
    """
    print("\nAnalyzing Regional Patterns...")

    # Define broad regions based on coordinates
    def assign_region(row):
        lat, lon = row['Latitude'], row['Longitude']
        if lat >= 45:
            if lon < -100 or lon > 100:
                return 'Alaska/North Pacific'
            else:
                return 'Northern Hemisphere'
        elif lat <= -45:
            return 'Southern Hemisphere'
        elif -30 < lat < 45:
            if -170 < lon < -60:
                return 'Americas'
            elif -60 < lon < 60:
                return 'Europe/Africa'
            else:
                return 'Asia/Pacific'
        else:
            return 'Other'

    df_copy = df.copy()
    df_copy['region'] = df_copy.apply(assign_region, axis=1)

    # Summary by region
    region_stats = df_copy.groupby('region').agg({
        'Magnitude': ['count', 'mean', 'max'],
        'Depth': ['mean', 'max']
    }).round(2)

    print("\n  Regional Statistics:")
    print(region_stats.to_string())

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Subplot 1: Count by region
    ax1 = axes[0]
    region_counts = df_copy['region'].value_counts()
    ax1.barh(region_counts.index, region_counts.values, color='steelblue')
    ax1.set_xlabel('Earthquake Count')
    ax1.set_title('Earthquakes by Region', fontsize=12, fontweight='bold')

    # Add count labels
    for idx, val in enumerate(region_counts.values):
        ax1.text(val, idx, f' {val:,}', va='center', fontsize=9)

    # Subplot 2: Mean magnitude by region
    ax2 = axes[1]
    region_mag = df_copy.groupby('region')['Magnitude'].mean().sort_values(ascending=True)
    ax2.barh(region_mag.index, region_mag.values, color='coral')
    ax2.set_xlabel('Mean Magnitude')
    ax2.set_title('Mean Magnitude by Region', fontsize=12, fontweight='bold')

    plt.suptitle('Regional Pattern Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}regional_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {OUTPUT_DIR}regional_patterns.png")

    return region_stats


# ============================================================
# SECTION 8: MAIN EXECUTION
# ============================================================

def generate_summary_report(df, overview, missing_df, correlations, pca_results):
    """
    Generate final summary report.

    Parameters:
        df: DataFrame
        overview: Overview dictionary
        missing_df: Missing values DataFrame
        correlations: Correlation results
        pca_results: PCA results tuple
    """
    print("\n" + "="*60)
    print("EDA SUMMARY REPORT")
    print("="*60)

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("EARTHQUAKE DATA EDA SUMMARY REPORT")
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 60)

    # Dataset Overview
    summary_lines.append("\n1. DATASET OVERVIEW")
    summary_lines.append("-" * 40)
    summary_lines.append(f"   Total Records: {overview['rows']:,}")
    summary_lines.append(f"   Total Columns: {overview['columns']}")
    summary_lines.append(f"   Time Period: {df['year'].min()} - {df['year'].max()}")
    summary_lines.append(f"   Memory Usage: {overview['memory_usage_mb']:.2f} MB")

    # Data Quality
    summary_lines.append("\n2. DATA QUALITY")
    summary_lines.append("-" * 40)
    missing_cols = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_cols) > 0:
        summary_lines.append(f"   Columns with Missing Values: {len(missing_cols)}")
        for _, row in missing_cols.head(5).iterrows():
            summary_lines.append(f"     - {row['Column']}: {row['Missing Percentage']:.1f}%")
    else:
        summary_lines.append("   No missing values found")

    # Key Statistics
    summary_lines.append("\n3. KEY STATISTICS")
    summary_lines.append("-" * 40)
    summary_lines.append(f"   Magnitude Range: {df['Magnitude'].min():.1f} - {df['Magnitude'].max():.1f}")
    summary_lines.append(f"   Mean Magnitude: {df['Magnitude'].mean():.2f}")
    summary_lines.append(f"   Depth Range: {df['Depth'].min():.1f} - {df['Depth'].max():.1f} km")
    summary_lines.append(f"   Mean Depth: {df['Depth'].mean():.1f} km")

    # Correlations
    if correlations:
        summary_lines.append("\n4. KEY CORRELATIONS")
        summary_lines.append("-" * 40)
        summary_lines.append(f"   Depth-Magnitude Pearson: {correlations['pearson']:.4f}")
        summary_lines.append(f"   Depth-Magnitude Spearman: {correlations['spearman']:.4f}")

    # PCA Results
    if pca_results:
        pca, _, loadings = pca_results
        summary_lines.append("\n5. PCA INSIGHTS")
        summary_lines.append("-" * 40)
        var_explained = pca.explained_variance_ratio_
        summary_lines.append(f"   PC1 Variance: {var_explained[0]*100:.1f}%")
        summary_lines.append(f"   PC2 Variance: {var_explained[1]*100:.1f}%")
        summary_lines.append(f"   Top 3 PCs: {sum(var_explained[:3])*100:.1f}% total variance")

    # Category Distribution
    summary_lines.append("\n6. CATEGORY DISTRIBUTION")
    summary_lines.append("-" * 40)
    if 'magnitude_category' in df.columns:
        cat_counts = df['magnitude_category'].value_counts()
        for cat, count in cat_counts.head(5).items():
            pct = (count / len(df)) * 100
            summary_lines.append(f"   {cat}: {count:,} ({pct:.1f}%)")

    # Print summary
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save summary
    with open('./outputs/eda_summary.txt', 'w') as f:
        f.write(summary_text)

    print(f"\n✓ Summary saved: ./outputs/eda_summary.txt")


def main():
    """
    Main execution function - runs complete EDA pipeline.
    """
    print("=" * 60)
    print("EARTHQUAKE DATA - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Step 1: Load Data
    print("\n[1/8] Loading data...")
    df = load_data(DATA_PATH)

    # Step 2: Data Overview
    print("\n[2/8] Generating data overview...")
    overview = get_data_overview(df)
    stats = get_descriptive_stats(df)

    # Step 3: Data Quality Assessment
    print("\n[3/8] Assessing data quality...")
    missing_df = analyze_missing_values(df)
    duplicates = check_duplicates(df)
    consistency = validate_data_consistency(df)
    outliers = detect_outliers(df, ['Magnitude', 'Depth', 'energy'])

    # Step 4: Univariate Analysis
    print("\n[4/8] Performing univariate analysis...")
    mag_stats = plot_magnitude_distribution(df)      # ★ KEY
    depth_stats = plot_depth_distribution(df)
    plot_energy_distribution(df)
    plot_categorical_distributions(df)
    analyze_quality_metrics(df)

    # Step 5: Bivariate Analysis
    print("\n[5/8] Performing bivariate analysis...")
    correlations = plot_depth_vs_magnitude(df)       # ★ KEY
    analyze_temporal_patterns(df)
    plot_magnitude_vs_energy(df)
    analyze_quality_relationships(df)

    # Step 6: Multivariate Analysis
    print("\n[6/8] Performing multivariate analysis...")
    corr_matrix = create_correlation_matrix(df)      # ★ KEY
    create_pair_plot(df, ['Magnitude', 'Depth', 'energy', 'rms'], 'magnitude_category')
    pca_results = perform_pca_analysis(df)           # ★ KEY
    grouped_stats = perform_grouped_analysis(df)

    # Step 7: Geospatial Analysis
    print("\n[7/8] Performing geospatial analysis...")
    plot_earthquake_locations(df)
    create_density_heatmap(df)
    analyze_regional_patterns(df)

    # Step 8: Summary
    print("\n[8/8] Generating summary...")
    generate_summary_report(df, overview, missing_df, correlations, pca_results)

    print("\n" + "=" * 60)
    print("EDA COMPLETE! Check outputs/figures/ for visualizations")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()

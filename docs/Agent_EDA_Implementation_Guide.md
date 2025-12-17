# 🤖 Agent Implementation Guide: Earthquake EDA

## Overview

This document provides step-by-step instructions for an AI agent to implement the Exploratory Data Analysis (EDA) on the earthquake dataset. **Important:** Create Python scripts first, convert to notebook only after completion.

---

## 📁 Project Structure

```
.
├── data/
│   └── processed/
│       └── final_earthquake_data_2002_2025.csv
├── docs/
│   ├── EDA_Implementation_Plan_Earthquake_Data.md
│   └── Agent_EDA_Implementation_Guide.md  (this file)
├── src/
│   └── eda_earthquake.py  (CREATE THIS FIRST)
├── outputs/
│   └── figures/           (save all plots here)
└── notebooks/
    └── EDA_Earthquake_Analysis.ipynb  (CREATE LAST - after all code works)
```

---

## 🎯 Agent Mission

**Primary Goal:** Implement complete EDA analysis in Python first, then convert to Jupyter Notebook.

**Workflow:**
1. **Phase A:** Create Python script (`src/eda_earthquake.py`) with all analysis functions
2. **Phase B:** Test and validate all code works correctly
3. **Phase C:** Convert to Jupyter Notebook with markdown explanations (FINAL STEP)

**Required Deliverables:**
1. ★ Histogram for Magnitude distribution
2. ★ Scatter plot for Depth analysis
3. ★ Correlation Matrix heatmap
4. ★ PCA analysis with visualizations
5. Markdown explanations (in final notebook)

---

## 📋 Pre-Implementation Checklist

Before starting, the agent MUST:

- [ ] Verify dataset exists at `./data/processed/final_earthquake_data_2002_2025.csv`
- [ ] Create directory structure: `src/`, `outputs/figures/`, `notebooks/`
- [ ] Read reference plan at `./docs/EDA_Implementation_Plan_Earthquake_Data.md`
- [ ] Verify Python environment has required libraries

**Required Libraries:**
```
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
```

---

# 🐍 PHASE A: PYTHON SCRIPT IMPLEMENTATION

---

## STEP 1: Create Main Python File Structure

**Action:** Create `src/eda_earthquake.py` with the following structure.

**File Structure:**
```python
"""
EDA Earthquake Analysis
=======================
Comprehensive Exploratory Data Analysis for Earthquake Dataset (2002-2025)

Author: [Agent]
Date: [Current Date]
"""

# ============================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================

# ============================================================
# SECTION 2: DATA LOADING FUNCTIONS
# ============================================================

# ============================================================
# SECTION 3: DATA QUALITY FUNCTIONS
# ============================================================

# ============================================================
# SECTION 4: UNIVARIATE ANALYSIS FUNCTIONS
# ============================================================

# ============================================================
# SECTION 5: BIVARIATE ANALYSIS FUNCTIONS
# ============================================================

# ============================================================
# SECTION 6: MULTIVARIATE ANALYSIS FUNCTIONS
# ============================================================

# ============================================================
# SECTION 7: GEOSPATIAL ANALYSIS FUNCTIONS
# ============================================================

# ============================================================
# SECTION 8: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
```

---

## STEP 2: Implement Section 1 - Imports and Configuration

**Action:** Add all required imports and global settings.

**Instructions:**
1. Import data manipulation libraries (pandas, numpy)
2. Import visualization libraries (matplotlib, seaborn)
3. Import statistical libraries (scipy.stats)
4. Import ML libraries (sklearn for PCA)
5. Set global plot styles and configurations
6. Define constants (file paths, figure sizes, color palettes)

**Constants to Define:**
```python
# File paths
DATA_PATH = './data/processed/final_earthquake_data_2002_2025.csv'
OUTPUT_DIR = './outputs/figures/'

# Figure settings
FIGSIZE_SMALL = (8, 6)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_LARGE = (16, 10)

# Column groups for analysis
NUMERICAL_COLS = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 
                  'nst', 'gap', 'rms', 'horizontalError', 'depthError', 
                  'year', 'month', 'day', 'hour', 'energy']
CATEGORICAL_COLS = ['magnitude_category', 'depth_category', 'magType', 
                    'type', 'status', 'net']
```

---

## STEP 3: Implement Section 2 - Data Loading Functions

**Action:** Create functions for loading and initial inspection.

**Functions to Create:**

### Function 3.1: `load_data(filepath)`
- **Purpose:** Load CSV file with proper datetime parsing
- **Input:** filepath (string)
- **Output:** pandas DataFrame
- **Requirements:**
  - Parse 'Timestamp' and 'updated' as datetime
  - Handle any encoding issues
  - Print confirmation message with shape

### Function 3.2: `get_data_overview(df)`
- **Purpose:** Generate comprehensive data overview
- **Input:** DataFrame
- **Output:** Dictionary with overview statistics
- **Requirements:**
  - Return: shape, dtypes, memory usage
  - Return: numerical columns list
  - Return: categorical columns list
  - Print formatted summary

### Function 3.3: `get_descriptive_stats(df)`
- **Purpose:** Generate descriptive statistics
- **Input:** DataFrame
- **Output:** Tuple of (numerical_stats_df, categorical_stats_df)
- **Requirements:**
  - Use df.describe() for numerical
  - Use df.describe(include='object') for categorical
  - Include value_counts for key categorical columns

---

## STEP 4: Implement Section 3 - Data Quality Functions

**Action:** Create functions for data quality assessment.

**Functions to Create:**

### Function 4.1: `analyze_missing_values(df)`
- **Purpose:** Analyze and visualize missing values
- **Input:** DataFrame
- **Output:** DataFrame with missing value statistics
- **Requirements:**
  - Calculate count and percentage per column
  - Sort by missing percentage descending
  - Create bar chart visualization
  - Save figure to `OUTPUT_DIR/missing_values.png`

### Function 4.2: `check_duplicates(df)`
- **Purpose:** Check for duplicate records
- **Input:** DataFrame
- **Output:** Dictionary with duplicate counts
- **Requirements:**
  - Check exact duplicates
  - Check duplicates by key fields: ['Timestamp', 'Latitude', 'Longitude', 'Magnitude']
  - Return counts for both

### Function 4.3: `validate_data_consistency(df)`
- **Purpose:** Validate data ranges and consistency
- **Input:** DataFrame
- **Output:** Dictionary with validation results
- **Requirements:**
  - Check Latitude range: -90 to 90
  - Check Longitude range: -180 to 180
  - Check Depth: >= 0
  - Check Magnitude: typically -2 to 10
  - Check Year: 2002 to 2025
  - Return boolean for each check + any outliers found

### Function 4.4: `detect_outliers(df, columns, method='iqr')`
- **Purpose:** Detect outliers using IQR or Z-score method
- **Input:** DataFrame, list of columns, method
- **Output:** DataFrame with outlier flags
- **Requirements:**
  - IQR method: Q1 - 1.5*IQR, Q3 + 1.5*IQR
  - Z-score method: |z| > 3
  - Create boxplot visualization for specified columns
  - Save figure to `OUTPUT_DIR/outliers_boxplot.png`

---

## STEP 5: Implement Section 4 - Univariate Analysis Functions

**Action:** Create functions for single variable analysis.

**Functions to Create:**

### Function 5.1: `plot_magnitude_distribution(df)` ★ KEY DELIVERABLE
- **Purpose:** Create comprehensive magnitude distribution visualization
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Create figure with 3 subplots (1 row, 3 columns)
  - Subplot 1: Histogram with KDE overlay (bins=30)
  - Subplot 2: Boxplot showing quartiles
  - Subplot 3: Bar chart of magnitude_category counts
  - Add title, labels, and statistics annotations
  - Save figure to `OUTPUT_DIR/magnitude_distribution.png`
  - Return statistics: mean, median, std, skewness, kurtosis

### Function 5.2: `plot_depth_distribution(df)` ★ KEY DELIVERABLE
- **Purpose:** Create comprehensive depth distribution visualization
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Create figure with 3 subplots
  - Subplot 1: Histogram with KDE
  - Subplot 2: Boxplot
  - Subplot 3: Bar chart of depth_category counts
  - Save figure to `OUTPUT_DIR/depth_distribution.png`
  - Return statistics

### Function 5.3: `plot_energy_distribution(df)`
- **Purpose:** Analyze energy distribution (log-scale)
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Use log transformation for visualization
  - Show relationship with magnitude categories
  - Save figure to `OUTPUT_DIR/energy_distribution.png`

### Function 5.4: `plot_categorical_distributions(df)`
- **Purpose:** Create bar charts for all categorical variables
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Create subplot for each categorical column
  - Show counts and percentages
  - Save figure to `OUTPUT_DIR/categorical_distributions.png`

### Function 5.5: `analyze_quality_metrics(df)`
- **Purpose:** Analyze distribution of quality metrics
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Analyze: nst, gap, rms, horizontalError, depthError
  - Create histogram for each
  - Save figure to `OUTPUT_DIR/quality_metrics.png`

---

## STEP 6: Implement Section 5 - Bivariate Analysis Functions

**Action:** Create functions for two-variable analysis.

**Functions to Create:**

### Function 6.1: `plot_depth_vs_magnitude(df)` ★ KEY DELIVERABLE
- **Purpose:** Create scatter plot of Depth vs Magnitude
- **Input:** DataFrame
- **Output:** Figure object, correlation coefficients
- **Requirements:**
  - Main scatter plot with alpha=0.5 for transparency
  - Color points by magnitude_category
  - Add regression line with equation
  - Calculate Pearson and Spearman correlations
  - Add correlation values as text annotation
  - Optional: Create hexbin version for dense data
  - Save figure to `OUTPUT_DIR/depth_vs_magnitude_scatter.png`

### Function 6.2: `analyze_temporal_patterns(df)`
- **Purpose:** Analyze earthquake patterns over time
- **Input:** DataFrame
- **Output:** Figure object with multiple temporal views
- **Requirements:**
  - Subplot 1: Yearly earthquake counts (line plot)
  - Subplot 2: Monthly pattern (bar chart, aggregated across years)
  - Subplot 3: Day of week distribution
  - Subplot 4: Hourly distribution
  - Subplot 5: Year × Month heatmap
  - Save figure to `OUTPUT_DIR/temporal_patterns.png`

### Function 6.3: `plot_magnitude_vs_energy(df)`
- **Purpose:** Visualize magnitude-energy relationship
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Scatter plot with log-scale y-axis
  - Show exponential relationship
  - Calculate and display correlation
  - Save figure to `OUTPUT_DIR/magnitude_vs_energy.png`

### Function 6.4: `analyze_quality_relationships(df)`
- **Purpose:** Analyze relationships between quality metrics
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - nst vs horizontalError scatter
  - gap vs depthError scatter
  - Calculate correlations
  - Save figure to `OUTPUT_DIR/quality_relationships.png`

---

## STEP 7: Implement Section 6 - Multivariate Analysis Functions

**Action:** Create functions for multi-variable analysis.

**Functions to Create:**

### Function 7.1: `create_correlation_matrix(df)` ★ KEY DELIVERABLE
- **Purpose:** Create and visualize correlation matrix
- **Input:** DataFrame
- **Output:** Correlation matrix DataFrame, Figure object
- **Requirements:**
  - Select numerical columns only (from NUMERICAL_COLS constant)
  - Calculate Pearson correlation matrix
  - Create heatmap with seaborn
  - Use 'coolwarm' colormap, center at 0
  - Add annotations with correlation values
  - Mask upper triangle (optional but cleaner)
  - Highlight strong correlations (|r| > 0.7)
  - Save figure to `OUTPUT_DIR/correlation_matrix.png`
  - Return DataFrame with correlation values

### Function 7.2: `create_pair_plot(df, columns, hue_col)`
- **Purpose:** Create pair plot for selected variables
- **Input:** DataFrame, list of columns (4-6), hue column name
- **Output:** Figure object
- **Requirements:**
  - Use seaborn pairplot
  - Color by magnitude_category or depth_category
  - Select columns: ['Magnitude', 'Depth', 'energy', 'rms', 'gap']
  - Save figure to `OUTPUT_DIR/pair_plot.png`

### Function 7.3: `perform_pca_analysis(df)` ★ KEY DELIVERABLE
- **Purpose:** Perform PCA and visualize results
- **Input:** DataFrame
- **Output:** PCA object, transformed data, Figure object
- **Requirements:**

  **Step 7.3.1: Data Preparation**
  - Select numerical columns for PCA (exclude year, month, day, hour)
  - Handle missing values (drop or impute)
  - Standardize features using StandardScaler
  
  **Step 7.3.2: Fit PCA**
  - Fit PCA with all components first
  - Calculate explained variance ratio
  - Determine n_components for 95% variance explained
  
  **Step 7.3.3: Create Visualizations**
  - Subplot 1: Scree plot (explained variance by component)
  - Subplot 2: Cumulative variance plot
  - Subplot 3: 2D scatter of PC1 vs PC2, colored by magnitude_category
  - Subplot 4: Loadings heatmap (feature contributions to each PC)
  
  **Step 7.3.4: Output**
  - Save figure to `OUTPUT_DIR/pca_analysis.png`
  - Return: PCA object, transformed data, loadings DataFrame

### Function 7.4: `perform_grouped_analysis(df)`
- **Purpose:** Compare statistics across different groups
- **Input:** DataFrame
- **Output:** Dictionary of grouped statistics
- **Requirements:**
  - Group by magnitude_category: compare depth, energy means
  - Group by depth_category: compare magnitude distribution
  - Group by net: compare data quality metrics
  - Create visualization comparing groups
  - Save figure to `OUTPUT_DIR/grouped_analysis.png`

---

## STEP 8: Implement Section 7 - Geospatial Analysis Functions

**Action:** Create functions for spatial analysis.

**Functions to Create:**

### Function 8.1: `plot_earthquake_locations(df)`
- **Purpose:** Create scatter plot of earthquake locations
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Plot Longitude vs Latitude
  - Color by magnitude_category
  - Size points by magnitude (scaled)
  - Add world map boundaries if possible (optional)
  - Save figure to `OUTPUT_DIR/earthquake_locations.png`

### Function 8.2: `create_density_heatmap(df)`
- **Purpose:** Create heatmap showing earthquake density
- **Input:** DataFrame
- **Output:** Figure object
- **Requirements:**
  - Use 2D histogram or KDE
  - Show hotspots clearly
  - Save figure to `OUTPUT_DIR/earthquake_density.png`

### Function 8.3: `analyze_regional_patterns(df)`
- **Purpose:** Analyze patterns by geographic region
- **Input:** DataFrame
- **Output:** Summary statistics by region
- **Requirements:**
  - Define regions based on coordinates or extract from 'place'
  - Compare magnitude and depth distributions by region
  - Save figure to `OUTPUT_DIR/regional_patterns.png`

---

## STEP 9: Implement Section 8 - Main Execution

**Action:** Create main function that orchestrates all analysis.

**Function: `main()`**

**Requirements:**
```python
def main():
    """
    Main execution function - runs complete EDA pipeline
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
    missing = analyze_missing_values(df)
    duplicates = check_duplicates(df)
    consistency = validate_data_consistency(df)
    outliers = detect_outliers(df, ['Magnitude', 'Depth', 'energy'])
    
    # Step 4: Univariate Analysis
    print("\n[4/8] Performing univariate analysis...")
    mag_stats = plot_magnitude_distribution(df)      # ★ KEY
    depth_stats = plot_depth_distribution(df)        # ★ KEY
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
    generate_summary_report(df, overview, stats, missing, 
                           correlations, pca_results)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE! Check outputs/figures/ for visualizations")
    print("=" * 60)
    
    return df  # Return for further analysis if needed
```

---

## STEP 10: Create Summary Report Function

**Action:** Create function to generate final summary.

**Function: `generate_summary_report(...)`**

**Requirements:**
- Compile key findings from all analyses
- Print formatted summary to console
- Save summary to `outputs/eda_summary.txt`
- Include:
  - Dataset overview statistics
  - Data quality summary
  - Key distribution characteristics
  - Top correlations found
  - PCA insights
  - Notable patterns discovered

---

# 📓 PHASE B: TESTING AND VALIDATION

---

## STEP 11: Test Python Script

**Action:** Run and validate the complete script.

**Instructions:**
1. Run `python src/eda_earthquake.py`
2. Verify all outputs are generated in `outputs/figures/`
3. Check for any errors or warnings
4. Validate all visualizations look correct
5. Review console output for completeness

**Validation Checklist:**
- [ ] Script runs without errors
- [ ] All 4 key deliverable figures are generated:
  - [ ] `magnitude_distribution.png`
  - [ ] `depth_vs_magnitude_scatter.png`
  - [ ] `correlation_matrix.png`
  - [ ] `pca_analysis.png`
- [ ] All other supporting figures are generated
- [ ] Summary report is created
- [ ] Statistics are reasonable and match expectations

---

# 📔 PHASE C: NOTEBOOK CONVERSION (FINAL STEP)

---

## STEP 12: Create Jupyter Notebook

**Action:** Convert Python script to Jupyter Notebook with markdown explanations.

**IMPORTANT:** Only proceed to this step after Phase A and B are complete!

**Instructions:**

### 12.1 Notebook Structure
Create `notebooks/EDA_Earthquake_Analysis.ipynb` with the following cell structure:

```
Cell 1 [Markdown]: Title and Introduction
Cell 2 [Code]: Imports
Cell 3 [Markdown]: Section 1 - Data Loading
Cell 4 [Code]: Load data
Cell 5 [Markdown]: Observations about data
Cell 6 [Markdown]: Section 2 - Data Quality
Cell 7 [Code]: Missing values analysis
Cell 8 [Markdown]: Missing values interpretation
Cell 9 [Code]: Duplicates and consistency
Cell 10 [Markdown]: Quality assessment summary
... continue pattern ...
```

### 12.2 Markdown Content Guidelines

**For each section, include:**
- Section header with emoji
- Brief explanation of what analysis does
- Code cell(s) with the analysis
- Interpretation/insights markdown cell after results

**Example Markdown Cells:**

**Introduction:**
```markdown
# 🌍 Exploratory Data Analysis: Global Earthquake Data (2002-2025)

## Project Overview
This notebook presents a comprehensive EDA of earthquake data from the USGS database, 
covering the period 2002-2025. 

## Objectives
1. Assess data quality and completeness
2. Understand magnitude and depth distributions
3. Discover temporal and spatial patterns
4. Identify correlations between seismic variables
5. Apply PCA for dimensionality reduction

## Dataset
- **Source:** USGS Earthquake Database
- **Period:** 2002-2025
- **File:** `final_earthquake_data_2002_2025.csv`
```

**After Magnitude Distribution:**
```markdown
### 📊 Magnitude Distribution Insights

**Key Findings:**
- The magnitude distribution shows [describe shape: normal/skewed/bimodal]
- Most earthquakes fall in the [X-Y] magnitude range
- Mean magnitude: X.XX, Median: X.XX
- [X]% of earthquakes are classified as "major" (≥7.0)

**Interpretation:**
[Explain what this means in seismological context]
```

### 12.3 Required Markdown Sections

Each notebook section MUST have explanatory markdown:

| Section | Required Explanation |
|---------|---------------------|
| Data Loading | Dataset description, columns explained |
| Data Quality | Summary of issues found, recommendations |
| Magnitude Distribution ★ | Statistical summary, shape interpretation |
| Depth Distribution ★ | Depth categories explanation, geological context |
| Scatter Plot ★ | Correlation interpretation, relationship explanation |
| Correlation Matrix ★ | Key correlations highlighted, implications |
| PCA Analysis ★ | Variance explained, feature importance |
| Temporal Patterns | Trends over time, seasonality findings |
| Geospatial | Geographic hotspots, regional patterns |
| Conclusions | Summary of all key findings |

---

## STEP 13: Final Review and Documentation

**Action:** Review notebook and finalize documentation.

**Checklist:**
- [ ] All cells execute without errors
- [ ] All visualizations display correctly
- [ ] Markdown explanations are clear and insightful
- [ ] Key deliverables are clearly marked (★)
- [ ] Conclusions section summarizes all findings
- [ ] Notebook is well-organized and easy to follow

---

# ✅ FINAL DELIVERABLES CHECKLIST

## Files to Produce:

| File | Location | Status |
|------|----------|--------|
| `eda_earthquake.py` | `src/` | Required |
| `magnitude_distribution.png` | `outputs/figures/` | ★ Required |
| `depth_vs_magnitude_scatter.png` | `outputs/figures/` | ★ Required |
| `correlation_matrix.png` | `outputs/figures/` | ★ Required |
| `pca_analysis.png` | `outputs/figures/` | ★ Required |
| `EDA_Earthquake_Analysis.ipynb` | `notebooks/` | Required (LAST) |
| `eda_summary.txt` | `outputs/` | Optional |

## Key Visualizations:

1. ★ **Magnitude Histogram** - Shows frequency distribution of earthquake magnitudes
2. ★ **Depth Scatter Plot** - Shows Depth vs Magnitude relationship
3. ★ **Correlation Matrix** - Heatmap of all numerical variable correlations
4. ★ **PCA Analysis** - Scree plot, 2D projection, loadings heatmap

---

# 📝 NOTES FOR AGENT

## Error Handling
- If a column doesn't exist, skip that analysis and log warning
- If visualization fails, save error message and continue
- Handle missing values appropriately for each analysis type

## Code Quality
- Add docstrings to all functions
- Use meaningful variable names
- Add comments for complex logic
- Follow PEP 8 style guidelines

## Visualization Standards
- All figures should have titles
- All axes should be labeled
- Use consistent color scheme
- Include legends where appropriate
- Set appropriate figure sizes for readability

## Performance Considerations
- For large datasets, consider sampling for pair plots
- Use efficient data types (category for categorical columns)
- Close figures after saving to free memory

---

*Guide Version: 1.0*
*Last Updated: December 2024*

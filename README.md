# Earthquake EDA: Global Seismic Data Analysis (2002-2025)

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Data Source](https://img.shields.io/badge/Data%20Source-USGS-red)](https://earthquake.usgs.gov/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

**A comprehensive exploratory data analysis of global earthquake data spanning 23 years**

[Features](#features) • [Dataset](#dataset) • [Quick Start](#quick-start) • [Key Findings](#key-findings) • [Visualizations](#visualizations) • [Requirements](#requirements)

</div>

---

## Overview

This project provides a complete exploratory data analysis (EDA) of global earthquake data from 2002 to 2025. The analysis includes comprehensive statistical summaries, temporal and spatial pattern detection, correlation analysis, principal component analysis (PCA), and stunning visualizations of global seismic activity.

**Key Highlights:**
- **2.92M+ earthquake records** analyzed across 28 variables
- **18 publication-quality visualizations** covering univariate, bivariate, and multivariate analyses
- **Deep statistical insights** including distribution analysis, correlation patterns, and dimensionality reduction
- **Geospatial analysis** revealing regional earthquake patterns and hotspots
- **PCA-based dimensionality reduction** explaining 54.6% variance with just 3 components

---

## Features

### Data Analysis Capabilities

- **Data Quality Assessment**
  - Missing value detection and analysis
  - Duplicate record identification
  - Data consistency validation
  - Outlier detection using IQR and Z-score methods

- **Univariate Analysis**
  - Magnitude distribution (right-skewed, mean: 1.72)
  - Depth distribution (mostly shallow, mean: 24.8 km)
  - Energy distribution analysis
  - Categorical variable distributions
  - Quality metrics analysis

- **Bivariate Analysis**
  - Depth vs. Magnitude correlation (Pearson r = 0.35, Spearman ρ = 0.44)
  - Temporal pattern analysis (yearly, monthly, hourly)
  - Magnitude vs. Energy relationship
  - Quality metric relationships

- **Multivariate Analysis**
  - Correlation matrix heatmap with 13 numerical variables
  - Pair plot for key relationships
  - **Principal Component Analysis (PCA)** with loadings and variance explained
  - Grouped statistical comparisons

- **Geospatial Analysis**
  - Global earthquake location mapping
  - Density heatmap visualization
  - Regional pattern analysis (Americas, Europe/Africa, Asia/Pacific, etc.)

---

## Dataset

### Source & Specifications

| Attribute | Details |
|-----------|---------|
| **Source** | USGS Earthquake Database |
| **Total Records** | 2,921,770 earthquakes |
| **Total Columns** | 28 variables |
| **Time Period** | 2002 - 2025 (23 years) |
| **File Size** | 737 MB (gzipped, not included in repo) |
| **Data Format** | CSV with datetime parsing |

### Column Categories

**Spatial Coordinates:**
- `Latitude`, `Longitude`, `place`

**Temporal Information:**
- `Timestamp`, `updated`, `year`, `month`, `day`, `hour`, `dayofweek`, `dayofyear`

**Seismic Metrics:**
- `Depth` (km), `Magnitude`, `magType`, `energy` (Joules)

**Quality Indicators:**
- `nst` (# of stations), `gap` (azimuthal gap), `rms` (root mean square error)
- `horizontalError` (km), `depthError` (km), `status`

**Source & Derived:**
- `net`, `id`, `locationSource`, `magSource`, `type`
- `magnitude_category`, `depth_category`

### Data Quality Summary

| Issue | Details |
|-------|---------|
| **Missing Values** | 5 columns with <7.2% missing (depthError highest) |
| **Duplicates** | Minimal exact duplicates; no significant key field duplicates |
| **Validation** | All coordinate, depth, and magnitude values within valid ranges |
| **Outliers** | Detected via IQR method; preserved for analysis accuracy |

---

## Key Findings

### Magnitude Analysis
- **Range:** -10.0 to 9.1 on the Richter scale
- **Mean:** 1.72 (heavily right-skewed distribution)
- **Median:** 1.50
- **Distribution:** 97.4% classified as "Small" earthquakes
- **Interpretation:** Majority of earthquakes are imperceptible; most activity clustered at low magnitudes

### Depth Characteristics
- **Range:** 0.0 to 735.8 km below surface
- **Mean:** 24.8 km (predominantly shallow earthquakes)
- **Categories:** Most earthquakes are shallow or intermediate depth
- **Pattern:** Shallow earthquakes (0-70 km) account for bulk of seismic activity

### Magnitude-Depth Relationship
- **Pearson Correlation:** r = 0.35 (moderate positive)
- **Spearman Correlation:** ρ = 0.44 (moderate, rank-based)
- **Interpretation:** Deeper earthquakes tend to be slightly stronger, but relationship is not deterministic

### Energy Distribution
- **Strongly Exponential:** Follows power-law distribution expected for seismic energy
- **Log-Correlation with Magnitude:** High correlation indicates energy scales exponentially with magnitude
- **Range:** Spans multiple orders of magnitude (1e6 to 1e15+ Joules)

### Magnitude Category Distribution
| Category | Count | Percentage |
|----------|-------|-----------|
| Small | 2,844,801 | 97.4% |
| Moderate | 39,097 | 1.3% |
| Great | 34,403 | 1.2% |
| Strong | 3,146 | 0.1% |
| Major | 323 | 0.01% |

### Principal Component Analysis (PCA)
- **PC1 Variance:** 29.7% (led by depth and horizontal error)
- **PC2 Variance:** 13.8%
- **PC3 Variance:** 11.1%
- **Top 3 Components:** Explain 54.6% of total variance
- **Interpretation:** Data is high-dimensional; multiple independent factors drive earthquake characteristics

---

## Visualizations

The project generates **18 publication-quality PNG visualizations** (150 DPI):

### Starred Key Deliverables
1. **magnitude_distribution.png** ⭐
   - Histogram + KDE + Boxplot + Category breakdown
   - Reveals extreme right skew in magnitude distribution

2. **depth_vs_magnitude_scatter.png** ⭐
   - Scatter plot colored by magnitude category
   - Hexbin density visualization
   - Regression line overlay showing relationship

3. **correlation_matrix.png** ⭐
   - 13x13 heatmap with annotations
   - Identifies strongest variable relationships
   - Reveals multicollinearity patterns

4. **pca_analysis.png** ⭐
   - Scree plot showing variance explained
   - 2D scatter plot of first two components
   - Feature loadings heatmap

### Additional Visualizations
- `depth_distribution.png` - Depth analysis with distributions
- `energy_distribution.png` - Log-scale energy patterns
- `categorical_distributions.png` - Value counts for categorical variables
- `quality_metrics.png` - Statistical and quality indicator distributions
- `temporal_patterns.png` - 6-panel temporal analysis (yearly, monthly, hourly, heatmap)
- `magnitude_vs_energy.png` - Exponential relationship visualization
- `quality_relationships.png` - Inter-dependencies of quality metrics
- `pair_plot.png` - Multi-variable scatter matrix
- `grouped_analysis.png` - Statistics by magnitude and depth categories
- `earthquake_locations.png` - Global map of earthquake locations
- `earthquake_density.png` - 2D histogram heatmap
- `regional_patterns.png` - Analysis by geographic regions
- `missing_values.png` - Missing data distribution
- `outliers_boxplot.png` - Outlier detection visualization

**All visualizations saved to:** `/outputs/figures/`

---

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/brixtonpham/earthquake-eda.git
cd earthquake-eda
```

2. **Create a Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Analysis

#### Option 1: Python Script (Recommended)
```bash
python src/eda_earthquake.py
```

**Output:**
- Console output with progress and statistics
- Summary report saved to: `outputs/eda_summary.txt`
- All visualizations saved to: `outputs/figures/`

**Execution Time:** ~5-10 minutes (depending on system performance)

#### Option 2: Jupyter Notebook
```bash
jupyter notebook notebooks/EDA_Earthquake_Analysis.ipynb
```

Run cells sequentially to execute analysis step-by-step with interactive outputs.

---

## Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=1.3.0 | Data manipulation and analysis |
| `numpy` | >=1.20.0 | Numerical computations |
| `matplotlib` | >=3.4.0 | 2D plotting and visualizations |
| `seaborn` | >=0.11.0 | Statistical data visualization |
| `scipy` | >=1.7.0 | Statistical analysis (correlation, distribution) |
| `scikit-learn` | >=0.24.0 | PCA and preprocessing |

### Full Requirements File

See `requirements.txt` for complete dependency list with pinned versions.

### Python Version
- **Minimum:** Python 3.8
- **Recommended:** Python 3.9+

---

## Project Structure

```
earthquake-eda/
├── data/
│   └── processed/
│       └── final_earthquake_data_2002_2025.csv  (737MB, gitignored)
│
├── src/
│   └── eda_earthquake.py                         (Main EDA script)
│
├── notebooks/
│   └── EDA_Earthquake_Analysis.ipynb             (Interactive notebook)
│
├── outputs/
│   ├── figures/                                  (18 PNG visualizations)
│   └── eda_summary.txt                           (Summary statistics report)
│
├── docs/
│   ├── EDA_Implementation_Plan_Earthquake_Data.md
│   └── Agent_EDA_Implementation_Guide.md
│
├── .gitignore                                    (Excludes large data files)
├── requirements.txt                              (Python dependencies)
└── README.md                                     (This file)
```

---

## Code Structure

The main EDA script (`src/eda_earthquake.py`) is organized into 8 logical sections:

1. **Imports & Configuration** - Library setup and global plot settings
2. **Data Loading Functions** - CSV parsing and initial assessment
3. **Data Quality Functions** - Missing values, duplicates, validation
4. **Univariate Analysis** - Distribution analysis for individual variables
5. **Bivariate Analysis** - Relationship analysis between variable pairs
6. **Multivariate Analysis** - Correlation matrices, PCA, grouped statistics
7. **Geospatial Analysis** - Location mapping and regional patterns
8. **Main Execution** - Orchestrates full EDA pipeline

Each section contains well-documented functions with clear docstrings and type hints.

---

## Key Insights Summary

### Data Quality
✓ Comprehensive USGS earthquake catalog with excellent spatial and temporal coverage
✓ Minimal data quality issues (<7% missing values)
✓ Consistent data validation across all records

### Seismic Patterns
✓ Earthquake magnitudes follow expected power-law distribution
✓ Shallow earthquakes dominate (~75% < 70km depth)
✓ Weak but consistent depth-magnitude correlation
✓ Clear seasonal and temporal patterns in occurrence rates

### Regional Distribution
✓ Pacific Ring of Fire shows highest concentration
✓ Oceanic regions have higher activity than continental
✓ Stronger earthquakes concentrated in specific tectonic zones

### Methodological Insights
✓ PCA shows earthquake characteristics driven by multiple independent factors
✓ Quality metrics are generally well-distributed
✓ Energy follows expected exponential relationship with magnitude

---

## Important Notes

### Data File Not Included
The earthquake dataset (`final_earthquake_data_2002_2025.csv`) is **not included** in this repository due to its large size (737 MB).

**To obtain the data:**
1. Download from USGS Earthquake Hazards Program: https://earthquake.usgs.gov/earthquakes/search/
2. Filter for time period 2002-01-01 to 2025-12-17
3. Export as CSV format
4. Save to: `./data/processed/final_earthquake_data_2002_2025.csv`

Alternatively, download the processed file from the project's data repository or Google Drive link (if available).

### Performance Considerations
- Full dataset analysis requires ~2-3 GB RAM
- Visualizations are generated with sampling for large datasets (50k+ points)
- PCA computation may take 1-2 minutes on older systems
- All operations are parallelizable for future optimization

---

## Contributing

Contributions are welcome! Areas for enhancement:
- Advanced statistical modeling (time series, clustering)
- Interactive visualizations (Plotly, Dash)
- Predictive analysis for earthquake forecasting
- Machine learning classification models
- Extended geospatial analysis with mapping libraries

Please open an issue or submit a pull request.

---

## References

### Data Source
- USGS Earthquake Hazards Program: https://earthquake.usgs.gov/earthquakes/

### Key EDA Papers
- Exploratory Data Analysis (Tukey, 1977)
- A Systematic Approach to EDA (Behrens, 1997)

### Seismic Science Resources
- Richter Scale: https://en.wikipedia.org/wiki/Richter_scale
- Earthquake Energy: https://earthquake.usgs.gov/earthquakes/events/
- Plate Tectonics: https://www.usgs.gov/faqs/what-plate-tectonics

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Authors & Acknowledgments

**Created:** December 2024
**Data Source:** United States Geological Survey (USGS)
**Analysis Framework:** Python Scientific Stack (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)

### Acknowledgments
- USGS for maintaining and providing public earthquake data
- Open source community for excellent data science libraries
- Seismic research community for domain knowledge

---

## Contact & Support

For questions, issues, or suggestions:
- **Repository:** https://github.com/brixtonpham/earthquake-eda
- **Issue Tracker:** https://github.com/brixtonpham/earthquake-eda/issues
- **Email:** [Contact information]

---

<div align="center">

**Last Updated:** December 17, 2025

Made with ⚡ for seismic data enthusiasts

</div>

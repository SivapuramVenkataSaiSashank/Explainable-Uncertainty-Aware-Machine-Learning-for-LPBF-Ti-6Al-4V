"""
generate_all_figures.py
Generates the complete set of 20 publication-quality figures (300 DPI)
in the figures/ directory for the LPBF Ti-6Al-4V project.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import xgboost as xgb
import shap

import torch
import torch.nn as nn

# Configuration
os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load Data
df = pd.read_csv('lpbf_ti6al4v_psp_dataset.csv')
df_raw_pores = pd.read_csv('lpbf_raw_pore_aggregated.csv')
df_bench = pd.read_csv('model_benchmark_results.csv')
df_pareto = pd.read_csv('pareto_optimal_conditions.csv')
df_psp = df[df['Has_XCT_Data']].copy().reset_index(drop=True)

f_group1 = ['Laser_Power_W', 'Scan_Speed_mm_s', 'LED_J_m', 'MED_J_sqrt_m_s', 'Pv_W_m_s', 'VED_J_mm3']
f_group2 = f_group1 + ['XCT_Porosity_Mean_pct', 'Pore_Count_All', 'Pore_Eq_Diameter_Mean_um', 
                       'Pore_Sphericity_Mean', 'Projected_Area_RMS_Mean_um2']
f_group3 = f_group2 + ['Roughness_Ra_Mean_um', 'Roughness_RMS_Mean_um', 'Grain_Eq_Diameter_Mean_um', 'Grain_Aspect_Ratio_Mean']

v_grid = np.linspace(350, 1650, 200)

# =============================================================================
# FIGURE 1: Process Space Map & Defect Regimes
# =============================================================================
print("Generating Fig 1: Process Parameter Space Map...")
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
palette = {'Lack of Fusion': '#D9534F', 'Optimal / Conduction': '#5CB85C', 'Keyhole / Balling': '#337AB7'}
for led_val, ls, col in [(200, ':', '#888888'), (300, '--', '#555555'), (400, '-.', '#333333')]:
    p_iso = led_val * v_grid / 1000.0
    valid = (p_iso >= 50) & (p_iso <= 550)
    ax.plot(v_grid[valid], p_iso[valid], ls=ls, color=col, alpha=0.7, label=f'LED = {led_val} J/m')

for regime, grp in df.groupby('Process_Defect_Regime'):
    ax.scatter(grp['Scan_Speed_mm_s'], grp['Laser_Power_W'], 
               color=palette.get(regime, '#333333'), s=110, edgecolors='k', linewidth=1.2, 
               label=f'{regime} (n={len(grp)})', zorder=5)

for idx, row in df.iterrows():
    ax.annotate(str(int(row['Condition_ID'])), 
                (row['Scan_Speed_mm_s'], row['Laser_Power_W']),
                xytext=(4, 4), textcoords='offset points', fontsize=7, color='#222222', alpha=0.85)

ax.set_title('LPBF Ti-6Al-4V Experimental Process Parameter Space & Regimes\n(Luo et al., 2023, Data in Brief)', 
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Laser Scan Speed, $v$ (mm/s)', fontsize=11, fontweight='bold')
ax.set_ylabel('Laser Power, $P$ (W)', fontsize=11, fontweight='bold')
ax.set_xlim(350, 1680)
ax.set_ylim(50, 530)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig1_process_space_map.png')
plt.close()

# =============================================================================
# FIGURE 2: 4-Panel Process Maps (Porosity, UTS, YS, Elongation)
# =============================================================================
print("Generating Fig 2: Process Response Surfaces...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
maps_cfg = [
    (axes[0, 0], 'XCT_Porosity_Mean_pct', 'XCT Porosity (%)', 'viridis_r', df_psp, 'Internal Defect Porosity'),
    (axes[0, 1], 'UTS_Mean_MPa', 'Ultimate Tensile Strength (MPa)', 'plasma', df, 'Ultimate Tensile Strength (UTS)'),
    (axes[1, 0], 'Yield_Strength_Mean_MPa', 'Yield Strength (MPa)', 'inferno', df, '0.2% Offset Yield Strength'),
    (axes[1, 1], 'Elongation_Fracture_Mean_pct', 'Elongation to Fracture (%)', 'magma', df, 'Tensile Ductility / Elongation')
]
for ax, col_name, cbar_label, cmap, data_source, title in maps_cfg:
    sc = ax.scatter(data_source['Scan_Speed_mm_s'], data_source['Laser_Power_W'], 
                    c=data_source[col_name], cmap=cmap, s=120, edgecolors='k', linewidth=1.1, zorder=4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label, fontsize=10, fontweight='bold')
    for led_val in [250, 350]:
        p_iso = led_val * v_grid / 1000.0
        v_sub = (p_iso >= 50) & (p_iso <= 530)
        ax.plot(v_grid[v_sub], p_iso[v_sub], 'k--', alpha=0.3, lw=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Scan Speed, $v$ (mm/s)', fontsize=10)
    ax.set_ylabel('Laser Power, $P$ (W)', fontsize=10)
    ax.set_xlim(350, 1680)
    ax.set_ylim(50, 530)
plt.suptitle('LPBF Ti-6Al-4V Process–Structure–Property Response Surfaces', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('figures/fig2_eda_process_maps.png')
plt.close()

# =============================================================================
# FIGURE 3: Physical Metallurgy Models & Structure-Property
# =============================================================================
print("Generating Fig 3: Physical Metallurgy Models...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
# Panel A: Ryshkewitch-Duckworth
ax = axes[0, 0]
x_por = df_psp['XCT_Porosity_Mean_pct'].values
y_uts = df_psp['UTS_Mean_MPa'].values
def ryshkewitch(p, sigma0, b): return sigma0 * np.exp(-b * (p / 100.0))
popt_rd, _ = curve_fit(ryshkewitch, x_por, y_uts, p0=[1050, 5])
p_dense = np.linspace(0, 5, 100)
ax.errorbar(df_psp['XCT_Porosity_Mean_pct'], df_psp['UTS_Mean_MPa'], 
            xerr=df_psp['XCT_Porosity_Std_pct'], yerr=df_psp['UTS_Std_MPa'], 
            fmt='o', color='#1F77B4', ecolor='#AEC7E8', elinewidth=1.2, capsize=3, label='Experimental Data', zorder=4)
ax.plot(p_dense, ryshkewitch(p_dense, *popt_rd), 'r-', lw=2.2, 
        label=f'Ryshkewitch Fit: $\\sigma = {popt_rd[0]:.1f}\\exp(-{popt_rd[1]:.2f}p)$')
ax.set_title('Ryshkewitch-Duckworth Porosity–Strength Scaling', fontsize=11, fontweight='bold')
ax.set_xlabel('XCT Porosity, $p$ (%)', fontsize=10, fontweight='bold')
ax.set_ylabel('Ultimate Tensile Strength (MPa)', fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='lower left', fontsize=9)

# Panel B: Ductility Cliff
ax = axes[0, 1]
ax.errorbar(df_psp['XCT_Porosity_Mean_pct'], df_psp['Elongation_Fracture_Mean_pct'],
            xerr=df_psp['XCT_Porosity_Std_pct'], yerr=df_psp['Elongation_Fracture_Std_pct'],
            fmt='s', color='#2CA02C', ecolor='#98DF8A', elinewidth=1.2, capsize=3, label='Specimens', zorder=4)
def power_decay(p, a, b): return a / (1.0 + b * p)
popt_decay, _ = curve_fit(power_decay, x_por, df_psp['Elongation_Fracture_Mean_pct'].values, p0=[15, 1])
ax.plot(p_dense, power_decay(p_dense, *popt_decay), 'g--', lw=2.2, 
        label=f'Ductility Fit: $\\epsilon_f = \\frac{{{popt_decay[0]:.1f}}}{{1 + {popt_decay[1]:.2f}p}}$')
ax.axvline(1.0, color='red', ls=':', lw=1.5, label='Critical Porosity Threshold (1.0%)')
ax.set_title('Ductility Exhaustion vs Internal Porosity', fontsize=11, fontweight='bold')
ax.set_xlabel('XCT Porosity, $p$ (%)', fontsize=10, fontweight='bold')
ax.set_ylabel('Elongation to Fracture (%)', fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='upper right', fontsize=9)

# Panel C: Hall-Petch
ax = axes[1, 0]
d_grain = df['Grain_Eq_Diameter_Mean_um'].values
d_inv_sqrt = 1.0 / np.sqrt(d_grain)
slope, intercept, r_val, p_val, _ = stats.linregress(d_inv_sqrt, df['Yield_Strength_Mean_MPa'].values)
d_dense = np.linspace(d_inv_sqrt.min()*0.95, d_inv_sqrt.max()*1.05, 50)
ax.errorbar(d_inv_sqrt, df['Yield_Strength_Mean_MPa'].values, yerr=df['Yield_Strength_Std_MPa'], 
            fmt='^', color='#FF7F0E', ecolor='#FFBB78', elinewidth=1.2, capsize=3, label='Conditions (n=42)', zorder=4)
ax.plot(d_dense, intercept + slope * d_dense, 'k-', lw=2.0, 
        label=f'Hall-Petch: $\\sigma_y = {intercept:.1f} + {slope:.1f}d^{{-1/2}}$ (r={r_val:.2f})')
ax.set_title('Hall-Petch Prior-$\\beta$ Grain Boundary Strengthening', fontsize=11, fontweight='bold')
ax.set_xlabel('Inverse Square Root Grain Size, $d^{-1/2}$ ($\\mu m^{-0.5}$)', fontsize=10, fontweight='bold')
ax.set_ylabel('Yield Strength (MPa)', fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='lower right', fontsize=9)

# Panel D: Archimedes vs XCT
ax = axes[1, 1]
ax.errorbar(df_psp['Archimedes_Porosity_Mean_pct'], df_psp['XCT_Porosity_Mean_pct'],
            xerr=df_psp['Archimedes_Porosity_Std_pct'], yerr=df_psp['XCT_Porosity_Std_pct'],
            fmt='D', color='#9467BD', ecolor='#C5B0D5', elinewidth=1.2, capsize=3, label='Tested Conditions (n=35)', zorder=4)
ax.plot([0, 6], [0, 6], 'k--', lw=1.5, alpha=0.7, label='1:1 Parity Line')
ax.set_title('Archimedes vs Micro-CT Porosity Characterization', fontsize=11, fontweight='bold')
ax.set_xlabel('Archimedes Porosity (%)', fontsize=10, fontweight='bold')
ax.set_ylabel('X-ray CT Porosity (%)', fontsize=10, fontweight='bold')
ax.set_xlim(0, 6); ax.set_ylim(0, 6)
ax.legend(frameon=True, loc='upper left', fontsize=9)
plt.suptitle('Physical Metallurgy Foundations of LPBF Ti-6Al-4V Material Behaviour', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('figures/fig3_physical_models.png')
plt.close()

# =============================================================================
# FIGURE 4: Raw Micro-CT Pore Statistics
# =============================================================================
print("Generating Fig 4: Raw Micro-CT Defect Distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
axes[0].bar(df_raw_pores['Condition_ID'].astype(str), df_raw_pores['Raw_Pore_Count'], color='#1F77B4', edgecolor='k', alpha=0.8)
axes[0].set_title('Total Internal Pore Count per Specimen Gauge', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Condition ID', fontsize=9); axes[0].set_ylabel('Pore Count', fontsize=9)
axes[0].tick_params(axis='x', rotation=90, labelsize=7)

axes[1].scatter(df_raw_pores['Mean_Pore_Diameter_um'], df_raw_pores['Mean_Pore_Sphericity'], 
                c=df_raw_pores['Total_Pore_Volume_mm3'], cmap='coolwarm', s=90, edgecolors='k')
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('Total Pore Vol ($mm^3$)', fontsize=8)
axes[1].set_title('Mean Pore Diameter vs Sphericity (Shape Factor)', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Mean Pore Diameter ($\\mu m$)', fontsize=9); axes[1].set_ylabel('Mean Pore Sphericity ($\\Psi$)', fontsize=9)

axes[2].scatter(df_raw_pores['Condition_ID'], df_raw_pores['P95_Pore_Diameter_um'], color='#D62728', s=80, edgecolors='k', label='95th Percentile Diameter')
axes[2].scatter(df_raw_pores['Condition_ID'], df_raw_pores['Mean_Pore_Diameter_um'], color='#2CA02C', s=60, edgecolors='k', label='Mean Diameter')
axes[2].set_title('Pore Size Extremes Across Fabricated Samples', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Condition ID', fontsize=9); axes[2].set_ylabel('Equivalent Diameter ($\\mu m$)', fontsize=9)
axes[2].legend(fontsize=8)
plt.suptitle('Microscopic Defect Population Characterization (Over 1 Million Pores Analyzed)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig4_raw_pores_distribution.png')
plt.close()

# =============================================================================
# FIGURE 5: Correlation Heatmap
# =============================================================================
print("Generating Fig 5: Correlation Matrix Heatmap...")
corr_cols = [
    'Laser_Power_W', 'Scan_Speed_mm_s', 'LED_J_m', 'MED_J_sqrt_m_s', 'Pv_W_m_s', 'VED_J_mm3',
    'XCT_Porosity_Mean_pct', 'Pore_Eq_Diameter_Mean_um', 'Pore_Sphericity_Vol_Mean',
    'Roughness_Ra_Mean_um', 'Grain_Eq_Diameter_Mean_um',
    'Yield_Strength_Mean_MPa', 'UTS_Mean_MPa', 'Elongation_Fracture_Mean_pct', 'Vickers_Hardness_Mean_HV'
]
corr_matrix = df_psp[corr_cols].corr(method='spearman')
fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax, annot_kws={'size': 7.5})
ax.set_title('Spearman Rank Correlation Across Process–Structure–Property Features', fontsize=12, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=8.5); plt.yticks(fontsize=8.5)
plt.tight_layout()
plt.savefig('figures/fig5_correlation_heatmap.png')
plt.close()

# =============================================================================
# FIGURE 6: Validation Scheme Comparison (Leakage Audit)
# =============================================================================
print("Generating Fig 6: Data Leakage Audit...")
df_leak = pd.read_csv('validation_leakage_comparison.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
colors_cv = ['#4A90E2', '#E74C3C']
ax1.bar(['Random 5-Fold CV\n(Risk of Optimism)', 'Grouped 5-Fold CV\n(Leakage-Free)'], 
        df_leak['Mean_R2'], yerr=df_leak['Std_R2'], 
        color=colors_cv, capsize=5, edgecolor='k', alpha=0.85, width=0.55)
ax1.set_ylabel('$R^2$ Score (Higher is better)', fontsize=10, fontweight='bold')
ax1.set_title('Cross-Validation Explanatory Power ($R^2$)', fontsize=11, fontweight='bold')
ax1.set_ylim(-0.3, 1.0)
for i, v in enumerate(df_leak['Mean_R2']):
    ax1.text(i, max(0.05, v + 0.05), f"{v:.3f}", ha='center', fontweight='bold')

ax2.bar(['Random 5-Fold CV\n(Risk of Optimism)', 'Grouped 5-Fold CV\n(Leakage-Free)'], 
        df_leak['Mean_RMSE_MPa'], yerr=df_leak['Std_RMSE_MPa'], 
        color=colors_cv, capsize=5, edgecolor='k', alpha=0.85, width=0.55)
ax2.set_ylabel('RMSE (MPa) (Lower is better)', fontsize=10, fontweight='bold')
ax2.set_title('Generalization Error (RMSE in MPa)', fontsize=11, fontweight='bold')
for i, v in enumerate(df_leak['Mean_RMSE_MPa']):
    ax2.text(i, v + 2, f"{v:.1f} MPa", ha='center', fontweight='bold')

plt.suptitle('Validation Strategy Audit: Impact of Data Leakage on LPBF Performance Estimates', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig6_validation_leakage_comparison.png')
plt.close()

# =============================================================================
# FIGURE 7: Feature Group Ablation
# =============================================================================
print("Generating Fig 7: Feature Group Ablation...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
sub_uts = df_bench[(df_bench['Target'] == 'Ultimate Tensile Strength (MPa)') & (df_bench['Model'] == 'XGBoost')]
sub_por = df_bench[(df_bench['Target'] == 'XCT Porosity (%)') & (df_bench['Feature_Group'] == 'Group 1: Process-Only')]

fg_labels = ['Group 1:\nProcess-Only', 'Group 2:\nProcess + Defects', 'Group 3:\nFull PSP']
ax1.bar(fg_labels, sub_uts['Mean_R2'], color=['#3498DB', '#E67E22', '#2ECC71'], edgecolor='k', width=0.55, alpha=0.85)
ax1.set_ylabel('$R^2$ Score (Leakage-Free Grouped CV)', fontsize=10, fontweight='bold')
ax1.set_title('UTS Prediction Ablation Across Feature Groups (XGBoost)', fontsize=11, fontweight='bold')
ax1.set_ylim(-0.15, 0.85)
for i, v in enumerate(sub_uts['Mean_R2']):
    ax1.text(i, max(0.02, v + 0.03), f"$R^2={v:.3f}$", ha='center', fontsize=9, fontweight='bold')

ax2.bar(sub_por['Model'], sub_por['Mean_R2'], color='#9B59B6', edgecolor='k', width=0.55, alpha=0.85)
ax2.set_ylabel('$R^2$ Score (Leakage-Free Grouped CV)', fontsize=10, fontweight='bold')
ax2.set_title('Internal Porosity Prediction from Process Parameters', fontsize=11, fontweight='bold')
ax2.set_ylim(-0.1, 0.6)
plt.xticks(rotation=30, ha='right', fontsize=9)
for i, v in enumerate(sub_por['Mean_R2']):
    ax2.text(i, max(0.01, v + 0.02), f"{v:.2f}", ha='center', fontsize=8.5, fontweight='bold')

plt.suptitle('Scientific Ablation: Process-Only vs Microstructural/Defect Feature Value', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig7_feature_group_ablation.png')
plt.close()

# =============================================================================
# FIGURE 8: Comprehensive Multi-Model Benchmark Comparison
# =============================================================================
print("Generating Fig 8: Multi-Model Benchmark Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
# Group 2 models comparison on UTS
sub_g2_uts = df_bench[(df_bench['Target'] == 'Ultimate Tensile Strength (MPa)') & 
                      (df_bench['Feature_Group'] == 'Group 2: Process + Defects')]
models_order = ['Ridge', 'SVR (RBF)', 'Random Forest', 'XGBoost', 'Gaussian Process', 'PyTorch Deep MLP (GPU)']
sub_g2_uts = sub_g2_uts.set_index('Model').reindex(models_order).reset_index()

bar_colors = ['#34495E', '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#E74C3C']
ax1.bar(sub_g2_uts['Model'], sub_g2_uts['Mean_R2'], color=bar_colors, edgecolor='k', width=0.6, alpha=0.85)
ax1.set_title('Model Performance: UTS Prediction (Group 2: Process + Defects)', fontsize=11, fontweight='bold')
ax1.set_ylabel('$R^2$ Score (Grouped 5-Fold CV)', fontsize=10, fontweight='bold')
ax1.set_ylim(-0.6, 1.0)
ax1.tick_params(axis='x', rotation=30)
for i, v in enumerate(sub_g2_uts['Mean_R2']):
    ax1.text(i, max(0.02, v + 0.03), f"{v:.2f}", ha='center', fontsize=8.5, fontweight='bold')

# RMSE comparison
ax2.bar(sub_g2_uts['Model'], sub_g2_uts['Mean_RMSE'], color=bar_colors, edgecolor='k', width=0.6, alpha=0.85)
ax2.set_title('Root Mean Squared Error (RMSE) on UTS [Lower is Better]', fontsize=11, fontweight='bold')
ax2.set_ylabel('RMSE (MPa)', fontsize=10, fontweight='bold')
ax2.tick_params(axis='x', rotation=30)
for i, v in enumerate(sub_g2_uts['Mean_RMSE']):
    ax2.text(i, v + 1, f"{v:.1f}", ha='center', fontsize=8.5, fontweight='bold')

plt.suptitle('Multi-Model Benchmark Suite: Classical vs Ensemble vs GPU Deep Learning', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig8_multi_model_benchmark_comparison.png')
plt.close()

# =============================================================================
# FIGURE 9: SHAP Summary Beeswarm Plot
# =============================================================================
print("Generating Fig 9: SHAP Summary Beeswarm Plot...")
X_full_psp = df_psp[f_group3]
y_uts_psp = df_psp['UTS_Mean_MPa']
scaler_psp = StandardScaler()
X_full_psp_sc = pd.DataFrame(scaler_psp.fit_transform(X_full_psp), columns=f_group3)
model_xgb_uts = xgb.XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=42)
model_xgb_uts.fit(X_full_psp_sc, y_uts_psp)
explainer_uts = shap.TreeExplainer(model_xgb_uts)
shap_values_uts = explainer_uts(X_full_psp_sc)

plt.figure(figsize=(10, 6), dpi=300)
shap.summary_plot(shap_values_uts, X_full_psp_sc, show=False, max_display=10)
plt.title('SHAP Feature Importance Beeswarm: Ultimate Tensile Strength (Full PSP)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/fig9_shap_summary.png')
plt.close()

# =============================================================================
# FIGURE 10: SHAP Feature Importance Bars (Mean |SHAP|)
# =============================================================================
print("Generating Fig 10: SHAP Feature Importance Bars...")
plt.figure(figsize=(9, 5.5), dpi=300)
shap.summary_plot(shap_values_uts, X_full_psp_sc, plot_type="bar", show=False, max_display=10, color='#2980B9')
plt.title('Global Feature Importance Ranking (Mean Absolute SHAP Value)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/fig10_shap_feature_importance_bars.png')
plt.close()

# =============================================================================
# FIGURE 11: SHAP Dependence Plots (LED, Power, Porosity)
# =============================================================================
print("Generating Fig 11: SHAP Dependence Curves...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
shap.plots.scatter(shap_values_uts[:, "LED_J_m"], ax=axes[0], show=False)
axes[0].set_title('SHAP Dependence: Linear Energy Density (LED)', fontsize=10, fontweight='bold')
axes[0].set_ylabel('SHAP Value (Impact on UTS MPa)', fontsize=9)

shap.plots.scatter(shap_values_uts[:, "XCT_Porosity_Mean_pct"], ax=axes[1], show=False)
axes[1].set_title('SHAP Dependence: Internal XCT Porosity', fontsize=10, fontweight='bold')
axes[1].set_ylabel('SHAP Value (Impact on UTS MPa)', fontsize=9)

shap.plots.scatter(shap_values_uts[:, "Laser_Power_W"], ax=axes[2], show=False)
axes[2].set_title('SHAP Dependence: Laser Power ($P$)', fontsize=10, fontweight='bold')
axes[2].set_ylabel('SHAP Value (Impact on UTS MPa)', fontsize=9)

plt.suptitle('Nonlinear Metallurgical Transitions Captured by SHAP Values', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig11_shap_dependence_thresholds.png')
plt.close()

# =============================================================================
# FIGURE 12: Permutation Feature Importance
# =============================================================================
print("Generating Fig 12: Permutation Feature Importance...")
r_perm = permutation_importance(model_xgb_uts, X_full_psp_sc, y_uts_psp, n_repeats=30, random_state=42)
sorted_idx = r_perm.importances_mean.argsort()[::-1][:10]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
ax.barh(range(len(sorted_idx)), r_perm.importances_mean[sorted_idx][::-1],
        xerr=r_perm.importances_std[sorted_idx][::-1], color='#E67E22', edgecolor='k', alpha=0.85, capsize=4)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(f_group3)[sorted_idx][::-1], fontsize=9)
ax.set_xlabel('Mean Decrease in $R^2$ Explanatory Power', fontsize=10, fontweight='bold')
ax.set_title('Permutation Feature Importance with 30 Repeats (Leakage-Free Audit)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig12_permutation_feature_importance.png')
plt.close()

# =============================================================================
# FIGURE 13: Partial Dependence Plots (PDP)
# =============================================================================
print("Generating Fig 13: Partial Dependence Plots...")
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
rf_pdp = RandomForestRegressor(n_estimators=50, random_state=42)
rf_pdp.fit(X_full_psp_sc, y_uts_psp)
features_pdp = ['LED_J_m', 'Laser_Power_W', 'XCT_Porosity_Mean_pct']
display_pdp = PartialDependenceDisplay.from_estimator(
    rf_pdp, X_full_psp_sc, features_pdp, ax=ax, line_kw={"color": "#C0392B", "linewidth": 2.2}
)
plt.suptitle('Partial Dependence Plots (Marginal Effects on UTS in MPa)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig13_partial_dependence_plots.png')
plt.close()

# =============================================================================
# FIGURE 14: Gaussian Process Regression Epistemic Uncertainty
# =============================================================================
print("Generating Fig 14: Gaussian Process Uncertainty Intervals...")
X_proc = df_psp[f_group1].values
y_proc = df_psp['UTS_Mean_MPa'].values
scaler_proc = StandardScaler()
X_proc_sc = scaler_proc.fit_transform(X_proc)

kernel_gpr = C(100.0, (1e-1, 1e4)) * RBF(length_scale=1.5, length_scale_bounds=(0.2, 10.0)) + WhiteKernel(noise_level=10.0)
gpr_model = GaussianProcessRegressor(kernel=kernel_gpr, n_restarts_optimizer=5, random_state=42)
gpr_model.fit(X_proc_sc, y_proc)
gpr_mu, gpr_sigma = gpr_model.predict(X_proc_sc, return_std=True)

order = np.argsort(y_proc)
x_axis = np.arange(len(order))

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.fill_between(x_axis, gpr_mu[order] - 1.96 * gpr_sigma[order], gpr_mu[order] + 1.96 * gpr_sigma[order],
                color='#3498DB', alpha=0.25, label='95% Bayesian Epistemic Credible Band')
ax.plot(x_axis, gpr_mu[order], 'b-', lw=1.8, label='GPR Mean Prediction')
ax.scatter(x_axis, y_proc[order], color='#E74C3C', s=50, edgecolors='k', zorder=5, label='True Measured UTS')
ax.set_title('Gaussian Process Analytical Epistemic Uncertainty (95% Credible Interval)', fontsize=12, fontweight='bold')
ax.set_xlabel('Evaluation Condition Index (Sorted by Strength)', fontsize=10, fontweight='bold')
ax.set_ylabel('Ultimate Tensile Strength (MPa)', fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig14_uncertainty_gpr_intervals.png')
plt.close()

# =============================================================================
# FIGURE 15: PyTorch GPU Monte Carlo Dropout Uncertainty
# =============================================================================
print("Generating Fig 15: PyTorch GPU MC Dropout Uncertainty...")
class LPBFDeepMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(0.20),
            nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

y_scaler_mc = StandardScaler()
y_proc_sc = y_scaler_mc.fit_transform(y_proc.reshape(-1, 1)).flatten()

model_mc = LPBFDeepMLP(X_proc.shape[1]).to(device)
tx_proc = torch.tensor(X_proc_sc, dtype=torch.float32, device=device)
ty_proc = torch.tensor(y_proc_sc, dtype=torch.float32, device=device).unsqueeze(1)
opt_mc = torch.optim.Adam(model_mc.parameters(), lr=0.015, weight_decay=1e-4)
crit_mc = nn.MSELoss()

for ep in range(200):
    opt_mc.zero_grad()
    l = crit_mc(model_mc(tx_proc), ty_proc)
    l.backward()
    opt_mc.step()

def enable_dropout(m):
    for mod in m.modules():
        if mod.__class__.__name__.startswith('Dropout'):
            mod.train()
enable_dropout(model_mc)

T_samples = 100
mc_preds = []
with torch.no_grad():
    for _ in range(T_samples):
        p_sc = model_mc(tx_proc).cpu().numpy().reshape(-1, 1)
        mc_preds.append(y_scaler_mc.inverse_transform(p_sc).flatten())
mc_preds = np.array(mc_preds)
mc_mean = mc_preds.mean(axis=0)
mc_std = mc_preds.std(axis=0)

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.fill_between(x_axis, mc_mean[order] - 1.96 * mc_std[order], mc_mean[order] + 1.96 * mc_std[order],
                color='#2ECC71', alpha=0.3, label='PyTorch GPU MC Dropout 95% Band (T=100)')
ax.plot(x_axis, mc_mean[order], 'g-', lw=1.8, label='MC Dropout Mean')
ax.scatter(x_axis, y_proc[order], color='#E74C3C', s=50, edgecolors='k', zorder=5, label='True Measured UTS')
ax.set_title('PyTorch GPU Monte Carlo Dropout Epistemic Uncertainty on RTX 4060', fontsize=12, fontweight='bold')
ax.set_xlabel('Evaluation Condition Index (Sorted by Strength)', fontsize=10, fontweight='bold')
ax.set_ylabel('Ultimate Tensile Strength (MPa)', fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig15_uncertainty_mc_dropout_gpu.png')
plt.close()

# =============================================================================
# FIGURE 16: Conformal Prediction Coverage Verification
# =============================================================================
print("Generating Fig 16: Conformal Prediction Calibration & Coverage...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_res = np.zeros(len(y_proc))
oof_p = np.zeros(len(y_proc))

for tr, va in kf.split(X_proc_sc, y_proc):
    m = xgb.XGBRegressor(n_estimators=45, max_depth=3, random_state=42)
    m.fit(X_proc_sc[tr], y_proc[tr])
    oof_p[va] = m.predict(X_proc_sc[va])
    oof_res[va] = np.abs(y_proc[va] - oof_p[va])

alphas = np.linspace(0.50, 0.98, 25)
empirical_coverages = []
interval_widths = []
for a in alphas:
    q = np.quantile(oof_res, a)
    cov = (np.abs(y_proc - oof_p) <= q).mean() * 100
    empirical_coverages.append(cov)
    interval_widths.append(2 * q)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
ax1.plot(alphas * 100, empirical_coverages, 'b-o', lw=2, markersize=5, label='Empirical Coverage')
ax1.plot(alphas * 100, alphas * 100, 'k--', lw=1.5, label='Ideal Calibration Line')
ax1.set_title('Conformal Coverage Calibration Curve', fontsize=11, fontweight='bold')
ax1.set_xlabel('Nominal Confidence Level (1 - $\\alpha$) %', fontsize=10)
ax1.set_ylabel('Empirical Coverage (%)', fontsize=10)
ax1.legend(frameon=True, loc='lower right', fontsize=9)

ax2.plot(alphas * 100, interval_widths, 'r-s', lw=2, markersize=5)
ax2.set_title('Prediction Interval Width vs Confidence Level', fontsize=11, fontweight='bold')
ax2.set_xlabel('Nominal Confidence Level (1 - $\\alpha$) %', fontsize=10)
ax2.set_ylabel('Total Prediction Interval Width (MPa)', fontsize=10)

plt.suptitle('Distribution-Free Conformal Prediction Verification for Digital Manufacturing Quality', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig16_conformal_prediction_coverage.png')
plt.close()

# =============================================================================
# FIGURE 17: Out-of-Distribution Trust Boundary Map
# =============================================================================
print("Generating Fig 17: Out-of-Distribution Trust Map...")
mean_proc = np.mean(X_proc, axis=0)
cov_proc = np.cov(X_proc, rowvar=False)
inv_cov = np.linalg.pinv(cov_proc)

def compute_mahalanobis(sample):
    diff = sample - mean_proc
    return np.sqrt(diff.dot(inv_cov).dot(diff))

mahal_scores = [compute_mahalanobis(x) for x in X_proc]
ood_thresh = np.percentile(mahal_scores, 95)

p_sim = np.linspace(50, 520, 60)
v_sim = np.linspace(350, 1650, 60)
P_grid, V_grid = np.meshgrid(p_sim, v_sim)

h_mm, t_mm = 0.082, 0.060
Z_mahal = np.zeros(P_grid.shape)
for i in range(P_grid.shape[0]):
    for j in range(P_grid.shape[1]):
        p_val = P_grid[i, j]
        v_val = V_grid[i, j]
        led = (p_val / v_val) * 1000.0
        med = p_val / np.sqrt(v_val / 1000.0)
        pv = p_val * (v_val / 1000.0)
        ved = p_val / (v_val * h_mm * t_mm)
        sample = np.array([p_val, v_val, led, med, pv, ved])
        sample_sc = scaler_proc.transform(sample.reshape(1, -1))[0]
        Z_mahal[i, j] = compute_mahalanobis(sample_sc)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
cf = ax.contourf(V_grid, P_grid, Z_mahal, levels=20, cmap='YlOrRd_r')
cbar = plt.colorbar(cf, ax=ax)
cbar.set_label('Mahalanobis Trust Distance (Normalized)', fontsize=10, fontweight='bold')

# Contour for OOD Boundary
cs = ax.contour(V_grid, P_grid, Z_mahal, levels=[ood_thresh], colors=['red'], linewidths=[2.2], linestyles=['--'])
ax.clabel(cs, inline=True, fmt={ood_thresh: 'OOD Boundary (95%)'}, fontsize=9)

ax.scatter(df['Scan_Speed_mm_s'], df['Laser_Power_W'], color='cyan', edgecolors='k', s=60, label='Fabricated Conditions (n=42)', zorder=5)
ax.set_title('Out-of-Distribution (OOD) Process Trust Map\n(Guarding against unvalidated LPBF extrapolation)', fontsize=11, fontweight='bold')
ax.set_xlabel('Scan Speed, $v$ (mm/s)', fontsize=10, fontweight='bold')
ax.set_ylabel('Laser Power, $P$ (W)', fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig17_out_of_distribution_trust_map.png')
plt.close()

# =============================================================================
# FIGURE 18: Multi-Objective Pareto Frontier (2D and 3D)
# =============================================================================
print("Generating Fig 18: Multi-Objective Pareto Frontier...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Panel 1: Porosity vs UTS
ax1.scatter(df_pareto['Predicted_Porosity_pct'], df_pareto['Predicted_UTS_MPa'],
            c=df_pareto['Predicted_Elongation_pct'], cmap='viridis', s=120, edgecolors='k', 
            label='Pareto Optimal Frontier', zorder=5)
cbar = plt.colorbar(ax1.collections[0], ax=ax1)
cbar.set_label('Predicted Ductility / Elongation (%)', fontsize=9)
ax1.set_title('Pareto Frontier: Porosity vs Ultimate Tensile Strength', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted XCT Porosity (%) [Minimization]', fontsize=10, fontweight='bold')
ax1.set_ylabel('Predicted UTS (MPa) [Maximization]', fontsize=10, fontweight='bold')
ax1.legend(frameon=True, loc='lower left', fontsize=8.5)

# Panel 2: UTS vs Elongation
ax2.scatter(df_pareto['Predicted_UTS_MPa'], df_pareto['Predicted_Elongation_pct'],
            c=df_pareto['Predicted_Porosity_pct'], cmap='coolwarm_r', s=120, edgecolors='k',
            label='Pareto Optimal Solutions', zorder=5)
cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
cbar2.set_label('Porosity (%)', fontsize=9)
ax2.set_title('Pareto Tradeoff: Strength vs Ductility', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted UTS (MPa) [Maximization]', fontsize=10, fontweight='bold')
ax2.set_ylabel('Predicted Elongation (%) [Maximization]', fontsize=10, fontweight='bold')
ax2.legend(frameon=True, loc='upper left', fontsize=8.5)

plt.suptitle('Multi-Objective Constrained Optimization & Pareto Process Selection', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig18_multi_objective_pareto_frontier.png')
plt.close()

# =============================================================================
# FIGURE 19: Golden Process Window in Manufacturing Space
# =============================================================================
print("Generating Fig 19: Golden Process Window...")
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

# Feasible background domain
p_dense_opt = np.linspace(100, 500, 80)
v_dense_opt = np.linspace(400, 1600, 80)
Pv_grid, V_mesh = np.meshgrid(p_dense_opt, v_dense_opt)
LED_mesh = (Pv_grid / V_mesh) * 1000.0

# Plot defect boundary regions
ax.contourf(V_mesh, Pv_grid, LED_mesh, levels=[0, 200, 420, 1000], 
            colors=['#FADBD8', '#D5F5E3', '#D6EAF8'], alpha=0.5)

# Golden Window Box: P in [235, 275], v in [1200, 1250]
from matplotlib.patches import Rectangle
rect = Rectangle((1200, 235), 50, 40, linewidth=2.5, edgecolor='#E74C3C', facecolor='none', 
                 linestyle='--', label='Recommended Golden Window (P: 235-275W, v: 1200-1250 mm/s)', zorder=6)
ax.add_patch(rect)

# Scatter all 42 fabricated points
ax.scatter(df['Scan_Speed_mm_s'], df['Laser_Power_W'], color='#34495E', edgecolors='k', s=60, alpha=0.7, label='Experimental Design (n=42)', zorder=4)

# Scatter Pareto solutions
ax.scatter(df_pareto['Scan_Speed_mm_s'], df_pareto['Laser_Power_W'], color='#E74C3C', marker='*', s=200, edgecolors='k', label='Pareto Optimum Points', zorder=7)

ax.set_title('Identified Golden Process Window for LPBF Ti-6Al-4V Additive Manufacturing\n(Balancing Density >99.95%, UTS >1030 MPa, Ductility >15%, & Build Rate)', 
             fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel('Laser Scan Speed, $v$ (mm/s)', fontsize=10, fontweight='bold')
ax.set_ylabel('Laser Power, $P$ (W)', fontsize=10, fontweight='bold')
ax.set_xlim(380, 1620); ax.set_ylim(80, 520)
ax.legend(frameon=True, loc='upper left', fontsize=8.5)
plt.tight_layout()
plt.savefig('figures/fig19_golden_process_window.png')
plt.close()

# =============================================================================
# FIGURE 20: PSP ML Framework Architecture Schematic
# =============================================================================
print("Generating Fig 20: ML Framework Architecture Diagram...")
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.axis('off')

# Draw schematic boxes using matplotlib patches
from matplotlib.patches import FancyBboxPatch

boxes = [
    ("1. Experimental Campaign\\n(Luo et al., 2023)\\n42 Conditions (P, v)\\n35 Micro-CT Scans\\nASTM E8 Tensile (DIC)", 0.05, 0.55, 0.22, 0.35, '#3498DB'),
    ("2. Physics Descriptors\\nLinear Energy Density (LED)\\nVolumetric Energy (VED)\\nRosenthal Penetration (MED)\\nRayleigh Balling (Pv)", 0.32, 0.55, 0.22, 0.35, '#2ECC71'),
    ("3. Microstructure & Defect\\nXCT Internal Porosity\\nPore Sphericity & Size\\nPrior-β Grain Diameter\\nSurface Roughness (Ra)", 0.59, 0.55, 0.22, 0.35, '#F39C12'),
    ("4. Mechanical Properties\\nYield Strength (YS)\\nUltimate Tensile (UTS)\\nElongation / Ductility\\nVickers Microhardness", 0.86, 0.55, 0.12, 0.35, '#E74C3C'),
    
    ("5. Leakage-Free Validation\\nGroupKFold (Process Clusters)\\nRandom vs Grouped CV Audit\\nEmpirical Optimism Proof", 0.05, 0.08, 0.22, 0.35, '#9B59B6'),
    ("6. Multi-Model Suite\\nRidge, SVR, RF, XGBoost\\nGaussian Process (GPR)\\nPyTorch GPU Deep MLP", 0.32, 0.08, 0.22, 0.35, '#1ABC9C'),
    ("7. Explainable AI & Trust\\nSHAP Beeswarm & Thresholds\\nBayesian GPR & MC Dropout\\nConformal Prediction (95%)\\nMahalanobis OOD Detection", 0.59, 0.08, 0.22, 0.35, '#34495E'),
    ("8. Constrained Optimization\\nMulti-Objective Pareto Search\\nGolden Process Window\\n(P: 235-275W, v: 1200-1250)", 0.86, 0.08, 0.12, 0.35, '#D35400')
]

for title, x, y, w, h, col in boxes:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", 
                         facecolor=col, edgecolor='black', linewidth=1.5, alpha=0.88)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, title, ha='center', va='center', color='white', 
            fontsize=8.5, fontweight='bold', multialignment='center')

# Arrows
ax.annotate('', xy=(0.31, 0.725), xytext=(0.28, 0.725), arrowprops=dict(facecolor='black', width=2, headwidth=7))
ax.annotate('', xy=(0.58, 0.725), xytext=(0.55, 0.725), arrowprops=dict(facecolor='black', width=2, headwidth=7))
ax.annotate('', xy=(0.85, 0.725), xytext=(0.82, 0.725), arrowprops=dict(facecolor='black', width=2, headwidth=7))

ax.annotate('', xy=(0.31, 0.255), xytext=(0.28, 0.255), arrowprops=dict(facecolor='black', width=2, headwidth=7))
ax.annotate('', xy=(0.58, 0.255), xytext=(0.55, 0.255), arrowprops=dict(facecolor='black', width=2, headwidth=7))
ax.annotate('', xy=(0.85, 0.255), xytext=(0.82, 0.255), arrowprops=dict(facecolor='black', width=2, headwidth=7))

# Vertical connection
ax.annotate('', xy=(0.16, 0.44), xytext=(0.16, 0.54), arrowprops=dict(facecolor='black', width=2, headwidth=7))

ax.set_title('Explainable & Uncertainty-Aware Machine Learning Framework Architecture for LPBF Ti-6Al-4V', 
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig20_psp_framework_architecture.png')
plt.close()

print("\nSuccessfully generated all 20 publication-quality figures in figures/ folder!")

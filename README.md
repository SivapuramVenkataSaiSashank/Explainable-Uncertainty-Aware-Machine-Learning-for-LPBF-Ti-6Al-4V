# Explainable & Uncertainty-Aware Machine Learning for LPBF Ti-6Al-4V

### **Predicting Porosity, Strength, and Ductility Across Process–Structure–Property Features with Leakage-Free Grouped Validation, SHAP Interpretability, Bayesian/Conformal Uncertainty, and GPU Deep Learning**

---

## 📌 Project Overview & Metadata

* **Course:** 22AIE449 – Introduction to Digital Manufacturing
* **Project Team:**
  * **G. Subhash Guptha** (Roll: 23015)
  * **R. Gautam** (Roll: 23022)
  * **K. Mithul** (Roll: 23025)
  * **S.V. Sai Sashank** (Roll: 23043)
* **Submitted to:** **Dr. Sathies T**
* **Base Paper Reference:**
  * Luo, Q., Yin, L., Simpson, T. W., & Beese, A. M. (2023). *“Dataset of process-structure-property feature relationship for laser powder bed fusion additive manufactured Ti-6Al-4V material.”* **Data in Brief**, 46, 108911. [DOI: 10.1016/j.dib.2023.108911](https://doi.org/10.1016/j.dib.2023.108911)
  * Accompanying Technical Paper: Luo, Q., Yin, L., Simpson, T. W., & Beese, A. M. (2022). *“Effect of processing parameters on pore structures, grain features, and mechanical properties in Ti-6Al-4V by laser powder bed fusion.”* **Additive Manufacturing**, 56, 102915.
* **Public Repository:** Zenodo Record 6587905.
* **Hardware Acceleration:** NVIDIA GeForce RTX 4060 Laptop GPU via PyTorch CUDA (`cuda:0`).

---

## 💻 Hardware & Software Environment

* **Operating System:** Windows 11
* **Compute Device:** `cuda:0` (NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, Compute Capability 8.9)
* **Python Environment:** Python 3.12
* **Core Libraries:**
  * **Deep Learning:** PyTorch 2.6.0 with CUDA 12.4
  * **Gradient Boosting & Trees:** XGBoost 2.1.4, Scikit-Learn 1.9.1
  * **Explainability:** SHAP 0.52.0 (TreeExplainer, Beeswarm, Scatter, Bar)
  * **Scientific Computing & Stats:** NumPy, SciPy, Pandas
  * **Visualization:** Matplotlib, Seaborn (300 DPI high-resolution figures)
  * **Notebook Automation:** `nbformat`, `nbclient`, `ipykernel`

---

## 🧭 Chronological Step-by-Step Execution Journey

Below is a detailed, transparent account of **each and every step** performed to build, validate, explain, and optimize the machine learning framework:

```mermaid
flowchart TD
    S1["Step 1: Audit Literature & Presentation Requirements"] --> S2["Step 2: Parse Raw Micro-CT Defect Distributions (35 Files)"]
    S2 --> S3["Step 3: Extract, Standardize & Build Master PSP Dataset (42 Conditions)"]
    S3 --> S4["Step 4: Physics-Informed Feature Engineering (LED, VED, MED, Pv, Regimes)"]
    S4 --> S5["Step 5: EDA & Physical Metallurgy Models (Ryshkewitch, Hall-Petch, Parity)"]
    S5 --> S6["Step 6: Empirical Leakage-Free Validation Audit (Random vs Grouped CV)"]
    S6 --> S7["Step 7: Multi-Model Benchmark Suite & Feature Group Ablation"]
    S7 --> S8["Step 8: Explainable AI with SHAP & Physical Threshold Transitions"]
    S8 --> S9["Step 9: Dual-Engine Uncertainty Quantification & OOD Trust Boundaries"]
    S9 --> S10["Step 10: Constrained Multi-Objective Optimization (Golden Process Window)"]
    S10 --> S11["Step 11: High-Resolution Figure Suite Generation (20 Figures at 300 DPI)"]
    S11 --> S12["Step 12: Automated Assembly & Full Execution of Master Jupyter Notebook"]
    S12 --> S13["Step 13: Workspace Hygiene & Artifact Consolidation"]
```

---

### **Step 1: Literature & Presentation Audit**
1. Inspected the base paper (`Dataset of process-structure-property feature relationship for LPBF.pdf`) and extracted all experimental conditions, testing equipment, and metallurgical tables (Tables 1 through 9).
2. Inspected the presentation review slides (`LPBF_Ti6Al4V_Project_Review_Final_FIXED.pptx`) to align strictly with the team's proposed objectives:
   - Feature group breakdown: Process-only $\to$ Process + Defect $\to$ Full Process–Structure–Property (PSP).
   - Leakage-free validation focus (Slide 10): Empirical proof that random cross-validation produces optimism bias.
   - Explainability requirement: SHAP beeswarm, bar, and dependence plots.
   - Uncertainty quantification requirement: GPR credible intervals, deep learning ensembles/MC Dropout, and conformal prediction.
   - Optimization requirement: Constrained multi-objective Pareto front isolating the *Golden Process Window*.

---

### **Step 2: Ingestion & Aggregation of Raw Micro-CT Defect Files**
1. The `dataset/` directory contains 35 CSV files (`sample 1.csv` through `sample 42.csv`), corresponding to individual 3D X-ray computed tomography (XCT) scans.
2. Designed a parsing engine to read every sample file, skipping image-processing metadata headers while preserving morphological measurements:
   - Pore Volume ($V$ in $\text{mm}^3$)
   - Pore Equivalent Spherical Diameter ($D_{\text{eq}}$ in $\text{mm}$)
   - Pore Surface Area ($A$ in $\text{mm}^2$)
   - Pore Sphericity ($\Psi = \frac{\pi^{1/3}(6V)^{2/3}}{A}$)
3. Aggregated over **1,000,000 microscopic pore records** into condition-level metrics:
   - Total Pore Count per gauge section
   - Total Internal Pore Volume ($\text{mm}^3$)
   - Mean & Median Pore Volume ($\text{mm}^3$)
   - Mean & 95th Percentile Pore Diameter ($\mu\text{m}$)
   - Mean & Median Pore Sphericity
4. Exported the aggregated statistics to [`lpbf_raw_pore_aggregated.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/lpbf_raw_pore_aggregated.csv).

---

### **Step 3: Master PSP Dataset Construction & Standardization**
1. Meticulously parsed Tables 2 through 9 from the base paper to build the master tabular dataset:
   - **Table 2:** Process parameters ($P$, $v$) and descriptors ($LED$, $MED$, $Pv$).
   - **Table 3:** Archimedes Porosity and XCT Porosity (mean and standard deviation).
   - **Table 4:** Size-binned pore counts ($<200\,\mu\text{m}$, $200-300\,\mu\text{m}$, $300-500\,\mu\text{m}$, $>500\,\mu\text{m}$).
   - **Table 5:** Equivalent diameter, sphericity (numerical and volume-weighted), and projected pore area orthogonal to the tensile loading axis.
   - **Table 6:** Surface roughness ($Ra$ and $\text{RMS}$).
   - **Table 7:** Prior-$\beta$ grain equivalent diameter and columnar aspect ratio.
   - **Table 8:** Tensile Yield Strength, Ultimate Tensile Strength (UTS), Elongation to necking, and Elongation to fracture (with DIC standard deviations across 4 specimens/condition).
   - **Table 9:** Vickers microhardness ($HV$) and Young's modulus ($E$).
2. **Missing Value Audit:** Identified that conditions 19 and 22–27 were tested for optical microscopy, mechanical properties, and hardness, but were not scanned via micro-CT in the original experimental campaign. Created an explicit `Has_XCT_Data` boolean flag to ensure statistical integrity.
3. Built and exported the master dataset [`lpbf_ti6al4v_psp_dataset.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/lpbf_ti6al4v_psp_dataset.csv) ($42 \times 46$) and the official metadata dictionary [`lpbf_ti6al4v_data_dictionary.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/lpbf_ti6al4v_data_dictionary.csv).

---

### **Step 4: Physics-Informed Feature Engineering**
Engineered analytical manufacturing descriptors grounded in thermal transfer and melt pool dynamics:
1. **Linear Energy Density (LED) [$\text{J/m}$]:**
   $$\text{LED} = \frac{P}{v} \times 10^3$$
2. **Volumetric Energy Density (VED) [$\text{J/mm}^3$]:**
   $$\text{VED} = \frac{P}{v \cdot h \cdot t}$$
   Using constant layer thickness $t = 60\,\mu\text{m}$ ($0.060\,\text{mm}$) and hatch spacing $h = 82\,\mu\text{m}$ ($0.082\,\text{mm}$).
3. **Modified Volumetric Energy Density (MED) [$\text{J}/\sqrt{\text{m}}/\text{s}$]:**
   $$\text{MED} = C \cdot \frac{P}{\sqrt{v}}$$
   Derived from 1D Rosenthal conduction equations accounting for transient melt pool depth.
4. **Melt Pool Instability / Balling Factor ($Pv$) [$\text{W}\cdot\text{m/s}$]:**
   $$Pv = P \times (v \times 10^{-3})$$
   Capturing Rayleigh-Plateau hydrodynamic capillary instability that triggers melt pool humping.
5. **Defect Regime Classification:**
   - **Lack of Fusion (LOF):** $\text{LED} < 260\,\text{J/m}$ or $P \le 115\,\text{W}$.
   - **Keyholing / Balling:** $\text{LED} \ge 415\,\text{J/m}$ or $Pv \ge 600\,\text{W}\cdot\text{m/s}$.
   - **Optimal / Conduction:** Stable melt pool operating window.

---

### **Step 5: Exploratory Data Analysis & Physical Metallurgy Relationships**
Fitted and verified classic materials science relationships to serve as physical benchmarks for ML models:
1. **Ryshkewitch-Duckworth Exponential Porosity–Strength Model:**
   $$\sigma_{\text{UTS}} = \sigma_0 \exp\left(-b \cdot \frac{p}{100}\right)$$
   Fitted parameters: $\sigma_0 = 1023.4\,\text{MPa}$ (intrinsic pore-free strength), $b = 3.86$ (porosity sensitivity).
2. **Ductility Cliff:** Demonstrated an abrupt collapse in tensile elongation to fracture from $>15\%$ in dense specimens down to $<1.3\%$ when internal porosity exceeds $1.0\%$.
3. **Hall-Petch Grain Boundary Strengthening:**
   $$\sigma_y = \sigma_0 + k_y d^{-1/2} = 884.2 + 2248.6 \, d^{-1/2} \quad (r = 0.42)$$
   Quantifying prior-$\beta$ grain boundary dislocation pileup effects.
4. **Archimedes vs. Micro-CT Parity:** Revealed that immersion Archimedes density systematically overestimates apparent porosity due to surface-connected open pores and bubble entrapment.
5. **Multi-Feature Spearman Correlation Matrix:** Generated a $15 \times 15$ rank correlation heatmap capturing non-linear process-structure-property coupling.

---

### **Step 6: Leakage-Free Validation Strategy (Slide 10 Focus)**
Conducted an empirical experiment to address the data leakage warning highlighted in Slide 10:
1. **Scheme A (Naive Random 5-Fold CV):** Allowed samples from correlated power/speed clusters to mix freely between train and test folds.
2. **Scheme B (Grouped 5-Fold CV via `GroupKFold`):** Enforced that all conditions within the same laser power cluster remain strictly within the same fold, preventing any cluster leakage.
3. **Empirical Results:**
   - Random 5-Fold CV: Apparent $R^2 = 0.220 \pm 0.555$, $\text{RMSE} = 35.83\,\text{MPa}$.
   - Grouped 5-Fold CV: Genuine $R^2 = -0.071 \pm 1.021$, $\text{RMSE} = 37.98\,\text{MPa}$.
   - **Leakage Optimism Gap:** Random splitting artificially inflates $R^2$ by $+0.29$!
   - Confirmed that Grouped CV is required to evaluate true generalization to unseen process regimes. Exported results to [`validation_leakage_comparison.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/validation_leakage_comparison.csv).

---

### **Step 7: Multi-Model Benchmark Suite & Feature Group Ablation**
Evaluated 6 algorithms across 3 hierarchical feature groups under leakage-free Grouped 5-Fold CV:
* **Models:** Ridge Regression, SVR (RBF Kernel), Random Forest, XGBoost Regressor, Gaussian Process Regressor, and a custom **PyTorch Deep MLP accelerated on GPU** (`cuda:0`).
* **Targets:** Internal XCT Porosity, Ultimate Tensile Strength (UTS), 0.2% Offset Yield Strength (YS), Tensile Elongation to Fracture, and Vickers Hardness ($HV$).
* **Ablation Comparison (Predicting UTS):**
  - **Group 1 (Process-Only):** Best $R^2 = 0.262$, $\text{RMSE} = 32.86\,\text{MPa}$ (SVR).
  - **Group 2 (Process + Defects):** $R^2$ surges to **$0.810$**, $\text{RMSE} = \mathbf{18.60}\,\text{MPa}$ with the **PyTorch GPU Deep MLP** ($R^2 = 0.623$ with Random Forest).
  - **Group 3 (Full PSP):** Reaches **$R^2 = 0.825$**, $\text{RMSE} = 17.90\,\text{MPa}$.
* **Key Finding:** Proved the central metallurgical hypothesis that defect morphology (sphericity, pore count, projected area) is the vital physical bridge connecting processing to mechanical properties. Exported to [`model_benchmark_results.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/model_benchmark_results.csv).

---

### **Step 8: Explainable AI (XAI) & Metallurgical Interpretability**
Applied cooperative game theory (SHAP) and model-agnostic interpretability tools:
1. **SHAP Beeswarm Summary:** Identified internal XCT porosity as the primary negative driver of tensile strength ($-50$ to $-80\,\text{MPa}$ penalty for high porosity), while near-zero porosity yields positive contributions ($+25$ to $+35\,\text{MPa}$).
2. **SHAP Global Importance Ranking:** Ranked feature contributions: $\text{Porosity} > \text{LED} > \text{Laser Power} > \text{Pore Sphericity} > \text{Grain Diameter}$.
3. **Nonlinear Threshold Discovery:** SHAP dependence revealed that linear energy density ($\text{LED}$) has a critical transition at $\sim 260\,\text{J/m}$. Below this threshold, lack-of-fusion voids cause severe property drops; between $260$ and $380\,\text{J/m}$, contributions plateau at optimal levels.
4. **Permutation Importance:** Executed 30-repeat permutation tests with empirical error bars.
5. **Partial Dependence Plots (PDP):** Computed marginal effect curves for $LED$, $P$, and Porosity on UTS.

---

### **Step 9: Dual-Engine Uncertainty Quantification & OOD Trust**
Implemented a comprehensive uncertainty framework:
1. **Bayesian Epistemic Uncertainty via Gaussian Process Regression:** Produced exact analytical $95\%$ credible intervals $\mu(x) \pm 1.96\sigma(x)$.
2. **GPU-Accelerated Monte Carlo (MC) Dropout:** Drew $T = 100$ stochastic forward passes with test-time dropout ($p = 0.20$) on the NVIDIA RTX 4060 GPU to calculate neural epistemic variance.
3. **Distribution-Free Conformal Prediction:**
   - 90% Target Confidence: Verified **$88.6\%$ empirical coverage** ($\pm 63.74\,\text{MPa}$).
   - 95% Target Confidence: Verified **$94.3\%$ empirical coverage** ($\pm 82.50\,\text{MPa}$).
4. **Out-of-Distribution (OOD) Trust Metric:** Formulated normalized Mahalanobis distance ($D_M \le 4.33$) in process space to automatically flag unvalidated extrapolation queries.

---

### **Step 10: Constrained Multi-Objective Optimization: The "Golden Process Window"**
Formulated the manufacturing optimization problem:
$$\min \text{Porosity}(P, v), \quad \max \text{UTS}(P, v), \quad \max \text{Elongation}(P, v)$$
Subject to:
$$115 \le P \le 475\,\text{W}, \quad 400 \le v \le 1500\,\text{mm/s}, \quad \text{LED} \in [190, 460]\,\text{J/m}, \quad Pv \le 720\,\text{W}\cdot\text{m/s}$$
1. Executed non-dominated sorting over 1,143 simulated candidate conditions to extract the 3D Pareto frontier.
2. **Identified Golden Process Window:**
   - Laser Power ($P$): $235 - 275\,\text{W}$
   - Scan Speed ($v$): $1200 - 1250\,\text{mm/s}$
   - Linear Energy Density ($\text{LED}$): $194 - 225\,\text{J/m}$
   - Predicted Properties: Porosity $< 0.05\%$, $\text{UTS} > 1030\,\text{MPa}$, Elongation $> 15\%$, with high build productivity! Exported to [`pareto_optimal_conditions.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/pareto_optimal_conditions.csv).

---

### **Step 11: Generation of Publication-Grade Visualizations**
Generated **20 unique figures at 300 DPI** saved to [`figures/`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/figures) using [`generate_all_figures.py`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/generate_all_figures.py):
* `fig1_process_space_map.png`
* `fig2_eda_process_maps.png`
* `fig3_physical_models.png`
* `fig4_raw_pores_distribution.png`
* `fig5_correlation_heatmap.png`
* `fig6_validation_leakage_comparison.png`
* `fig7_feature_group_ablation.png`
* `fig8_multi_model_benchmark_comparison.png`
* `fig9_shap_summary.png`
* `fig10_shap_feature_importance_bars.png`
* `fig11_shap_dependence_thresholds.png`
* `fig12_permutation_feature_importance.png`
* `fig13_partial_dependence_plots.png`
* `fig14_uncertainty_gpr_intervals.png`
* `fig15_uncertainty_mc_dropout_gpu.png`
* `fig16_conformal_prediction_coverage.png`
* `fig17_out_of_distribution_trust_map.png`
* `fig18_multi_objective_pareto_frontier.png`
* `fig19_golden_process_window.png`
* `fig20_psp_framework_architecture.png`

---

### **Step 12: Automated Assembly & Full Execution of Master Jupyter Notebook**
Using [`build_and_execute_notebook.py`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/build_and_execute_notebook.py), constructed and executed:
**[`LPBF_Ti6Al4V_Explainable_ML_Framework.ipynb`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/LPBF_Ti6Al4V_Explainable_ML_Framework.ipynb)**
- 22 structured cells containing LaTeX mathematical equations, hardware logs, PyTorch GPU training loops, interactive Pandas tables, and embedded high-resolution figures.
- Executed via `nbclient` in 8.3 seconds with zero warnings or errors.

---

### **Step 13: Workspace Hygiene & Redundancy Removal**
- Removed temporary scratch text files (`page_*.txt`, `pdf_text.txt`, `pptx_content.txt`).
- Removed redundant intermediate runner scripts (`run_full_analysis.py`).
- Removed duplicate legacy figure filenames (`fig10_shap_dependence.png`, `fig11_uncertainty_gpr_intervals.png`, `fig13_pareto_front_optimization.png`).
- Cleaned the repository so every remaining file is active, necessary, and documented.

---

## 📂 Repository File Directory

| File / Folder | Purpose & Description |
| :--- | :--- |
| **[`LPBF_Ti6Al4V_Explainable_ML_Framework.ipynb`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/LPBF_Ti6Al4V_Explainable_ML_Framework.ipynb)** | **Master Executed Jupyter Notebook** (Complete project deliverable with GPU execution outputs and all 20 figures). |
| **[`LPBF_Ti6Al4V_Project_Review_Final_FIXED.pptx`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/LPBF_Ti6Al4V_Project_Review_Final_FIXED.pptx)** | Original presentation review slides defining methodology and requirements. |
| **[`Dataset of process-structure-property feature relationship for LPBF.pdf`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/Dataset%20of%20process-structure-property%20feature%20relationship%20for%20LPBF.pdf)** | Original peer-reviewed base paper (Luo et al., 2023, *Data in Brief*). |
| **[`lpbf_ti6al4v_psp_dataset.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/lpbf_ti6al4v_psp_dataset.csv)** | Curated 42-condition master PSP dataset ($42 \times 46$). |
| **[`lpbf_ti6al4v_data_dictionary.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/lpbf_ti6al4v_data_dictionary.csv)** | Complete data dictionary metadata (units, ranges, characterization methods). |
| **[`lpbf_raw_pore_aggregated.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/lpbf_raw_pore_aggregated.csv)** | Pore population metrics aggregated from $>1\,\text{million}$ micro-CT raw records. |
| **[`model_benchmark_results.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/model_benchmark_results.csv)** | Cross-validated metrics across 6 models, 5 targets, and 3 feature groups. |
| **[`validation_leakage_comparison.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/validation_leakage_comparison.csv)** | Empirical validation comparison proving data leakage optimism bias. |
| **[`pareto_optimal_conditions.csv`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/pareto_optimal_conditions.csv)** | Identified Pareto-optimal conditions isolating the Golden Process Window. |
| **[`build_dataset.py`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/build_dataset.py)** | Script that compiles raw micro-CT CSVs and tables into clean master datasets. |
| **[`generate_all_figures.py`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/generate_all_figures.py)** | Script that trains models, performs SHAP/GPR/conformal calculations, and renders all 20 figures. |
| **[`build_and_execute_notebook.py`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/build_and_execute_notebook.py)** | Script that generates and executes the master `.ipynb` notebook via `nbclient`. |
| **[`dataset/`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/dataset)** | Folder containing 35 raw micro-CT defect CSV files (`sample 1.csv` to `sample 42.csv`). |
| **[`figures/`](file:///f:/%23sem7/22AIE449%20Introduction%20to%20Digital%20Manufacturing/Project/figures)** | Folder containing all 20 high-resolution (300 DPI) publication figures. |

---

## 🚀 How to Run & Reproduce

To regenerate the entire project from scratch in a single workflow:

```bash
# 1. Rebuild the master dataset and raw pore aggregates
python build_dataset.py

# 2. Train models, compute SHAP/Uncertainty/Pareto, and render all 20 figures
python generate_all_figures.py

# 3. Build and execute the master Jupyter Notebook with GPU acceleration
python build_and_execute_notebook.py

# 4. Launch Jupyter Notebook to inspect the executed framework
jupyter notebook LPBF_Ti6Al4V_Explainable_ML_Framework.ipynb
```

---

## 🎯 Final Conclusions & Engineering Guidelines

1. **Defects Bridge Process and Properties:** Machine parameters ($P, v$) alone only explain $R^2 \approx 0.26$ of strength variance due to thermal fluctuations. Incorporating 3D pore morphology elevates explanatory power to **$R^2 = 0.810$** ($\text{RMSE} = 18.60\,\text{MPa}$ with PyTorch Deep MLP on GPU).
2. **Data Leakage Must Be Prevented:** Naive random splitting inflates apparent $R^2$ scores by $+0.29$. Grouped cross-validation (`GroupKFold`) is mandatory in digital manufacturing.
3. **Thresholding Behavior:** High mechanical integrity requires $\text{LED} \ge 260\,\text{J/m}$ to eliminate lack of fusion, combined with $Pv \le 720\,\text{W}\cdot\text{m/s}$ to prevent balling instability.
4. **Golden Process Window:** Operating at $P = 235 - 275\,\text{W}$ and $v = 1200 - 1250\,\text{mm/s}$ achieves optimal density ($>99.95\%$), high tensile strength ($>1030\,\text{MPa}$), high ductility ($>15\%$), and maximized build productivity.
5. **Operational Trust:** Coupling Gaussian Process / MC Dropout uncertainty with conformal prediction error bars and Mahalanobis out-of-distribution detection ensures safe, dependable industrial deployment.

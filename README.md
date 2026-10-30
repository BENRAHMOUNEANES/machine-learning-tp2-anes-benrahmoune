# Machine Learning — Practical Session 2: Python for Standardized PCA

**National Higher School of Advanced Technologies (ENST)**  
**Academic Year:** 2026–2027  
**Student:** Anes Benrahmoune  
**Level:** 4th Year Student (Artificial Intelligence Specialty)  
**Group:** Group 1  
**GitHub:** [BENRAHMOUNEANES](https://github.com/BENRAHMOUNEANES)  
**Course:** Machine Learning  

---

## 1. Project Overview & Repository Structure
This repository contains the complete implementation and report for Practical Session 2 on **Standardized Principal Component Analysis (PCA)**. The solution implements PCA from scratch using linear algebra and NumPy, and compares the theoretical and numerical results with **Scikit-learn** (`StandardScaler` + `PCA`).

The repository structure is clean and self-contained:
```
tp2/
├── README.md                      # Comprehensive report, comparisons & interpretation
├── requirements.txt               # Dependencies
├── TP2_Standardized_PCA.ipynb     # Fully executed Jupyter Notebook with all outputs & plots
└── Work session 2.pdf             # Original lab specification sheet
```

---

## 2. Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the Jupyter Notebook
jupyter notebook TP2_Standardized_PCA.ipynb
```

---

## 3. Step-by-Step Numerical Results (Questions 1 to 18)

### Dataset
The dataset records the performance of six students ($S_1, \dots, S_6$) across four academic disciplines:
```
Student  Mathematics  Statistics  Programming  Algorithms
S1       12           11          14           13
S2       15           14          16           15
S3        8            9          10            9
S4       17           16          18           17
S5       10           12          11           10
S6       14           15          13           14
```

---

### Part A: Standardized PCA Implementation from Scratch

1. **Standardization:**
   Features are standardized using mean and population standard deviation ($ddof=0$):
   $$Z = \frac{X - \mu}{\sigma}$$
2. **Correlation Matrix ($\mathbf{R}$):**
   $$\mathbf{R} = \frac{1}{N} \mathbf{Z}^T \mathbf{Z} = \begin{pmatrix}
   1.0000 & 0.9263 & 0.9451 & 0.9911 \\
   0.9263 & 1.0000 & 0.7717 & 0.8744 \\
   0.9451 & 0.7717 & 1.0000 & 0.9635 \\
   0.9911 & 0.8744 & 0.9635 & 1.0000
   \end{pmatrix}$$
3. **Eigenvalues and Eigenvectors:**
   Calculated using `np.linalg.eigh(R)`:
   - $\lambda_1 = 3.7397$ (dominant component)
   - $\lambda_2 = 0.2402$
   - $\lambda_3 = 0.0191$
   - $\lambda_4 = 0.0009$
   Total variance $= \sum \lambda_j = 4.0$.
4. **Projection Matrix ($\mathbf{W}$):**
   Formed by the two leading eigenvectors:
   $$\mathbf{W} = \begin{pmatrix}
   -0.5167 & 0.0320 \\
   -0.4770 & 0.7822 \\
   -0.4928 & -0.5922 \\
   -0.5125 & -0.1909
   \end{pmatrix}$$
5. **Projected Coordinates ($\mathbf{F}_{\text{scratch}} = \mathbf{Z} \mathbf{W}$):**
   ```
   Student       PC1      PC2
   S1         0.4166  -0.6740
   S2        -1.4165  -0.2373
   S3         2.9506  -0.2279
   S4        -2.8814  -0.1357
   S5         1.6520   0.4827
   S6        -0.7214   0.7921
   ```
6. **Explained Variance Ratios:**
   - $\text{PC1}$: $\mathbf{93.49\%}$ (Cumulative: $93.49\%$)
   - $\text{PC2}$: $\mathbf{6.01\%}$ (Cumulative: $\mathbf{99.50\%}$)

---

### Part B & C: Scikit-Learn Comparison & Insights

1. **Coordinate Comparison:**
   $$\mathbf{F}_{\text{sklearn}} = \begin{pmatrix}
   -0.4166 & -0.6740 \\
   +1.4165 & -0.2373 \\
   -2.9506 & -0.2279 \\
   +2.8814 & -0.1357 \\
   -1.6520 & +0.4827 \\
   +0.7214 & +0.7921
   \end{pmatrix}$$
   Notice that $|F_{\text{scratch}}| == |F_{\text{sklearn}}|$.
2. **Sign Discrepancy Explanation (Question 17):**
   Because an eigenvector satisfies $\mathbf{R} \mathbf{v} = \lambda \mathbf{v}$, it also satisfies $\mathbf{R} (-\mathbf{v}) = \lambda (-\mathbf{v})$. Both orientations $+\mathbf{v}$ and $-\mathbf{v}$ are valid unit eigenvectors. Scikit-learn's LAPACK SVD chosen basis is an isometric reflection across the vertical axis that strictly preserves all Euclidean distances, angles, and variances.
3. **Eigenvalues vs `explained_variance_` (Question 15):**
   Scikit-learn divides sample variance by $N - 1 = 5$ instead of $N = 6$:
   $$\text{Var}_{\text{sklearn}} = 3.7397 \times \frac{6}{5} = \mathbf{4.4877}$$
   The normalized explained variance ratios are completely identical ($93.49\%$ and $6.01\%$).
4. **Total Retained Variance & Performance Interpretation (Question 18):**
   - **Total Variance Retained:** $93.49\% + 6.01\% = \mathbf{99.50\%}$. Dimensionality is reduced from 4 to 2 with only $0.50\%$ loss of information.
   - **PC1 (General Academic Mastery — 93.49%):** Reflects overall academic ability. Student **S4** is the top overall student, while **S3** is the weakest.
   - **PC2 (Curricular Specialization — 6.01%):** Contrasts Statistics ($+0.78$) against Programming ($-0.59$) and Algorithms ($-0.19$). **S6 and S5** excel in Statistics; **S1 and S4** excel in practical programming.

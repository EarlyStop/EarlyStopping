### README – Replication Scripts for *“\[Title of the paper]”* (arXiv:2503.16753)

This folder contains one script per figure (and one for Table 1) from the paper.
Running the ```main_replication.py``` script will reproduce all figures exactly as they appear in the manuscript exccept for the figures associated with the regression tree results, which take longer to run and are therefore separated out into their own scripts.

---

#### Script‑to‑Figure map (```simulations/main_replication.py```)

1. **`visualise_error_decomposition.py`**
     • Recreates the weak‑ and strong‑error decompositions shown in **Figure 2 (a) & (b)**.

2. **`signals.py`**
     • Generates the true and reconstructed signals plotted in **Figure 2 (c)**.

3. **`TruncatedSVD_Replication.py`**
     • Computes the **relative efficiencies** in **Figure 2 (d)** (Truncated SVD).

4. **`Landweber_Replication.py`**
     • Produces the efficiency curves in **Figure 3 (a) & (b)** (Landweber iteration).

5. **`ConjugateGradient_Replication.py`**
     • Recreates **Figure 4 (a) & (b)** using **Conjugate‑Gradient** regularisation.

6. **`L2Boost_signals.py`**
     • Generates the signal examples in **Figure 5 (a) & (b)** (L2 Boosting).

7. **`L2Boost_Replication.py`**
     • Computes the replication study in **Figure 6** (L2 Boosting).

8. **`signal_estimation_comparison.py`**
      • Recreates the signal‑estimation plots in **Figure 9 (a) & (b)**.

9. **`phillips_data.py`**
      • Generates the estimation results for phillips data in **Figure 10 (a) & (b)**.

10. **`ComparisonStudy.py`**
      • Produces the stopping‑time and error curves in **Figure 11 (a) & (b)**.

11. **`timing_es.py`**
      • Computes the numbers reported in **Table 1** (stopping times & errors).

12. **`Simulation_counterexample_landweber.py`**
      • Recreates **Figure 12 (b) & (d)** (error decomposition – Landweber).

13. **`Simulation_counterexample_tSVD.py`**
      • Recreates **Figure 12 (a) & (c)** (error decomposition – Truncated SVD).

---

#### Script‑to‑Figure map (```simulations/RegressionTree_additive_plots.py```)

**`RegressionTree_additive_plots.py`**
  • Produces the additive‑model signals in **Figure 7** (Regression trees).

---

#### Script‑to‑Figure map (```simulations/RegressionTree_Replication.py```)

**`RegressionTree_Replication.py`**
  • Replicates the efficiency results in **Figure 8** (Regression trees).

---

#### How to run

From the repository root:

```bash
python simulations/{insert_script_name}.py   # e.g. python simulations/RegressionTree_additive_plots.py
```

Each script saves its output figure(s) and prints any key numerical values to the console.

# Early stopping
Early stopping is a python library implementing computationally efficient model selection methods for iterative estimation procedures based on the theory in

- G. Blanchard, M. Hoffmann, M. Reiß.
  <a href="https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-12/issue-2/Early-stopping-for-statistical-inverse-problems-via-truncated-SVD-estimation/10.1214/18-EJS1482.full">
    "Early stopping for statistical inverse problems via truncated SVD estimation"
  </a>.
  In: <em>Electronic Journal of Statistics</em> 12(2): 3204-3231 (2018).

- G. Blanchard, M. Hoffmann, M. Reiß.
  <a href="https://arxiv.org/abs/1606.07702">
    "Optimal adaptation for early stopping in statistical inverse problems"
  </a>.
  In: <em>SIAM/ASA Journal of Uncertainty Quantification</em> 6(3), 1043–1075 (2018).

- B. Stankewitz. 
  <a href="https://arxiv.org/abs/2210.07850v1">
    "Early stopping for L2-boosting in high-dimensional linear models"
  </a>.
  arXiv:2210.07850 [math.ST] (2022).

Check out the [documentation](https://esfiep.github.io/EarlyStopping/) for more information.

## Development notes

### Installation for development
Required Installations
- git
- Python3
- JupyterNotebooks


Manual setup:
```bash
python3 -m pip install build virtualenv               # Install build tools
git clone https://github.com/ESFIEP/EarlyStopping.git # Clone git repository
python3 -m build                                      # Build package
python3 -m venv myenv                                 # Create virtual environment
source myenv/bin/activate                             # Activate virtual environment
python3 -m pip install numpy ipykernel                # Install python packages to the environment
python3 -m pip install -e .                           # Install the EarlyStopping package in editable mode
python3 -m ipykernel install --user --name=myenv      # Create Jupyter kernel from the environment
```
From the notebooks directory open the Jupyter notebook example.ipynb with the kernel myenv and run the code!

### Creating documentation locally
General instructions for documenting projects with [sphinx](https://www.sphinx-doc.org/en/master/index.html).

To generate the documentation locally run
```bash
sphinx-build -M html docs/source docs/build
```
in the EarlyStopping directory. The documentation will be generated in docs/build/html. To view them locally open index.html with firefox or another browser.

Under Linux, it was necessary to include
```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
```
in docs/source/conf.py. Please only include this locally and never push the change.



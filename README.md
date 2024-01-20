# Early stopping
Early stopping is a python library implementing computationally efficient model selection methods for iterative estimation procedures based on the theory in

- B. Stankewitz. 
  <a href="https://arxiv.org/abs/2210.07850v1">
    "Early stopping for L2-boosting in high-dimensional linear models".
  </a>
  arXiv:2210.07850 [math.ST] (2022).

- G. Blanchard, M. Hoffmann, M. Reiß.
  <a href="https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-12/issue-2/Early-stopping-for-statistical-inverse-problems-via-truncated-SVD-estimation/10.1214/18-EJS1482.full">
    "Early stopping for statistical inverse problems via truncated SVD estimation"
  </a>
  In: <em>Electronic Journal of Statistics</em> 12(2): 3204-3231 (2018).

Check out the [documentation](https://esfiep.github.io/EarlyStopping/) for more information.

# Installation instructions
Required Installations
- git
- Python3
- JupyterNotebooks

Install build tools
```bash
python3 -m pip install build virtualenv
```

Clone git repository
```bash
git clone https://github.com/ESFIEP/EarlyStopping.git
```

Move to repository directory
```bash
cd EarlyStopping
```

Build package
```bash
python3 -m build
```

Create virtual environment
```bash
python3 -m venv myenv
```

Activate virtual environment
```bash
source myenv/bin/activate
```

Install python packages to the environment
```bash
python3 -m pip install numpy ipykernel
```

Install the EarlyStopping package in editable mode
```bash
python3 -m pip install -e . 
```

Create Jupyter kernel from the environment "myenv"
```bash
python3 -m ipykernel install --user --name=myenv
```

From the notebooks directory open the Jupyter notebook example.ipynb with the kernel myenv and run the code!


# Instructions to create documentation locally
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



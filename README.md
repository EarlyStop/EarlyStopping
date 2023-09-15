# Early stopping
This is a simple package for early stopping methods

# Installation instructions
Required Installations
- Python3
- JupyterNotebooks

Install build tools
```bash
python3 -m pip install build venv
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

Open Jupyter notebook example.ipynb with the kernel myenv and run the code!



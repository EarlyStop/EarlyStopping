"""
A minimal example for the ADNI application
==========================================
"""
import os
import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path

password = os.environ.get("PASSWORD")

result = subprocess.run(
    [
        "gpg",
        "--batch",
        "--yes",
        "--pinentry-mode", "loopback",           # <-- important in CI
        "--passphrase", 
        password,
        "--decrypt", 
        "data/ADNI_data.gpg",
    ],
    capture_output=True,
    check=True,
    text=True,  # so result.stdout is already str
)

df = pd.read_csv(StringIO(result.stdout))
print(df.head())

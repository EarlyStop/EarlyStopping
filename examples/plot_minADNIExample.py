"""
A minimal example for the ADNI application
==========================================
"""

import subprocess
import os
from io import StringIO
import pandas as pd

password = os.environ["GPG_PASS"]  # safer than hardcoding

result = subprocess.run(
    [
        "gpg",
        "--batch",              # no interactive prompts
        "--yes",                # assume "yes" on overwrite
        # "--passphrase", password,
        "--passphrase", "HeadPhoneChalkEraserMat",
        "--decrypt", "testData.gpg",
    ],
    capture_output=True,
    check=True,
)

type(result)
type(result.stdout.decode("utf-8"))

df = pd.read_csv(StringIO(result.stdout.decode("utf-8")))

print(df.head())












import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es
import gnupg
from io import StringIO

import subprocess
import pandas as pd
from io import StringIO

# Path to your encrypted file
encrypted_file = "testData.gpg"

# Decrypt the file
result = subprocess.run(
    ["gpg", "--decrypt", encrypted_file],
    capture_output=True,
    check=True
)

# Convert bytes to string (UTF-8)
decrypted_csv = result.stdout.decode("utf-8")

# Load into Pandas using StringIO
df = pd.read_csv(StringIO(decrypted_csv))

print(df.head())



sns.set_theme()
password =  os.environ.get("PASSWORD")

if os.path.exists("testData.gpg"):
    print("File exists!")

gpg = gnupg.GPG()
with open("testData.gpg", "rb") as f:
    result = gpg.decrypt_file(f, passphrase="HeadPhoneChalkEraserMat")

df = pd.read_csv(StringIO(result))



# %%
# Opening the file
# -------------------------
# Here we open the csv file

gpg = gnupg.GPG()
with open("testData.gpg", "rb") as f:
    result = gpg.decrypt_file(f, passphrase=password)

type(result)
print(result)
print("test")

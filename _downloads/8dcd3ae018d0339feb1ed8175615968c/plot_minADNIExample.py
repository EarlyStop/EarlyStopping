"""
A minimal example for the ADNI application
==========================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es
import gnupg


sns.set_theme()
password =  os.environ.get("PASSWORD")

if os.path.exists("testData.gpg"):
    print("File exists!")

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

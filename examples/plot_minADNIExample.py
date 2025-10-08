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

# %%
# Opening the file
# -------------------------
# Here we open the csv file

gpg = gnupg.GPG()
with open("testData.gpg", "rb") as f:
    result = gpg.decrypt_file(f, passphrase=password)

print(result)
print("test")

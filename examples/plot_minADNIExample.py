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

if os.path.exists("testData.gpg"):
    print("File exists!")

gpg = gnupg.GPG()
with open("testData.gpg", "rb") as f:
    result = gpg.decrypt_file(f, passphrase="HeadPhoneChalkEraserMat")

type(result)
print(result)

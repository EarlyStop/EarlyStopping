"""
A minimal example for the ADNI application
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import EarlyStopping as es
import gnupg

sns.set_theme()


gpg = gnupg.GPG()
with open("testData.gpg", "rb") as f:
    result = gpg.decrypt_file(f, passphrase="Test")

print(result)

"""
A minimal example for the ADNI application
==========================================
"""

import os
import subprocess
import pandas as pd
from io import StringIO


password = os.environ["PASSWORD"]  # safer than hardcoding
print(password)

# Path to your encrypted file
encrypted_file = "testData.gpg"

if os.path.exists(encrypted_file):
    print("File exists!")
else:
    print("File not found!")

# 
# # Run GPG and capture the decrypted output
# result = subprocess.run(
#     ["gpg", "--batch", "--yes", "--passphrase", password, "--decrypt", encrypted_file],
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     check=True
# )
# 
# # Convert decrypted bytes to text
# decrypted_text = result.stdout.decode("utf-8")
# 
# # Load the CSV content into a DataFrame
# df = pd.read_csv(StringIO(decrypted_text))
# 
# print(df.head())

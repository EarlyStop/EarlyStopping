###################################################################################################
# ADNI example: data processing                                                                   #
###################################################################################################


#|%%--%%| <RTFTaisXbt|4mSeyVP6ks>
# Importing libraries -----------------------------------------------------------------------------
import numpy as np
import pandas as pd


#|%%--%%| <4mSeyVP6ks|AjXshz7rSK>
# Constructing a minimal ADNI I data sets ---------------------------------------------------------

# Importing data sets
cognitive_test_data = pd.read_csv("data/MMSE_26Mar2026.csv")
mri_data            = pd.read_csv("data/UCSFFSX7_26Mar2026.csv")

# Merge cognitive test score from 2nd data frame into 1st data frame only for rows where RID, VISCODE, and VISCODE2 match
merged_data = pd.merge(
    mri_data, 
    cognitive_test_data[['RID', 'VISCODE', 'VISCODE2', 'MMSCORE']],  # Only keep relevant columns from 2nd data frame
    on  = ['RID', 'VISCODE', 'VISCODE2'],                            # Merge on these columns
    how = 'inner'                                                    # Keep only rows that match
)
# merged_data.to_csv("data/merged_data.csv", index=False)

# Reduce data set to only ADNI I entries and cl
ADNI_I_data = merged_data[merged_data["PHASE"] == "ADNI1"]

# Clean up of the data set
ADNI_I_data = ADNI_I_data[ADNI_I_data["OVERALLQC"].isna()]     # Remove image records with failed quality control (There are none)
ADNI_I_data = ADNI_I_data.drop(columns = ["ST68SV", "ST8SV"])  # Drop features ST68SV and ST8SV for missing entries

# Write to file
ADNI_I_data.to_csv("data/ADNI_I_data.csv", index=False)

# Remove duplicate entries (Sometimes there are multiple imageids for the same entry)
# Remove all entries with field strength other than 1.5T (There are some with 3T, but we want to keep the data set as homogeneous as possible)

# Extract m06 data
ADNI_I_m06_data = ADNI_I_data[ADNI_I_data['VISCODE2'] == 'm06']
ADNI_I_m06_data = ADNI_I_m06_data[ADNI_I_m06_data["FIELD_STRENGTH"] == "1.5T"]
ADNI_I_m06_data = ADNI_I_m06_data.drop_duplicates(subset=["RID"], keep = "first") 

ADNI_I_m06_data.to_csv("data/ADNI_I_m06_data.csv", index=False)

# Extract m12 data
ADNI_I_m12_data = ADNI_I_data[ADNI_I_data['VISCODE2'] == 'm12']
ADNI_I_m12_data = ADNI_I_m12_data[ADNI_I_m12_data["FIELD_STRENGTH"] == "1.5T"]
ADNI_I_m12_data = ADNI_I_m12_data.drop_duplicates(subset=["RID"], keep = "first") 
ADNI_I_m12_data.to_csv("data/ADNI_I_m12_data.csv", index=False)

# Extract m18 data
ADNI_I_m18_data = ADNI_I_data[ADNI_I_data['VISCODE2'] == 'm18']
ADNI_I_m18_data = ADNI_I_m18_data[ADNI_I_m18_data["FIELD_STRENGTH"] == "1.5T"]
ADNI_I_m18_data = ADNI_I_m18_data.drop_duplicates(subset=["RID"], keep = "first") 
ADNI_I_m18_data.to_csv("data/ADNI_I_m18_data.csv", index=False)

# Extract m24 data
ADNI_I_m24_data = ADNI_I_data[ADNI_I_data['VISCODE2'] == 'm24']
ADNI_I_m24_data = ADNI_I_m24_data[ADNI_I_m24_data["FIELD_STRENGTH"] == "1.5T"]
ADNI_I_m24_data = ADNI_I_m24_data.drop_duplicates(subset=["RID"], keep = "first")
ADNI_I_m24_data.to_csv("data/ADNI_I_m24_data.csv", index=False)

# Extract m30 data
ADNI_I_m30_data = ADNI_I_data[ADNI_I_data['VISCODE2'] == 'm30']
ADNI_I_m30_data = ADNI_I_m30_data[ADNI_I_m30_data["FIELD_STRENGTH"] == "1.5T"]
ADNI_I_m30_data = ADNI_I_m30_data.drop_duplicates(subset=["RID"], keep = "first")
ADNI_I_m30_data.to_csv("data/ADNI_I_m30_data.csv", index=False)

# Extract m36 data
ADNI_I_m36_data = ADNI_I_data[ADNI_I_data['VISCODE2'] == 'm36']
ADNI_I_m36_data = ADNI_I_m36_data[ADNI_I_m36_data["FIELD_STRENGTH"] == "1.5T"]
ADNI_I_m36_data = ADNI_I_m36_data.drop_duplicates(subset=["RID"], keep = "first") 
ADNI_I_m36_data.to_csv("data/ADNI_I_m36_data.csv", index=False)
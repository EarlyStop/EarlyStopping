####################################################################################################
#         ADNI application                                                                         #
####################################################################################################


# Data download
https://ida.loni.usc.edu/explore/jsp/search_v2/search.jsp?project=ADNI
-> Download -> Search MMSE -> Mini-Mental State Examination (MMSE) [ADNI1,GO,2,3,4]
               Search UCSF -> UCSF - Cross-Sectional FreeSurfer (7.x) [ADNI1,GO,2,3,4]


# Feature data set: UCSF - Cross-Sectional FreeSurfer (7.x) [ADNI1,GO,2,3,4]
    RID: Roster ID identifies an individual (Same RIDs in two rows mean rows belong to the same individual).

    Field strength 3T vs. 1.5T: Some RIDS have both measurements with 3T and 1.5T. All subject seem to have the 1.5T measurement. Therefore, we reduce to this. 
    2026-03-26-BS-TODO: Check this not only for m06 and m36.

    There are more duplicates with different imageUIDs. Here  we just keep the first one.

    Visit codes:
    ADNI I:
    * sc: screening for eligibility (before baseline visit)
    *  f: Failed screening (Did not participate in study).
    * bl: baseline (first visit)

    
    ADNI II: 1st and 2nd visit codes not the same anymore.
    * scmri: MRI screening?



# Cognitive score data set: Mini-Mental State Examination (MMSE) [ADNI1,GO,2,3,4]


Diagnostic Summary data set: - Baseline Changes [ADNI1,GO,2,3,4]

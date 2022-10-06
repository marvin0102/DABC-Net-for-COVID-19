# Infer segmentation uncertainty through monte-carlo dropout
from DABC_uncertainty_colab import DABC_uncertainty
# run 5x inference
DABC_uncertainty('2020035365\\2020035365\\2020035365_0204_3050_20200204184413_4.nii.gz', '2020035365\\2020035365_output\\uncertainty', sample_value=5, uc_chosen='Both')


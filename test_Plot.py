# "Plot segmentation"
import pandas as pd
from utils.visualization import data_disease_slice, plot_segmentation, plot_progress_curve, plot_uncertainty
import warnings

warnings.filterwarnings("ignore")

# Severe patient
raw_severe, lung_severe, lesion_severe, ratio_severe = data_disease_slice(patientID='2020035365', slice_id=[175, 162, 195, 195, 195, 195, 195, 195])
print(raw_severe.shape)
meta_path='meta/2020035365.csv'
meta = pd.read_csv(meta_path, index_col=[0])
_meta_severe = meta[meta['slice'] > 100]
_meta_severe['ratio'] = ratio_severe

# # Mild patient
# # To avoid waiting, the segmentation results of mild patient have already run and saved in 2020035021_output/ folder.
# # If you want to reqeat the results, please change the patientID from 2020035365 to 2020035021 in inference steps.

raw_mild, lung_mild, lesion_mild, ratio_mild = data_disease_slice(patientID='2020035021', slice_id=[200, 200, 200, 200, 200, 200])

meta_path='meta/2020035021.csv'
meta = pd.read_csv(meta_path, index_col=[0])
_meta_mild = meta[meta['slice'] > 100]
_meta_mild['ratio'] = ratio_mild

plot_segmentation(raw_severe, lung_severe, lesion_severe, color_map='Reds', state='Severe', hspace=-0.6)

plot_segmentation(raw_mild, lung_mild, lesion_mild, color_map='Reds', state='Mild', hspace=-0.4)

# "Plot progress curve"
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
plot_progress_curve(_meta_severe, patientID=2020035365, line_color=sns.color_palette('Reds')[5], label='Severe patient')
plot_progress_curve(_meta_mild, patientID=2020035021, line_color=sns.color_palette('Greens')[3], label='Mild patient')
plt.legend(loc='upper right')
plt.title('Severe pateint vs Mild pateint', fontsize=26)
plt.xlabel('Time(Day)', fontsize=16)
plt.ylabel('Lesion ratio', fontsize=16)
plt.show()

# Example I:
plot_uncertainty(name_id='2020035365_0204_3050_20200204184413_4.nii.gz',slice_id=175)
# Example II:
plot_uncertainty(name_id='2020035365_0204_3050_20200204184413_4.nii.gz',slice_id=150)
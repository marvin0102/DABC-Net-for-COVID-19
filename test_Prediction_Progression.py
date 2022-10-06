import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.calculate_feature import *
from utils.visualization import *
from pipeline.data_pipeline import read_from_nii

# severe patient
meta_path_severe='meta/2020035365.csv'
meta_severe = pd.read_csv(meta_path_severe, index_col=[0])

raw_data_severe = read_from_nii('2020035365/*').astype('float32')
lung_severe = read_from_nii(r'2020035365_output/lung/*').astype('float32')
lesion_severe = read_from_nii(r'2020035365_output/covid/*').astype('float32')

# mild patient
meta_path_mild='meta/2020035021.csv'
meta_mild = pd.read_csv(meta_path_mild, index_col=[0])

def predict_base_learners(base_learners, x):
    P = np.zeros((x.shape[0], len(base_learners)))
    print('Generating base learner predictions.')
    for i, (name, m) in enumerate(base_learners.items()):
        print('%s...'% name, end='', flush=False)
        p = m.predict_proba(x)
        P[:, i] = p[:, 1]
    print('done.')
    return P

res_list_severe, all_info_severe = calculate(raw_data_severe, lung_severe, lesion_severe, meta_severe)

del raw_data_severe, lung_severe, lesion_severe  # release RAM

# mild patient
raw_data_mild = read_from_nii(r'2020035021/*').astype('float32')
lung_mild = read_from_nii(r'2020035021_output/lung/*').astype('float32')
lesion_mild = read_from_nii(r'2020035021_output/covid/*').astype('float32')

res_list_mild, all_info_mild = calculate(raw_data_mild, lung_mild, lesion_mild, meta_mild)
del raw_data_mild, lung_mild, lesion_mild

from utils.visualization import plot_fetures

plot_fetures(all_info_severe=all_info_severe, all_info_mild=all_info_mild, save_to_html=True)  # x-axis:time(day), y-axis: lesion ratio

from utils.visualization import plot_animation_curve

plot_animation_curve(all_info=all_info_severe)  # x-axis:time(day), y-axis: lesion ratio

import pickle
import json

with open('model/prediction.pkl', 'rb') as j:
    base_pred = pickle.load(j)

with open('model/min_max_prediction.json', 'r') as j:
    min_max_dict_pred = json.load(j)

with open('model/classification.pkl', 'rb') as j:
    base_cls = pickle.load(j)

with open('model/min_max_classification.json', 'r') as j:
    min_max_dict_cls = json.load(j)

feature = [
    'left_ratio', 'right_ratio',
    'left_lung', 'right_lung',
    'left_lesion', 'right_lesion',

    'left_weighted_lesion', 'right_weighted_lesion',

    'left_consolidation', 'right_consolidation',

    'left_z', 'right_z',
    'Age', 'sex',
]

X_severe = preprocessing(all_info_severe, feature)
X_mild = preprocessing(all_info_mild, feature)

def First_Two_Scans(X):
  # first two scan
  x_list = X.iloc[1].tolist()[:-2] + X.iloc[0].tolist()
  # min max scale
  x = min_max_scalar(np.array(x_list), np.array(min_max_dict_pred['min']), np.array(min_max_dict_pred['max']))

  # Predition
  P_pred = predict_base_learners(base_pred, np.array([x]))
  return P_pred.mean()

print('\n'+'*'*10+'\tSevere patient\t'+'*'*10)
print(First_Two_Scans(X_severe))
print('\n'+'*'*10+'\tMild patient\t'+'*'*10)
print(First_Two_Scans(X_mild))

with open('model/prediction_first_3.pkl', 'rb') as j:
    base_pred = pickle.load(j)

with open('model/min_max_prediction_first_3.json', 'r') as j:
    min_max_dict_pred = json.load(j)


def First_Three_Scans(X):
  # first two scan
  x_list = X.iloc[2].tolist()[:-2] + X.iloc[1].tolist()
  # min max scale
  x = min_max_scalar(np.array(x_list), np.array(min_max_dict_pred['min']), np.array(min_max_dict_pred['max']))

  # Predition
  P_pred = predict_base_learners(base_pred, np.array([x]))
  return P_pred.mean()

print('\n'+'*'*10+'\tSevere patient\t'+'*'*10)
print(First_Three_Scans(X_severe))
print('\n'+'*'*10+'\tMild patient\t'+'*'*10)
print(First_Three_Scans(X_mild))

from copy import deepcopy
with open('model/prediction_first.pkl', 'rb') as j:
    base_pred = pickle.load(j)

with open('model/min_max_prediction_first.json', 'r') as j:
    min_max_dict_pred = json.load(j)


def First_Scan(X):
  # first two scan
  x_list = X.iloc[0].tolist()
  # min max scale
  x = min_max_scalar(np.array(x_list), np.array(min_max_dict_pred['min']), np.array(min_max_dict_pred['max']))

  # Predition
  P_pred = predict_base_learners(base_pred, np.array([x]))
  return P_pred.mean()

print('\n'+'*'*10+'\tSevere patient\t'+'*'*10)
print(First_Scan(deepcopy(X_severe)))
print('\n'+'*'*10+'\tMild patient\t'+'*'*10)
print(First_Scan(deepcopy(X_mild)))

def Per_Scan_Classification(X):
  x = min_max_scalar(np.array(X), np.array(min_max_dict_cls['min']), np.array(min_max_dict_cls['max']))
  P_pred = predict_base_learners(base_cls, np.array(x))
  p = P_pred.mean(axis=1)
  return p

p_severe = Per_Scan_Classification(X_severe)
print('')
print('Prediction of severe patient(per scan):\n{}\n'.format(p_severe))
print('')
p_mild = Per_Scan_Classification(X_mild)
print('')
print('Prediction of mild patient(per scan):\n{}\n'.format(p_mild))

print('\n'+'*'*10+'\tSevere patient\t'+'*'*10)
print('pred\t{} \ngt\t{} \nprob {}'.format((p_severe > 0.5).astype('int'), np.array(all_info_severe['Severe']), p_severe))
print('\n'+'*'*10+'\tMild patient\t'+'*'*10)
print('pred\t{} \ngt\t{} \nprob {}'.format((p_mild > 0.5).astype('int'), np.array(all_info_mild['Severe']), p_mild))

slice_id = [175, 162, 195, 195, 195, 195, 195, 195]
raw, lesion, gt = data_disease_progress_slice(all_info_severe, patientID=2020035365, slice_id=slice_id, timepoint_count=8)
plot_progress(raw, lesion, p_severe, gt, state='severe', color_map='Reds', timepoint_count=8)

print('\n\n')
slice_id = [200, 200, 200, 200, 200, 200]
raw, lesion, gt = data_disease_progress_slice(all_info_mild, patientID=2020035021, slice_id=slice_id, timepoint_count=6)
plot_progress(raw, lesion, p_mild, gt, state='mild', color_map='Reds', timepoint_count=6)


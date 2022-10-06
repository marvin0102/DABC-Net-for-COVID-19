import numpy as np
import tensorflow as tf
from models import models as Model
from pipeline.inference_pipeline import local_inference
from pipeline.data_pipeline import save_pred_to_nii, read_from_nii, confirm_data
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def DABC_infer(nii_path='', save_path='', usage='covid', sform_code=1):
    save_path = save_path + '/*'
    nii_path = nii_path + '/*'
    all_src_data = read_from_nii(nii_path=nii_path, Hu_window=(-1024, 512), need_rotate=True)
    all_src_data = np.expand_dims(all_src_data, -1)

    print('\n**********\tInferring CT scans:\t**********\n')
    test_vol = confirm_data(all_src_data)
    '''
    infer
    '''
    if usage == 'covid':
        name = 'weight/Covid_05112327'
    elif usage == 'lung':
        name = 'weight/model_05090017'
    else:
        print('Please select correct model!')
        return None
    model = Model.DABC(input_size=(4, 256, 256, 1), load_weighted=name)
    pred = local_inference(test_vol, model)
    save_pred_to_nii(pred=pred, save_path=save_path.replace('*', ''), ref_path=nii_path,
                     need_resize=True, need_rotate=sform_code)


input_path='2020035021'
output_path='2020035021_output\\covid'
DABC_infer(input_path, output_path)
output_path='2020035021_output\\lung'
DABC_infer(input_path, output_path, usage='lung')
print("segmentation finish!\n")





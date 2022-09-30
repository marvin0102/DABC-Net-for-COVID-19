from cmd import IDENTCHARS
from models import models as Model
import numpy as np
import tensorflow as tf
from skimage.io import imread, imshow
from skimage.transform import resize
from tqdm import tqdm
import os

IMG_SLICE = 255
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 1

TRAIN_PATH = 'E:\\TCIA_Lung_SEG\\SEG2'
train_ids = next(os.walk(TRAIN_PATH.join('\\image')))[1]
train_ids = [ x.split('_')[0] for x in train_ids]
print(len(train_ids))


X_train = np.zeros((len(train_ids), IMG_SLICE, IMG_HEIGHT, IMG_WIDTH), dtype = np.unit8)
Y_train = np.zeros((len(train_ids), IMG_SLICE, IMG_HEIGHT, IMG_WIDTH), dtype = np.bool)

print("Resize training images and masks")
for n, id_ in tqdm(enumerate(train_ids), total = len(train_ids)):
    path = TRAIN_PATH + '\\image\\' + id_
    img = imread(path + 'ct_image.nii')[:,:,:]
    img = resize(img, (IMG_SLICE, IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
    X_train[n] = img # fill empty X_train with values form img
    mask_path = TRAIN_PATH + '\\mask\\' + id_
    mask = np.zeros((IMG_SLICE, IMG_HEIGHT, IMG_WIDTH), dtype = np.bool)
    for mask_file in next(os.walk(path))[2]:
        mask_ = imread(path + '\\masks\\' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', 
                                    preserve_range = True), axis = -1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask



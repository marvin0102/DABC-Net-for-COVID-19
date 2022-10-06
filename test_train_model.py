from cmd import IDENTCHARS
from gc import callbacks
from models import models as Model
import numpy as np
import matplotlib.pyplot as plt 
import random
import tensorflow as tf
from skimage.io import imread, imshow
from skimage.transform import resize
from tqdm import tqdm
import os

HOUSEFIELD_MIN = -3000
HOUSEFIELD_MAX = 3000
HOUSEFIELD_RANGE = HOUSEFIELD_MAX - HOUSEFIELD_MIN

def confirm_data(test_vol):
    if test_vol.shape[0] % 4 != 0:
        cut = test_vol.shape[0] % 4
        test_vol = test_vol[:-cut]
    assert test_vol.shape[0] % 4 == 0
    return test_vol

def get_infer_data(te_data2):
    te_data3 = []
    tag = 0
    for i in range(int(te_data2.shape[0] / 4)):
        te_data3.append(te_data2[tag:tag + 4])
        tag += 4

    te_data4 = np.array(te_data3)
    return te_data4

def normalizeImageIntensityRange(img):
    img[img < HOUSEFIELD_MIN] = HOUSEFIELD_MIN
    img[img > HOUSEFIELD_MAX] = HOUSEFIELD_MAX
    return (img - HOUSEFIELD_MIN)/HOUSEFIELD_RANGE

    

def  get_train_data(TRAIN_PATH = 'E:\\TCIA_Lung_SEG\\SEG2', IMG_SLICE = 4, IMG_HEIGHT = 512, IMG_WIDTH = 512):
    IMG_CHANNELS = 1

    train_ids = next(os.walk(os.path.join(TRAIN_PATH, 'image')))[1]
    train_ids = [ x.split('_')[0] for x in train_ids]

    X_train = np.empty(( 0, IMG_SLICE, IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.uint8)
    Y_train = np.empty(( 0, IMG_SLICE, IMG_HEIGHT, IMG_WIDTH, 1), dtype = bool)

    print("Resize training images and masks")
    for n, id_ in tqdm(enumerate(train_ids), total = len(train_ids)):
        path = TRAIN_PATH + '\\image\\' + id_ + '_Reconstruction'
        # read Nii shape = (slice, 512, 512)
        if os.path.isfile(path + '\\ct_image.nii'):
            img = imread(path + '\\ct_image.nii')
        else:
            continue
        # expend dim into (slice, height, width, 1)
        img = np.expand_dims( img, axis = -1)
        test_vol1 = confirm_data(img)
        # fill empty X_train with values form img
        test_data1 = get_infer_data(test_vol1)
        test_data1 = normalizeImageIntensityRange(test_data1)
        X_train = np.append(X_train, test_data1, axis = 0) 

        # process mask
        mask_path = TRAIN_PATH + '\\mask\\' + id_ + '_SEG'
        mask_ids = next(os.walk(mask_path))[1]
        # if mask dir is empty
        if not mask_ids :
            continue
        mask = np.zeros((img.shape[0], IMG_HEIGHT, IMG_WIDTH, 1), dtype = bool)
        for i in range(img.shape[0], 0, -1):
            # each slice's mask
            mask_id_ = ("1-%03d" % i)
            if not os.path.isdir(os.path.join(mask_path, mask_id_)):
                continue
            for mask_file in next(os.walk(os.path.join(mask_path, mask_id_)))[2]:
                
                mask_ = imread(os.path.join(mask_path, mask_id_, mask_file))
                mask_ = np.squeeze(mask_, axis = 0)
                # expand dim into (height, width, 1)
                mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', 
                                            preserve_range = True), axis = -1)
                mask[i] = np.maximum(mask[i], mask_)

        test_vol2 = confirm_data(mask)
        # fill empty X_train with values form img
        test_data2 = get_infer_data(test_vol2)
        Y_train = np.append(Y_train, test_data2, axis = 0) 
    print("Done!")
    return X_train, Y_train

def DABC_infer(trainingData_path='', save_path=''):

    print('\n**********\tInferring CT scans:\t**********\n')
    X_train, Y_train = get_train_data(trainingData_path, 4, 512, 512)
    print(X_train.shape, Y_train.shape)


    model = Model.DABC(input_size=(4, 512, 512, 1))
    model.summary()

    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_DABC.h5', verbose = 1, save_best_only = True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience = 3, monitor='val_loss'), 
        tf.keras.callbacks.TensorBoard(log_dir='logs'), 
        checkpointer
    ]

    result = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks = callbacks)

    def show_train_history(train_type,test_type):
        plt.plot(result.history[train_type])
        plt.plot(result.history[test_type])
        plt.title('Train History')
        if train_type == 'accuracy':
            plt.ylabel('Accuracy')
        else:
            plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    show_train_history('accuracy', 'val_accuracy')
    show_train_history('loss', 'val_loss')




DABC_infer('E:\\TCIA_Lung_SEG\\SEG2' )

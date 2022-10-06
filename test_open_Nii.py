import SimpleITK as sitk
import os
import numpy as np
import random
from pydicom import dcmread
from matplotlib import pyplot as plt
import matplotlib


def showNii(img):
    for i in range(img.shape[0]):
        print(i)
        plt.imshow(img[i, :, :], cmap='gray')
        plt.show()

def transp_imshow(data, tvmin=None, tvmax=None, tmax=1.,
                  gam=1., cmap='Blues', **kwargs):
    """
    Displays the 2d array `data` with pixel-dependent transparency.
    Parameters
    ----------
    data: 2d numpy array of floats or ints
        Contains the data to be plotted as a 2d map
    tvmin, tvmax: floats or None, optional
        The values (for the elements of `data`) that will be plotted
        with minimum opacity and maximum opacity, respectively.
        If no value is provided, this uses by default the arguments
        `vmin` and `vmax` of `imshow`, or the min and max of `data`.
    tmax: float, optional
        Value between 0 and 1. Maximum opacity, which is reached
        for pixel that have a value greater or equal to `tvmax`.
        Default: 1.
    gam: float, optional
        Distortion of the opacity with pixel-value.
        For `gam` = 1, the opacity varies linearly with pixel-value
        For `gam` < 1, low values have higher-than-linear opacity
        For `gam` > 1, low values have lower-than-linear opacity
    cmap: a string or a maplotlib.colors.Colormap object
        Colormap to be used
    kwargs: dict
        Optional arguments, which are passed to matplotlib's `imshow`.
    """
    # Determine the values between which the transparency will be scaled
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    else:
        vmax = data.max()
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    else:
        vmin = data.min()
    if tvmax is None:
        tvmax = vmax
    if tvmin is None:
        tvmin = vmin

    # Rescale the data to get the transparency and color
    color = (data - vmin) / (vmax - vmin)
    color[color > 1.] = 1.
    color[color < 0.] = 0.
    transparency = tmax * (data - tvmin) / (tvmax - tvmin)
    transparency[transparency > 1.] = 1
    transparency[transparency < 0.] = 0.
    # Application of a gamma distortion
    transparency = tmax * transparency ** gam

    # Get the colormap
    if isinstance(cmap, matplotlib.colors.Colormap):
        colormap = cmap
    elif type(cmap) == str:
        colormap = getattr(plt.cm, cmap)
    else:
        raise ValueError('Invalid type for argument `cmap`.')

    # Create an rgba stack of the data, using the colormap
    rgba_data = colormap(color)
    # Modify the transparency
    rgba_data[:, :, 3] = transparency

    sc = plt.imshow(rgba_data, **kwargs)

    # test
    # plt.colorbar(sc)
    return sc


IMG_HEIGHT = 512
IMG_WIDTH = 512
# itk_img = sitk.ReadImage('C:\\Users\\marvi\\Downloads\\2020035021\\2020035021\\2020035021_1_3050_20200129200802_4.nii.gz')
itk_img = sitk.ReadImage('E:\\MADRC_1a\\manifest-1608266677008\\SEG2\\1.2.826.0.1.3680043.10.474.419639.403650391453800318566197197100_Reconstruction\\ct_image.nii')
# itk_img = sitk.ReadImage('E:\\MADRC_1a\\manifest-1608266677008\\SEG2\\1.2.826.0.1.3680043.10.474.419639.403650391453800318566197197100_SEG\\U_R5KzRO-002.nii')
itk_mask_path = 'E:\\MADRC_1a\\manifest-1608266677008\\SEG2\\1.2.826.0.1.3680043.10.474.419639.403650391453800318566197197100_SEG\\'

img = sitk.GetArrayFromImage(itk_img)
print(img.shape)  # (88, 132, 175) indicates the number of slices in each dimension
print(img.max(), img.min())
img_number = random.randint(22, 210)

i = img_number
print(i)
ax = plt.subplot(131)
ax.imshow(img[(img.shape[0] - 1) - i, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])

random_mask_path = itk_mask_path + '1-' + ('%03d' % i)
itk_mask_files = os.listdir(random_mask_path)

mask = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=bool)
for file in itk_mask_files:
    itk_img_mask = sitk.ReadImage(os.path.join(random_mask_path,file))
    img_mask = sitk.GetArrayFromImage(itk_img_mask)
    print(img_mask.shape)
    mask = np.maximum(mask, img_mask)    
ax = plt.subplot(132)
ax.imshow(mask[0, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(img[(img.shape[0] - 1) - i, :, :], cmap='gray')
transp_imshow(mask[0, :, :], cmap='Reds', alpha=0.7)
# plt.title('No.{} scan lesion\n'.format(i + 1), fontsize=16)
plt.xticks([]), plt.yticks([])
plt.show()


# dcm_path = 'E:\\manifest-1608266677008\\MIDRC-RICORD-1A\\MIDRC-RICORD-1A-419639-000082\\1.2.826.0.1.3680043.10.474.419639.312580455409613733097488204614\\1.2.826.0.1.3680043.10.474.419639.403650391453800318566197197100'
# dcm_files = os.listdir(dcm_path)
# print(len(dcm_files))
# for idx, file in enumerate(dcm_files):
#     if idx%4 == 0:
#         ds = dcmread(os.path.join(dcm_path,file))
#         np_dcm = ds.pixel_array
#         print(idx)
#         fig=plt.figure()
#         ax = fig.add_subplot(121)
#         ax.imshow(np_dcm, cmap='gray')
#         ax = fig.add_subplot(122)
#         ax.imshow(img[(img.shape[0]-1)-idx, :, :], cmap='gray')
#         plt.show()
# file = dcm_files[i]
# ds = dcmread(os.path.join(dcm_path,file))
# np_dcm = ds.pixel_array
# fig=plt.figure()
# ax = fig.add_subplot(121)
# ax.imshow(np_dcm, cmap='gray')



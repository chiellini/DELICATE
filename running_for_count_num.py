from stardist import random_label_cmap
import logging
import os
# from stardist import Rays_GoldenSpiral
# from stardist.matching import matching, matching_dataset
from stardist.models import StarDist3D
from csbdeep.utils import normalize
from glob import glob
import nibabel as nib
import numpy as np
from tifffile import imread  # or from skimage.io import imread

from Utils.preprocess_lib import stack_nuc_slices


raw_image_dir=r'D:\tem'
embryo_name='200109plc1p1'
target_time_point=10
out_size=[256,356,214] # 214 for z 92 224 for z 94, 160 for z 68
delicate_model_root=r'D:\NucApp-develop\static\models'
delicate_model_name=r'DELICATE_ZhaoZYlabDeconv'

stardist_model = StarDist3D(None, name=delicate_model_name, basedir=delicate_model_root)
logging.debug('finish loaded model')


origin_files = sorted(glob(os.path.join(raw_image_dir, embryo_name, "tif", "{}_L1-t{}*.tif".format(embryo_name,str(target_time_point).zfill(3)))))

tem_3d_middle_folder = os.path.join(raw_image_dir, 'tem_middle',embryo_name, "RawNuc")
stack_nuc_slices([origin_files, tem_3d_middle_folder, embryo_name, target_time_point, out_size])


running_img_path = sorted(glob(os.path.join(tem_3d_middle_folder, "{}_{}_rawNuc.tif".format(embryo_name, str(target_time_point).zfill(3)))))

arr = imread(running_img_path)  # running_img_path is a string path
arr = np.asarray(arr).astype(np.float32, copy=False)

# 2) pick normalization axes robustly
#    - If there is a small "channel" dim at the end (<=4), exclude it from axes.
#    - Otherwise, normalize over all existing axes.
nd = arr.ndim
if nd == 0:
    raise ValueError(f"Loaded data from '{running_img_path}' is scalar; check the path/file.")
if nd == 1:
    axis_norm = (0,)  # 1D data; avoid axis 1
else:
    if arr.shape[-1] <= 4:   # treat last dim as channels if small
        axis_norm = tuple(range(nd-1))   # all but channel
    else:
        axis_norm = tuple(range(nd))     # all dims

# 3) normalize
img = normalize(arr, 1, 99.8, axis=axis_norm)

# axis_norm = (0, 1, 2)
# img = normalize(running_img_path, 1, 99.8, axis=axis_norm)
labels, details = stardist_model.predict_instances(img)
# Assuming testset and testset_copy have the same order and length
# embryo_tp = '_'.join(embryo_name,str(target_time_point).zfill(3))
pred_seg = labels.transpose([1, 2, 0])
print('!!!!!!!!!!!!!!!!!!!!There are totally:  ', len(np.unique(pred_seg)-1), ' cells..')
binary_pred = np.where(pred_seg > 0, 255, pred_seg)
segNuc_dir_tem=os.path.join(raw_image_dir, 'tem_middle',embryo_name, "SegNuc")
os.makedirs(segNuc_dir_tem, exist_ok=True)
nib.save(nib.Nifti1Image(binary_pred, np.eye(4)),
            os.path.join(segNuc_dir_tem, embryo_name+str(target_time_point).zfill(3) + '_predNuc.nii.gz'))

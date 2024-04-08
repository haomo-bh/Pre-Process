import SimpleITK as sitk
import numpy as np
import os 
import glob

def dice_coefficient(y_true, y_pred, nums, smooth=1e-5): 
    dsc = 0 
    for i in range(1, nums):
        y_true_f = y_true.flatten() == i  
        y_pred_f = y_pred.flatten() == i  
        intersection = np.sum(y_true_f * y_pred_f)  
        dice_i = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth) 
        dsc += dice_i
    return  dsc / (nums-1)
  

# seg_path = r'D:/Transmorph_Frame/Data/training_validation_datas_2/train_seg_nii/'
# ct_tumors = glob.glob(seg_path + '*CT*.nii.gz')
# mr_tumors = glob.glob(seg_path + '*MR*_resampled_cropped_warped.nii.gz')
# mr_tumors_2 = glob.glob(seg_path + '*MR*_resampled_cropped.nii.gz')
seg_path = r'D:/Transmorph_Frame/Data/training_validation_datas_5/validation_seg_nii/'
ct_tumors = glob.glob(seg_path + '*CT*.nii.gz')
mr_tumors = glob.glob(seg_path + '*MR*_cropped_warped.nii.gz')
mr_tumors_2 = glob.glob(seg_path + '*MR*_cropped.nii.gz')
dice_list = []
i = 0
for ct_path, mr_path, mr_path_2 in zip(ct_tumors, mr_tumors, mr_tumors_2):
    ct_tumor, mr_tumor, mr_tumor_2 = sitk.ReadImage(ct_path), sitk.ReadImage(mr_path), sitk.ReadImage(mr_path_2)
    ct_array, mr_array, mr_array_2 = sitk.GetArrayFromImage(ct_tumor), sitk.GetArrayFromImage(mr_tumor), sitk.GetArrayFromImage(mr_tumor_2)
    # if dice_coefficient(ct_array, mr_array, 2) <= dice_coefficient(ct_array, mr_array_2, 2):
    if dice_coefficient(ct_array, mr_array, 2) <= 0.2:
        i += 1
        print(ct_path)
    dice_list.append(dice_coefficient(ct_array, mr_array_2, 2))

dice_list = np.array(dice_list)
print(np.mean(dice_list))
print(i)
# seg_path = r"G:/PythonProject/TransMorph/pretrain/Data/SyN_tumor/validation/"
# seg_files = os.listdir(seg_path)
# file_len = len(seg_files)
# for i in range(0, file_len, 3):
#     seg_true = sitk.ReadImage(seg_path + seg_files[i])
#     seg_true = sitk.GetArrayFromImage(seg_true)
#     sum1 = np.sum(seg_true)
#     seg_orgin = sitk.ReadImage(seg_path + seg_files[i+1])
#     seg_orgin = sitk.GetArrayFromImage(seg_orgin)
#     sum2 = np.sum(seg_orgin)
#     seg_pred = sitk.ReadImage(seg_path + seg_files[i+2])
#     seg_pred = sitk.GetArrayFromImage(seg_pred)
#     sum = sum1 + sum2
#     print(f"第{i//3}组未经配准的dice系数为:{dice_coefficient(seg_true, seg_orgin)}")
#     print(f"第{i//3}组经配准的dice系数为:{dice_coefficient(seg_true, seg_pred)}")
#     print(f'第{i//3}组数据的理想dice系数为：{min(sum1, sum2) * 2/(sum1 + sum2)}')

import SimpleITK as sitk
import numpy as np
import os 
import glob


def images_standard(path):
    files = os.listdir(path)
    nums = len(files)
    CT_list, MR_list = [], []
    for file in files:
        if os.path.isfile(path + file):
            image = sitk.ReadImage(path + file)
            image_array = sitk.GetArrayFromImage(image)
            if 'CT' in file:
                CT_list.append(image_array)
            elif 'MR_warped' in file:
                MR_list.append(image_array)
    
    CT_array, MR_array = np.array(CT_list), np.array(MR_list)
    CT_mean, CT_std = np.mean(CT_array), np.std(CT_array)
    MR_mean, MR_std = np.mean(MR_array), np.std(MR_array)
    CT_array, MR_array = (CT_array - CT_mean) / CT_std, (MR_array - MR_mean) / MR_std
    i = 0
    for file in files:
        if os.path.isfile(path + file):
            image = sitk.ReadImage(path + file)
            if 'CT' in file:
                CT_file = CT_array[i//2]
                CT_stand = sitk.GetImageFromArray(CT_file)
                CT_stand.SetOrigin(image.GetOrigin())
                CT_stand.SetDirection(image.GetDirection())
                CT_stand.SetSpacing(image.GetSpacing())
                sitk.WriteImage(CT_stand, path + file)
                i = i + 1
            elif 'MR_warped' in file:
                MR_file = MR_array[i//2]
                MR_stand = sitk.GetImageFromArray(MR_file)
                MR_stand.SetOrigin(image.GetOrigin())
                MR_stand.SetDirection(image.GetDirection())
                MR_stand.SetSpacing(image.GetSpacing())
                sitk.WriteImage(MR_stand, path + file)
                i = i + 1


if __name__ == '__main__':
    nii_path = r"G:/PythonProject/TransMorph/pretrain/Data/zhejiangdaxue/validation/"
    images_standard(nii_path)
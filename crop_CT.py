import SimpleITK as sitk
import numpy as np
import glob

# img_path = r"D:/Transmorph_Frame/Data/raw_images/new_nii_files/new_nii_resampled_files/"
# tumor_path = r"D:/Transmorph_Frame/Data/raw_tumors_seg/new_nii_files/new_nii_resampled_files/"
img_path = r'D:/Transmorph_Frame/Data/seg_need_nii/raw_images/'
tumor_path = r'D:/Transmorph_Frame/Data/seg_need_nii/raw_tumors_seg/'
img_path = glob.glob(img_path + 'PLA-2026MR*')
tumor_path = glob.glob(tumor_path + 'PLA-2026MR*')
start_z = 11
end_z = 109
img = sitk.ReadImage(*img_path)
tumor = sitk.ReadImage(*tumor_path)
img_new = sitk.GetArrayFromImage(img)
tumor_new = sitk.GetArrayFromImage(tumor)
z_len, _, _ = img_new.shape
img_new = img_new[start_z:end_z, :, :]
tumor_new = tumor_new[start_z:end_z, :, :]
img_new = sitk.GetImageFromArray(img_new)
img_new.SetSpacing(img.GetSpacing())
tumor_new = sitk.GetImageFromArray(tumor_new)
tumor_new.SetSpacing(tumor.GetSpacing())
sitk.WriteImage(img_new, *img_path)
sitk.WriteImage(tumor_new, *tumor_path)

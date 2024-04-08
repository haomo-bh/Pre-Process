import SimpleITK as sitk
import numpy as np
import os
import glob


def nrrd2nii(path):
    try:
        files = os.listdir(path)
        os.makedirs(path + 'new_nii_files', exist_ok=True)
        for file in files:
            if file.endswith('.nrrd'):
                image = sitk.ReadImage(path + file)
                sitk.WriteImage(image, path + 'new_nii_files/' + file.replace(".nrrd", ".nii.gz"))
    except:
        print('The path did not contains any available files!!!')


def image_resample(path, new_spacing=[1.0, 1.0, 1.0], is_seg=False):
    try:
        files = os.listdir(path)
        os.makedirs(path + 'new_nii_resampled_files', exist_ok=True)
        for file in files:
            if os.path.isfile(path + file):
                image = sitk.ReadImage(path + file)
                original_size = image.GetSize()
                original_spacing = image.GetSpacing()
                transform = sitk.Transform()
                transform.SetIdentity()
                new_size = [int(old_space * old_size / new_space) for old_size, old_space, new_space in
                            zip(original_size, original_spacing, new_spacing)]
                resample = sitk.ResampleImageFilter()
                resample.SetTransform(transform)
                resample.SetSize(new_size)
                resample.SetOutputOrigin(image.GetOrigin())
                resample.SetOutputSpacing(new_spacing)
                resample.SetOutputDirection(image.GetDirection())
                if is_seg:
                    resample_method = sitk.sitkNearestNeighbor
                    resample_pixel_type = sitk.sitkUInt8
                else:
                    resample_method = sitk.sitkLinear
                    resample_pixel_type = sitk.sitkFloat32
                resample.SetInterpolator(resample_method)
                resample.SetOutputPixelType(resample_pixel_type)
                resampled_image = resample.Execute(image)
                sitk.WriteImage(resampled_image, path + 'new_nii_resampled_files/'
                                + file.replace(".nii.gz","_resampled.nii.gz"))
    except:
        print('The path did not contains any available files!!!')


def image_norm(path, ct_min, ct_max):
    try:
        files = os.listdir(path)
        os.makedirs(path + 'new_nii_resampled_cropped_normalized_files', exist_ok=True)
        for file in files:
            if os.path.isfile(path + file):
                image = sitk.ReadImage(path + file)
                image_array = sitk.GetArrayFromImage(image)
                # 对CT进行裁剪操作
                if 'CT' in file:
                    image_array = np.clip(image_array, ct_min, ct_max)
                image_mean = np.mean(image_array)
                image_std = np.std(image_array)
                image_array = (image_array - image_mean) / image_std
                image_normalized = sitk.GetImageFromArray(image_array)
                image_normalized.SetOrigin(image.GetOrigin())
                image_normalized.SetDirection(image.GetDirection())
                image_normalized.SetSpacing(image.GetSpacing())
                sitk.WriteImage(image_normalized, path +
                                'new_nii_resampled_cropped_normalized_files/'
                                + file.replace("_cropped.nii.gz", "_cropped_normalized.nii.gz"))
    except:
        print('The path did not contains any available files!!!')

def image_norm_percentile(path, ct_min=0, ct_max=2000):
    try:
        files = os.listdir(path)
        os.makedirs(path + 'new_nii_resampled_normalized_files', exist_ok=True)
        for file in files:
            if os.path.isfile(path + file):
                image = sitk.ReadImage(path + file)
                image_array = sitk.GetArrayFromImage(image)
                # 对CT进行裁剪操作
                if 'CT' in file:
                    image_array = np.clip(image_array, ct_min, ct_max)
                elif 'MR' in file:
                    MR_max = np.percentile(image_array, 99)
                    image_array = np.clip(image_array, 0, MR_max)
                image_max = np.max(image_array)
                image_min = np.min(image_array)
                image_array = (image_array - image_min) / (image_max - image_min)
                image_normalized = sitk.GetImageFromArray(image_array)
                image_normalized.SetOrigin(image.GetOrigin())
                image_normalized.SetDirection(image.GetDirection())
                image_normalized.SetSpacing(image.GetSpacing())
                sitk.WriteImage(image_normalized, path +
                                'new_nii_resampled_normalized_files/'
                                + file.replace(".nii.gz", "_normalized.nii.gz"))
    except:
        print('The path did not contains any available files!!!')

def image_crop_pad(path, bias, new_size=[112, 192, 224]):
    try:
        files = os.listdir(path)
        # 防止数组越界
        z_bias = abs(bias[0])
        x_bias = abs(bias[1])
        y_bias = abs(bias[2])
        new_size[0] += 2 * z_bias
        new_size[1] += 2 * x_bias
        new_size[2] += 2 * y_bias
        os.makedirs(path + 'new_nii_resampled_normalized_cropped_files', exist_ok=True)
        for file in files:
            if os.path.isfile(path + file):
                # z方向进行一定的偏移
                image = sitk.ReadImage(path + file)
                image_array = sitk.GetArrayFromImage(image)
                original_shape = image_array.shape
                # original_center = [int(size / 2) for size in image_array.shape]
                center = [int(size / 2) for size in new_size]
                center[0] -= z_bias
                center[1] -= x_bias
                center[2] -= y_bias
                if original_shape[0] < new_size[0]:
                    pad_size = int((new_size[0] - original_shape[0]) / 2)
                    image_array = np.pad(image_array, pad_width=((pad_size + 1, pad_size + 1), (0, 0), (0, 0)))
                if original_shape[1] < new_size[1]:
                    pad_size = int((new_size[1] - original_shape[1]) / 2)
                    image_array = np.pad(image_array, pad_width=((0, 0), (pad_size + 1, pad_size + 1), (0, 0)))
                if original_shape[2] < new_size[2]:
                    pad_size = int((new_size[2] - original_shape[2]) / 2)
                    image_array = np.pad(image_array, pad_width=((0, 0), (0, 0), (pad_size + 1, pad_size + 1)))
                new_center = [int(size / 2) for size in image_array.shape]
                new_center[0] += bias[0]
                new_center[1] += bias[1]
                new_center[2] += bias[2]
                image_array_new = image_array[new_center[0] - center[0]:new_center[0] + center[0],
                                    new_center[1] - center[1]:new_center[1] + center[1],
                                    new_center[2] - center[2]:new_center[2] + center[2]]
                print(image_array_new.shape)
                image_cropped = sitk.GetImageFromArray(image_array_new)
                image_cropped.SetOrigin(image.GetOrigin())
                image_cropped.SetDirection(image.GetDirection())
                image_cropped.SetSpacing(image.GetSpacing())
                sitk.WriteImage(image_cropped, path +
                                'new_nii_resampled_normalized_cropped_files/'
                                + file.replace(".nii.gz",
                                                "_cropped.nii.gz"))
    except:
        print('The path did not contains any available files!!!')


# def images_standard(path):
#     files = os.listdir(path)
#     nums = len(files)
#     os.makedirs(path + 'new_nii_resampled_cropped_normalized_files', exist_ok=True)
#     CT_list, MR_list = [], []
#     for file in files:
#         if os.path.isfile(path + file):
#             image = sitk.ReadImage(path + file)
#             image_array = sitk.GetArrayFromImage(image)
#             if 'CT' in file:
#                 CT_list.append(image_array)
#             else:
#                 MR_list.append(image_array)
#     CT_array, MR_array = np.array(CT_list), np.array(MR_list)
#     CT_mean, CT_std = np.mean(CT_array), np.std(CT_array)
#     MR_mean, MR_std = np.mean(MR_array), np.std(MR_array)
#     CT_array, MR_array = (CT_array - CT_mean) / CT_std, (MR_array - MR_mean) / MR_std
#     i = 0
#     for file in files:
#         if os.path.isfile(path + file):
#             image = sitk.ReadImage(path + file)
#             if 'CT' in file:
#                 CT_file = CT_array[i//2]
#                 CT_stand = sitk.GetImageFromArray(CT_file)
#                 CT_stand.SetOrigin(image.GetOrigin())
#                 CT_stand.SetDirection(image.GetDirection())
#                 CT_stand.SetSpacing(image.GetSpacing())
#                 sitk.WriteImage(CT_stand, path +
#                                 'new_nii_resampled_cropped_normalized_files/'
#                                 + file.replace("_resampled.nii.gz", "_resampled_cropped_normalized.nii.gz"))
#             else:
#                 MR_file = MR_array[i//2]
#                 MR_stand = sitk.GetImageFromArray(MR_file)
#                 MR_stand.SetOrigin(image.GetOrigin())
#                 MR_stand.SetDirection(image.GetDirection())
#                 MR_stand.SetSpacing(image.GetSpacing())
#                 sitk.WriteImage(MR_stand, path +
#                                 'new_nii_resampled_cropped_normalized_files/'
#                                 + file.replace("_resampled.nii.gz", "_resampled_cropped_normalized.nii.gz"))
#             i = i + 1

        


if __name__ == '__main__':
    image_path = 'D:/Transmorph_Frame/Data/raw_tumors_seg/new_nii_files/'
    is_seg = 'tumors_seg' in image_path
    # nrrd2nii(image_path)
    # image_path = image_path + 'new_nii_files/'
    # image_resample(image_path, is_seg=True)
    axis_bias = [-1, -30, 3]
    if not is_seg:
        image_path = image_path + 'new_nii_resampled_files/'
        # image_norm_percentile(image_path, ct_min=0, ct_max=2000)
        image_path = image_path + 'new_nii_resampled_normalized_files/'
        image_crop_pad(image_path, bias=axis_bias, new_size=[128, 192, 224])
    else:
        # image_path = image_path + 'new_nii_resampled_files/'
        image_crop_pad(image_path, bias=axis_bias, new_size=[128, 192, 224])
    # image_path = r'D:/Transmorph_Frame/Data/training_validation_datasets_tcia/train_nii/'
    # image_norm_percentile(image_path, ct_min=-1000, ct_max=1000)
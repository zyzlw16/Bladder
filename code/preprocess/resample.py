import SimpleITK as sitk
import os
import numpy as np


input_folder = "/data/Bladder/data/TCGA_V/V_nii/image/"
output_folder = "/data/Bladder/data/TCGA_V/V_resample/image/"


file_list = os.listdir(input_folder)


def resample_image(itk_image, out_spacing=[0.7, 0.7, 3.0]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
  
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline) 
    #resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(itk_image)

for nii_file in file_list:
    input_path = os.path.join(input_folder, nii_file)
    output_path = os.path.join(output_folder, nii_file)

    Original_img = sitk.ReadImage(input_path)
    
    Resample_img = resample_image(Original_img)
    
    sitk.WriteImage(Resample_img, output_path)

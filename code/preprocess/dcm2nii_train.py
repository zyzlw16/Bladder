import os
import SimpleITK as sitk
import numpy as np
import shutil

#dicom的文件夹
input_dir='/data/Bladder/data/TCGA/V'
output_dir='/data/Bladder/data/TCGA/V_nii'

def read_dcm(file_path):
    series_reader = sitk.ImageSeriesReader()
    series_files_path = series_reader.GetGDCMSeriesFileNames(file_path)

    series_reader.SetFileNames(series_files_path)

    img = series_reader.Execute()
    print("img:", img.GetDirection())
    file_reader = sitk.ImageFileReader()

    file_reader.SetFileName(series_files_path[6])
    file_reader.ReadImageInformation()
    manufacturer = file_reader.GetMetaData("0008|0070") if file_reader.HasMetaDataKey("0008|0070") else "unknown"
    print(f"**{manufacturer}**")
    return img, manufacturer


def dcm2nii(input_dir, output_dir, name):

    files = os.listdir(input_dir)
    if files[0].endswith('.nii.gz'): lbl, img = files[0], files[1]
    else : lbl, img = files[1], files[0]
    
    inp_dir = os.path.join(input_dir, img)
    print(inp_dir)
    data,_ = read_dcm(inp_dir)
    
    output_nifti_filename = os.path.join(output_dir, 'image', f"img_{name}.nii.gz")
    
    output = os.path.join(output_dir, output_nifti_filename)
    sitk.WriteImage(data,output)

    label = os.path.join(input_dir, lbl)
    output_label = os.path.join(output_dir, 'label', f"label_{name}.nii.gz")
    shutil.copy(label, output_label)


if __name__ == '__main__':

    for parent in sorted(os.listdir(input_dir)):
        parent_folder = os.path.join(input_dir, parent)
        if not os.path.isdir(parent_folder):
                print('no file!!!')
                continue
        # numbers = ''
        # for char in parent:
        #     if char.isdigit():
        #         numbers += char

        dcm2nii(parent_folder, output_dir, parent)
    
    print("over!")   
 

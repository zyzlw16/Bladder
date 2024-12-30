import numpy as np
import SimpleITK as sitk
import csv
import os

def GetdilateNii(ct_image, mask_image, save_nii_dir=None):
    ct = sitk.GetArrayFromImage(ct_image)
    mask_np = sitk.GetArrayFromImage(mask_image)


    z_min = np.argmax(np.any(mask_np == 1, axis=(1, 2)))
    z_max = len(mask_np) - np.argmax(np.any(mask_np[::-1] == 1, axis=(1, 2)))
    print(z_max,z_min)

    z_middle = z_min + (z_max - z_min)//2

    expand_z = 16
    z_min = max(0, z_middle - expand_z)
    z_max = min(mask_np.shape[0], z_middle + expand_z)

   
    non_zero_indices = np.argwhere(mask_np[z_middle] == 1)
    print(non_zero_indices.size)
    if non_zero_indices.size == 0: 
        
        center_x = mask_np.shape[1] // 2
        center_y = mask_np.shape[2] // 2
        print('!!!!!!!!!!!!!!!!')
    else:
        center_x = int(np.mean(non_zero_indices[:, 0]))
        center_y = int(np.mean(non_zero_indices[:, 1]))


    expand_xy = 64
    x_min = max(0, center_x - expand_xy)
    x_max = min(mask_np.shape[1], center_x + expand_xy)
    y_min = max(0, center_y - expand_xy)
    y_max = min(mask_np.shape[2], center_y + expand_xy)

    ct_crop = ct[z_min:z_max, x_min:x_max, y_min:y_max]

    final_img = sitk.GetImageFromArray(ct_crop)

    final_img.SetSpacing(ct_image.GetSpacing())
    final_img.SetOrigin(ct_image.GetOrigin())
    final_img.SetDirection(ct_image.GetDirection())

    if save_nii_dir:
        sitk.WriteImage(final_img, save_nii_dir)

    return final_img, ct_crop

all_ct_dir = '/data/Bladder/data/GDPH_V/V_resample/image'
all_mask_dir = '/data/Bladder/data/GDPH_V/V_resample/label'

output_csv = []
missing_files = []
with open('/data/Bladder/data/广东临床.csv', 'r', newline='',  encoding='gbk') as input_file:
     csv_reader = csv.DictReader(input_file)

     for row in csv_reader:
        ct_id = row['影像号']
      
        ct_image_dir = os.path.join(all_ct_dir,f'img_{ct_id}.nii.gz')
        mask_image_dir = os.path.join(all_mask_dir,f'label_{ct_id}.nii.gz')#[0]
        if not os.path.exists(ct_image_dir):
            missing_files.append(ct_id)
            continue

        ct_image = sitk.ReadImage(ct_image_dir)
        print(ct_image_dir)
        print(mask_image_dir)
        mask_image = sitk.ReadImage(mask_image_dir)
        
        dilate_mask_Nii = GetdilateNii(ct_image, mask_image, f"/data/Bladder/data/GDPH_V/ROI_area/{ct_id}.nii.gz")



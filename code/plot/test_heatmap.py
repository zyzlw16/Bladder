import matplotlib.pyplot as plt
#from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
#from torchcam.methods import SmoothGradCAMpp
import SimpleITK as sitk
from PIL import Image
import numpy as np
import os
from monai.visualize.utils import blend_images
import matplotlib
matplotlib.use('Agg')
img_path = '/data/Bladder/data/QDUH_V/ROI_area_mask'
map_path = '/data/Bladder/out_map/map'
save_path= '/data/Bladder/out_map/CAM'

for lists in os.listdir(map_path):
    img = sitk.ReadImage(os.path.join(img_path, lists))
    img = sitk.GetArrayFromImage(img)

    mid = int(16)

    map = sitk.ReadImage(os.path.join(map_path, lists))
    map = sitk.GetArrayFromImage(map)

    monai_result = blend_images(np.swapaxes(img[mid][np.newaxis,:],-1,1), map[mid][np.newaxis,:], alpha=0.2)

    # Create a subplot with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot img in the first subplot
    img_nii = np.swapaxes(img[mid], -1, 0)
    img_plot = axs[0].imshow(img_nii, cmap='gray')  # You may need to specify the colormap depending on your data
    axs[0].set_title('CT Image')
    axs[0].axis('off')
    # Add colorbar for the first subplot
    cbar_img = fig.colorbar(img_plot, ax=axs[0])

    # Plot monai_result[0] in the second subplot
    monai_plot = axs[1].imshow(monai_result[0], cmap='jet')  # You may need to specify the colormap depending on your data
    axs[1].set_title('Class Activation Map')
    axs[1].axis('off')
    # Add colorbar for the second subplot
    cbar_monai = fig.colorbar(monai_plot, ax=axs[1])

    plt.tight_layout()

    name = lists.split('.')[0] + '.png'
    plt.savefig(os.path.join(save_path, name))
    #break
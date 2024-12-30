import logging
import os
import sys
sys.path.append('/data/Bladder/code')
import numpy as np
import torch
import csv

import nibabel as nib
from logger import get_logger
from sklearn.model_selection import KFold
import monai
from monai.data import decollate_batch, DataLoader
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, RandFlipd, ScaleIntensityRanged
from monai.networks.nets import densenet
from monai.utils import set_determinism
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from densenet import DenseNet121
from monai.visualize import OcclusionSensitivity
import shap
import matplotlib
matplotlib.use('Agg')
def remove_prefix(state_dict, prefix):
    """
    Remove prefix from keys in state_dict recursively.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def load_pre_trained(model, path):
   
    print('Loading pre-trained weights!')
    pretrained_dict = torch.load(path, map_location='cpu')

    model_dict = model.state_dict()
    new_dict = remove_prefix(pretrained_dict, prefix='module.')
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in new_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model
def main():
   
    set_determinism(42)
    logger = get_logger('./train_risk.log')

    data_path = '/data/Bladder/data/QDUH_V/ROI_area_mask'
    images = sorted(os.listdir(data_path))
    image_map = {os.path.splitext(os.path.splitext(image)[0])[0]: os.path.join(data_path, image) for image in images}

    labels_map = {}
    with open('/data/Bladder/data/青医临床.csv', 'r', newline='',  encoding='gbk') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            if row[0].strip() != '':
                key = row[0].strip()
                time = row[3].strip()
                status = row[2].strip()
                labels_map[key] = [time, status]
    train_files = [(image_map[id], label) for id, label in labels_map.items() if id in image_map]

    data_path = '/data/Bladder/data/SDPH_V/ROI_area_mask'
    #data_path = '/data/Bladder/data/GDPH_V/ROI_area_mask'
    images_test = sorted(os.listdir(data_path))
    image_test_map = {os.path.splitext(os.path.splitext(image)[0])[0]: os.path.join(data_path, image) for image in images_test}
    test_map = {}
    with open('/data/Bladder/data/山东省立临床1.csv', 'r', newline='',  encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            if row[0].strip() != '':
                key = row[0].strip()
                time = row[3].strip()
                status = row[2].strip()
                test_map[key] = [time, status]
    test_files = [(image_test_map[id], label) for id, label in test_map.items() if id in image_test_map]
    splits = []

    splits.append({
        "train": train_files,
        "val": test_files
    })
    for idx, train_val_test in enumerate(splits):
        
        #val_files = [{"img": files_idx[0], "label": np.array(files_idx[1], dtype=int)} for files_idx in train_val_test['train']]
        #val_files = [{"img": files_idx[0], "label": np.array(files_idx[1], dtype=int)} for files_idx in train_val_test['val']]
        val_files = [{"img": files_idx[0], "label": np.array(files_idx[1], dtype=int)} for files_idx in train_val_test['train'][536:]]
        print(len(val_files))
    
        val_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityRanged(keys=["img"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                Resized(keys=["img"], spatial_size=(128, 128, 32)),
            ]
        )
    
        # create a validation data loader
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available(), shuffle=True)

        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
        #model = SSLHead(n_class=1).to(device)
        #model = SSLResnet(n_class=1).to(device)
        #pre_trained_path = '/data/Bladder/model_dilate/best_metric_model_40.pth'
        pre_trained_path = '/data/Bladder/model_cl/best_metric_model_densenet6_60.pth'
        model = load_pre_trained(model=model, path = pre_trained_path)
        logger.info("start infer....")

        model.eval()
        #with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        y_label = []
        for i , val_data in enumerate(val_loader):
            #fig = plt.figure()
            val_images = val_data["img"].to(device)#.requires_grad_(True)
            val_labels = val_data["label"].to(device)
            img_names = val_data["img"].meta["filename_or_obj"][1].split('/')[-1].split('.nii.gz')[0]
            #pre_cox= model(val_images)
            explainer = shap.DeepExplainer(model, val_images[1:])
            shap_values = explainer.shap_values(val_images[:1] ,check_additivity=False)
            shap_values_2D = shap_values.squeeze(1)[:,:,:,16] 
            image_2D = val_images[1:][:,:,:,:,16]
            shap_numpy = np.stack([np.swapaxes(np.swapaxes(x, 1, -1), 2, 1) for x in shap_values_2D])
            test_numpy = np.swapaxes(np.swapaxes(image_2D.cpu().numpy(), 1, -1), 2, 1)

            shap.image_plot(shap_numpy, test_numpy, show=False)
            plt.draw()
            plt.savefig(f'/data/Bladder/out_map/SHAP/{img_names}_16_shape.png')
           # plt.close(fig)
            print(shap_values.shape)
           
if __name__ == "__main__":
    main()           

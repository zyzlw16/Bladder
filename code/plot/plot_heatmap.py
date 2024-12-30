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
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

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
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            y_label = []
            for i , val_data in enumerate(val_loader):
                val_images = val_data["img"].to(device)
                val_labels = val_data["label"].to(device)
                occ_map = torch.tensor([], dtype=torch.float32, device=device)
                print(val_files[i])

                origin_img = nib.load(val_files[i]['img'])
                resize_inverse = monai.transforms.Resize(spatial_size=origin_img.shape)
            
                for depth_slice in range(val_images[0].shape[-1]):
                    occ_sens =OcclusionSensitivity(nn_module=model, mask_size=12, n_batch=10)
                    occ_sens_b_box = [-1, -1, -1, -1, depth_slice-1 , depth_slice]
                    occ_result, _ = occ_sens(x=val_images, b_box=occ_sens_b_box)
                    occ_result = occ_result[0, 0]#[None]
                    if depth_slice==0:
                        occ_map = occ_result
                    else: occ_map[...,depth_slice] = occ_result[:,:,0]
                    
                name = val_files[i]['img'].split('/')[-1]
               # img = resize_inverse(val_images[i][0].squeeze(0)[None]).squeeze(0).detach().cpu().numpy()
                img = val_images[0][0].detach().cpu().numpy()
                map_result = resize_inverse(occ_map[None]).squeeze(0).detach().cpu().numpy()
                map_result = nib.Nifti1Image(map_result, affine=origin_img.affine, header=origin_img.header)
                img = nib.Nifti1Image(img, affine=origin_img.affine, header=origin_img.header)
                nib.save(img, '/data/Bladder/out_map/img/' + name )
                nib.save(map_result, '/data/Bladder/out_map/map/' + name )
            #     pre_cox= model(val_images)
            #     y_pred = torch.cat([y_pred, pre_cox[:, 0]], dim=0)
            #     y = torch.cat([y, val_labels], dim=0)
            #     y_label.append(val_files[i]['img'].split('/')[-1])
            # os_ci_test = concordance_index(y[:, 0:2], -y_pred)
            # ci_lower, ci_upper = bootstrap_ci(y[:, 0:2], -y_pred)
            # logger.info(f'Test OS cindex: {os_ci_test:.4f} [{ci_lower: .3f}, {ci_upper: .3f}]')
        # model.eval()
        # with torch.no_grad():
            
        #     y = torch.tensor([], dtype=torch.long, device=device)
        #     for i , val_data in enumerate(val_loader):
        #         val_images = val_data["img"].to(device), 
        #         val_labels = torch.tensor([int(x) for x in val_data["label"]]).to(device)
        #         occ_map = torch.tensor([], dtype=torch.float32, device=device)
        #         print(val_files[i])

        #         origin_img = nib.load(val_files[i]['img'])
        #         resize_inverse = monai.transforms.Resize(spatial_size=origin_img.shape)
            
        #         for depth_slice in range(val_images[0].shape[2]):
        #             occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=12, n_batch=10)
        #             occ_sens_b_box = [depth_slice-1 , depth_slice, -1, -1, -1, -1]
        #             occ_result, _ = occ_sens(x=val_images[0], b_box=occ_sens_b_box)
        #             occ_result = occ_result[0, val_labels[0].argmax().item()]#[None]
        #             if depth_slice==0:
        #                 occ_map = occ_result
        #             else: occ_map[depth_slice,...] = occ_result
                    
        #         name = val_files[i]['img'].split('/')[-1]
        #         img = resize_inverse(val_images[i][0].squeeze(0)[None]).squeeze(0).detach().cpu().numpy()
        #         map_result = resize_inverse(occ_map[None]).squeeze(0).detach().cpu().numpy()
        #         map_result = nib.Nifti1Image(map_result, affine=origin_img.affine, header=origin_img.header)
        #         img = nib.Nifti1Image(img, affine=origin_img.affine, header=origin_img.header)
        #         nib.save(img, './out_map/img/' + name )
        #         nib.save(map_result, './out_map/map/' + name )
            # out_img = sitk.GetImageFromArray(val_images[0][0].squeeze(0).detach().cpu())
            # sitk.WriteImage(out_img, './img.nii.gz')
            # sitk.WriteImage(out_result, './map.nii.gz')
        #     y_pred = torch.cat([y_pred, model(val_images[0])], dim=0)
        #     y = torch.cat([y, val_labels], dim=0)

        # acc_value = torch.eq(y_pred.argmax(dim=1), y)
        # acc_metric = acc_value.sum().item() / len(acc_value)
        # y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        # y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        # auc_metric(y_pred_act, y_onehot)
        # auc_result = auc_metric.aggregate()
        # auc_metric.reset()
        # print(
        #         " current accuracy: {:.4f} current AUC: {:.4f} ".format(
        #                 acc_metric, auc_result)
        #             )
if __name__ == "__main__":
    main()
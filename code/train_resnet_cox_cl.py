import logging
import os
import sys

import numpy as np
import torch
import csv
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from logger import get_logger
from sklearn.model_selection import KFold

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityRanged, RandShiftIntensityd, RandFlipd, OneOf,CopyItemsd, RandCoarseDropoutd
from monai.networks.nets import densenet
from monai.utils import set_determinism

#from swinMM import SSLHead, load_pretrained_model
from monai.losses import ContrastiveLoss
from loss import cox_loss, concordance_index
from ResNet50 import SSLResnet, load_pretrained_model
#from ResNet50 import SSLResnet, load_pretrained_model
#from monai.networks.nets import densenet, resnet, vitautoenc, efficientnet
from densenet import DenseNet121
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def load_pretrained_model(model, pretrained_model, prefix='encoder.'):

    pretrained_state_dict = torch.load(pretrained_model, map_location='cpu')
    model_state_dict = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    load_keys = []

    for pre_key in pretrained_state_dict:
        model_key = prefix + pre_key
        if model_key in model_state_dict:
            if model_state_dict[model_key].shape == pretrained_state_dict[pre_key].shape:
                model_state_dict[model_key] = pretrained_state_dict[pre_key]
                load_keys.append(model_key)
            else:
                print(model_state_dict[model_key].shape, pretrained_state_dict[pre_key].shape)
                missing_keys.append(model_key)
        else:
            unexpected_keys.append(model_key)
    
    model.load_state_dict(model_state_dict)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(42)
    logger = get_logger('./train_risk_densenet.log')
    
    data_path = '/data/Bladder/data/QDUH_V/ROI_area_mask'
    images = sorted(os.listdir(data_path))
    image_map = {os.path.splitext(os.path.splitext(image)[0])[0]: os.path.join(data_path, image) for image in images}

    labels_map = {}
    with open('/data/Bladder/data/青医临床.csv', 'r', newline='',  encoding='gbk') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            if row[0].strip() != '':
                key = row[0].strip()
                time = row[3].strip()
                status = row[2].strip()
                labels_map[key] = [time, status]
    train_files = [(image_map[id], label) for id, label in labels_map.items() if id in image_map]

    data_path = '/data/Bladder/data/QDUH_V/ROI_area_mask'
    images_test = sorted(os.listdir(data_path))
    image_test_map = {os.path.splitext(os.path.splitext(image)[0])[0]: os.path.join(data_path, image) for image in images_test}
    test_map = {}
    with open('/data/Bladder/data/青医临床.csv', 'r', newline='',  encoding='gbk') as file:
        reader = csv.reader(file)
        next(reader)  
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
        
        train_files = [{"img": files_idx[0], "label": np.array(files_idx[1], dtype=int)} for files_idx in train_val_test['train'][:536]]
        val_files = [{"img": files_idx[0], "label": np.array(files_idx[1], dtype=int)} for files_idx in train_val_test['train'][536:]]

        print(len(train_files), len(val_files))
        # Define transforms for image
        train_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityRanged(
                keys=["img"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                ),
                Resized(keys=["img"], spatial_size=(128, 128, 32)),
                RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0,1]),
                RandFlipd(keys=["img"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                RandShiftIntensityd(keys=["img"],offsets=0.10,prob=0.50),
                CopyItemsd(keys=["img"], times=1, names=["img2"], allow_missing_keys=True),
                OneOf(
                transforms=[
                    RandRotate90d(keys=["img2"], prob=1.0, spatial_axes=[0,1], allow_missing_keys=True),
                    RandFlipd(keys=["img2"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                #RandShiftIntensityd(keys=["img2"],offsets=0.10,prob=0.50),
                ]
        ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityRanged(keys=["img"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                Resized(keys=["img"], spatial_size=(128, 128, 32)),
            ]
        )
        # Define dataset, data loader
        check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        check_loader = DataLoader(check_ds, batch_size=32, num_workers=4, pin_memory=torch.cuda.is_available())
        check_data = monai.utils.misc.first(check_loader)
        print(check_data["img"].shape, check_data["label"])

        # create a training data loader
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

        # create a validation data loader
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=32, num_workers=4, pin_memory=torch.cuda.is_available())

        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
        #model = efficientnet.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=1).to(device)
        # model = resnet.ResNet('basic',[2, 2, 2, 2],block_inplanes=[64, 128, 256, 512],
        # n_input_channels=1,
        # widen_factor=2,
        # conv1_t_stride=2,
        # num_classes = 1,
        # spatial_dims = 3
        # ).to(device)
        #load_pretrained_model(model, pretrained_model='/data/Bladder/model_ssl/best_metric_model_ssl_apr.pth', prefix='')


        #contrastive_loss = ContrastiveLoss(temperature=0.05)
        optimizer = torch.optim.Adam(model.parameters(), 1e-5)
        # start a typical PyTorch training
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        writer = SummaryWriter()
        logger.info("start trainging....")
        for epoch in range(100):
            logger.info("-" * 10)
            logger.info(f"epoch {epoch + 1}/{100}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs = batch_data["img"].to(device)

                labels = batch_data["label"].to(device)
         
                optimizer.zero_grad()
                outputs = model(inputs)

                loss1 = cox_loss(labels, outputs)

                loss = loss1 
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    for val_data in val_loader:
                        val_images = val_data["img"].to(device)
                        val_labels = val_data["label"].to(device)
                
                        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)

                    os_ci_test = concordance_index(y[:, 0:2], -y_pred)
                    logger.info(f'Test OS cindex: {os_ci_test:.4f}')
                    
                    if os_ci_test > best_metric:
                        best_metric = os_ci_test
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), f"/data/Bladder/model_scale/best_metric_model_{best_metric_epoch}.pth")
                        logger.info("saved new best metric model")
                    logger.info(
                        "current epoch: {} current os_ci_test: {:.4f} best ci: {:.4f} at epoch {}".format(
                            epoch + 1, os_ci_test, best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_accuracy", os_ci_test, epoch + 1)
        logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        writer.close()


if __name__ == "__main__":
    main()
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import numpy as np
import torch
import csv
from torch.utils.tensorboard import SummaryWriter

from logger import get_logger
from sklearn.model_selection import KFold

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, RandFlipd
from monai.networks.nets import densenet
from monai.utils import set_determinism
from swinMM import SSLHead

from loss import cox_loss, concordance_index

def load_pre_trained(model, path):
   
    print('Loading pre-trained weights!')
    pretrained_dict = torch.load(path, map_location='cpu')

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(42)
    logger = get_logger('./train.log')
    
    data_path = '/data/Bladder/data/A_nii_resample'
    images = sorted(os.listdir(data_path))

    image_map = {os.path.splitext(os.path.splitext(image)[0])[0]: os.path.join(data_path, image) for image in images}

    labels_map = {}
    with open('/data/Bladder/follow_up.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            if row[1].strip() != '':
                key = row[0].strip()
                time = row[2].strip()
                status = row[1].strip()
                labels_map[key] = [time, status]
                #labels_map[row[0].strip()] = int(row[-1].strip())

    data_files = [(image_map[id], label) for id, label in labels_map.items() if id in image_map]
    labels = np.array([label for _, label in data_files], dtype=np.int64)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []

    for trainval_index, test_index in kf.split(data_files):
        trainval_files =  [data_files[i] for i in trainval_index]
        
        kf_trainval = KFold(n_splits=4, shuffle=True, random_state=42)

        for train_index, val_index in kf_trainval.split(trainval_files):
            train_files = [trainval_files[i] for i in train_index]
            val_files = [trainval_files[i] for i in val_index]
        
        test_files = [data_files[i] for i in test_index]
        
        splits.append({
            "train": train_files,
            "val": val_files,
            "test": test_files
        })
    for idx, train_val_test in enumerate(splits):
        
        #train_files = [{"img": files_idx[0], "label": np.array(files_idx[1], dtype=int)} for files_idx in train_val_test['train']]
        val_files = [{"img": files_idx[0], "label": np.array(files_idx[1], dtype=int)} for files_idx in train_val_test['test']]

        print(len(val_files))
    
        val_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(256, 256, 8)),
            ]
        )
        
        # Define dataset, data loader

        # create a validation data loader
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SSLHead(n_class=1).to(device)
        #pre_trained_path = './cox_all_model/' + f"best_metric_model_classification3d_dict_{idx}.pth"
        pre_trained_path = './' + f"best_metric_model_classification3d_dict_{idx}.pth"
        model = load_pre_trained(model=model, path = pre_trained_path)
        
        writer = SummaryWriter()
        logger.info("start infer....")
        
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            y_label = []
            for i , val_data in enumerate(val_loader):
                val_images = val_data["img"].to(device)
                val_labels = val_data["label"].to(device)

                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                y_label.append(val_files[i]['img'].split('/')[-1])
            os_ci_test = concordance_index(y[:, 0:2], -y_pred)
            logger.info(f'Test OS cindex: {os_ci_test:.4f}')
            #print(-y_pred)
            pred = y_pred.detach().cpu().numpy().reshape(-1)
            label = y.detach().cpu().numpy()
            #print(label[:, 0])
            #print(y_label, label[0][0], label[0][1], pred)
            # with open(f'./pre_cox_min_re_val{idx}.csv', 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(['name', 'time', 'status', 'pre'])

            #     for row in zip(y_label, label[:, 0], label[:, 1], pred):
            #         writer.writerow(row)

if __name__ == "__main__":

    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

# ignore weird np warning
import warnings
import random

import numpy as np
import torch

from dataloader.custom_dataloader_SK import collate_fn_BEV_test, Custom, SemKITTI_label_name, spherical_dataset
from dataloader.data_adjustment import train2SemKITTI, genColors, removeObject, createFiles
from network.polarnet.BEV_Unet import BEV_Unet
from network.polarnet.ptBEV import ptBEVnet

warnings.filterwarnings("ignore")


def run(args):
    data_path = args.data_dir
    test_batch_size = args.test_batch_size
    model_save_path = args.model_save_path
    compression_model = args.grid_size[2]
    grid_size = args.grid_size
    pytorch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fea_dim = 9
    circular_padding = True

    # prepare miou fun
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1

    # prepare model
    my_BEV_model = BEV_Unet(n_class=len(unique_label), n_height=compression_model, input_batch_norm=True, dropout=0.5,
                            circular_padding=circular_padding)
    my_model = ptBEVnet(my_BEV_model, pt_model='pointnet', grid_size=grid_size, fea_dim=fea_dim, max_pt_per_encode=256,
                        out_pt_fea_dim=512, kernal_size=1, pt_selection='random', fea_compre=compression_model)
    if os.path.exists(model_save_path):
        my_model.load_state_dict(torch.load(model_save_path))
    my_model.to(pytorch_device)


    # prepare dataset
    test_pt_dataset = Custom(data_path + '/infer/', return_ref=True)
    test_dataset = spherical_dataset(test_pt_dataset, grid_size=grid_size, ignore_label=0, fixed_volume_space=True,
                                     return_test=True)
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=test_batch_size,
                                                      collate_fn=collate_fn_BEV_test,
                                                      shuffle=False,
                                                      num_workers=4)
    # test
    with torch.no_grad():
        for i_iter_test, (_, _, test_grid, _, test_pt_fea, test_index) in enumerate(test_dataset_loader):
            # predict
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i[:, :2]).to(pytorch_device) for i in test_grid]

            predict_labels = my_model(test_pt_fea_ten, test_grid_ten)
            predict_labels = torch.argmax(predict_labels, 1).type(torch.uint8)
            predict_labels = predict_labels.cpu().detach().numpy()

            for count, i_test_grid in enumerate(test_grid):
                test_pred_label = predict_labels[
                    count, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]]
                test_pred_label = train2SemKITTI(test_pred_label)
                test_pred_label = np.expand_dims(test_pred_label, axis=1)
                test_pred_label = test_pred_label.astype(np.uint32)
    del test_grid, test_pt_fea, test_index

    test_pred_label = np.squeeze(test_pred_label, axis=1)

    visual = genColors(test_pred_label, test_pt_dataset.learning_map_inv, test_pt_dataset.color_map,
                       test_pt_dataset.label_map)
    pointCloud = np.hstack((test_pt_dataset.__getitem__(0)[0], visual))

    createFiles(pointCloud)

# Run Inference
if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='data')
    parser.add_argument('-p', '--model_save_path', default='pretrained_weight/SemKITTI_PolarSeg.pt')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default=[480, 360, 32],
                        help='grid size of BEV representation (default: [480,360,32])')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for training (default: 1)')

    args = parser.parse_args()
    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')

    run(args)

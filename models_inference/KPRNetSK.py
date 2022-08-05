import argparse

import numpy as np
import torch
from tqdm import tqdm

from dataloader.inference_dataloader_kpr import (
    SemanticKitti,
    map_inv,
)

from network.kprnet import deeplab

# -*- coding: utf-8 -*-
import os

import numpy as np
import rospy
import ros_numpy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import torch

from dataloader.data_adjustment import train2SemKITTI, genColors, labels2colors
import yaml

# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")


class KPRNetSK:
    # initialize model
    def __init__(self, model_save_path='/home/grl_ra22/PycharmProjects/liveModel/pretrained_weight/kpr_trained.pth', grid_size=[480, 360, 32],
                 pytorch_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), circular_padding=True,
                 fea_dim=9, scans_per_mesh=100):

        # set class variables
        self.grid_size = grid_size
        self.circular_padding = circular_padding
        self.compression_model = self.grid_size[2]
        self.fea_dim = fea_dim
        self.pytorch_device = pytorch_device
        self.curr_msg = None
        self.scans2mesh = scans_per_mesh
        self.pcd = []

        # get labels
        self.semkittiyaml = None
        with open("/home/grl_ra22/PycharmProjects/PolarNet-ROS/kprnet.yaml", 'r') as stream:
            self.semkittiyaml = yaml.safe_load(stream)
        stream.close()

        self.learning_map = self.semkittiyaml['learning_map']
        self.learning_map_inv = self.semkittiyaml['map_inv']
        self.color_map = self.semkittiyaml['color_map']

        # create ros subscriber to os1 pointcloud2 msg
        self.subscriber = rospy.Subscriber("/os1_cloud_node/points", PointCloud2, self.callback, queue_size=1)
        self.publisher = rospy.Publisher("labeled/points", PointCloud2)

        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('intensity', 12, PointField.FLOAT32, 1),
                       PointField('r', 16, PointField.FLOAT32, 1),
                       PointField('g', 20, PointField.FLOAT32, 1),
                       PointField('b', 24, PointField.FLOAT32, 1)
                       ]
        self.header = Header()
        self.header.frame_id = 'map'

        # ros message control
        rospy.logwarn("Initializing")

        # create and prepare model
        self.model = deeplab.resnext101_aspp_kp(19)
        self.model.to(self.pytorch_device)
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.eval()

        rospy.logwarn("Initialized")

    # run model for real time inference
    def run(self):

        # wait for one message at a time
        while not rospy.is_shutdown():
            if self.curr_msg is not None:
                # preprocess data
                data = ros_numpy.numpify(self.curr_msg)
                data = np.concatenate(data[['x', 'y', 'z', 'intensity']]).ravel()
                data = np.array(data.tolist())

                # get output of model
                # result = np.delete(self.output(data), obj=3, axis=1).astype(float)
                result = np.hstack((data, self.output(data)))

                # prepare and send new labeled pointcloud
                pc2 = point_cloud2.create_cloud(self.header, self.fields, result.tolist())
                self.publisher.publish(pc2)

                # empty message for callback for new data
                self.curr_msg = None

    # output semantically segmented point clouds
    def output(self, data):
        # load data
        dataset = SemanticKitti(data)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
        )


        with torch.no_grad():
            for step, items in tqdm(enumerate(loader), total=len(loader)):
                images = items["image"].to(self.pytorch_device)
                py = items["py"].float().to(self.pytorch_device)
                px = items["px"].float().to(self.pytorch_device)
                pxyz = items["points_xyz"].float().to(self.pytorch_device)
                knns = items["knns"].long().to(self.pytorch_device)
                predictions = self.model(images, px, py, pxyz, knns)
                _, predictions_argmax = torch.max(predictions, 1)
                test_pred_label = predictions_argmax.cpu().numpy()
                test_pred_label = test_pred_label.astype(np.uint32)

        visual = genColors(test_pred_label, self.learning_map_inv, self.color_map)

        return visual

    # get new messages and deal with threading
    def callback(self, msg):
        if self.curr_msg is None:
            self.curr_msg = msg

# a launch test of the model
def main():
    rospy.init_node('polarnet_ros')
    model = KPRNetSK()
    model.run()


if __name__ == "__main__":
    main()


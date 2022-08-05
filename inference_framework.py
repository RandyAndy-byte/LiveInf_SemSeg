# -*- coding: utf-8 -*-
import os

import numpy as np
import rospy
import ros_numpy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import torch
from typing import Optional

from dataloader.data_adjustment import genColors, removeObject
import yaml


# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")


class liveModel:
    # initialize model
    def __init__(self, model_save_path,
                 pytorch_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        # set class variables
        self.pytorch_device = pytorch_device
        self.curr_msg = None
        self.pcd = []

        # get labels
        self.semkittiyaml = None
        with open("semantic-kitti.yaml", 'r') as stream:
            self.semkittiyaml = yaml.safe_load(stream)
        stream.close()

        self.learning_map = self.semkittiyaml['learning_map']
        self.learning_map_inv = self.semkittiyaml['learning_map_inv']
        self.color_map = self.semkittiyaml['color_map']
        self.color_map_inv = self.semkittiyaml['color_map_inv']
        self.label_map = self.semkittiyaml['labels']

        # load Semantic KITTI class info
        SemKITTI_label_name = dict()
        for i in sorted(list(self.learning_map.keys()))[::-1]:
            SemKITTI_label_name[self.learning_map[i]] = self.label_map[i]

        unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1

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
        self.initialize_model()

        rospy.logwarn("Initialized")

    def initialize_model(self) -> None:
        return None

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
                visual = genColors(self.output(data), self.learning_map_inv, self.color_map, self.label_map)
                result = np.hstack((data, visual))

                result = removeObject(result, 'road')
                # prepare and send new labeled pointcloud
                pc2 = point_cloud2.create_cloud(self.header, self.fields, result.tolist())
                self.publisher.publish(pc2)

                # empty message for callback for new data
                self.curr_msg = None

    def load_data(self, data) -> Optional[torch.utils.data.DataLoader]:
        return None

    # output semantically segmented point clouds
    def output(self, data) -> Optional[np.ndarray]:
        # load data

        # Return Test Pred Labels
        return None

    # get new messages and deal with threading
    def callback(self, msg):
        if self.curr_msg is None:
            self.curr_msg = msg

# a launch test of the model
def main():
    rospy.init_node('polarnet_ros')
    model = liveModel()
    model.run()


if __name__ == "__main__":
    main()

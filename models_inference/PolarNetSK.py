# -*- coding: utf-8 -*-
import os

import numpy as np
import rospy
import ros_numpy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import torch

from dataloader.inference_dataloader_SK import collate_fn_BEV_test, Custom, spherical_dataset
from dataloader.data_adjustment import train2SemKITTI, genColors, removeObject
import yaml
from network.polarnet.BEV_Unet import BEV_Unet
from network.polarnet.ptBEV import ptBEVnet

# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")


class PolarNetSK:
    # initialize model
    def __init__(self, model_save_path='pretrained_weight/SemKITTI_PolarSeg.pt', grid_size=[480, 360, 32],
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
        my_BEV_model = BEV_Unet(n_class=len(unique_label), n_height=self.compression_model, input_batch_norm=True,
                                dropout=0.5,
                                circular_padding=self.circular_padding)
        self.my_model = ptBEVnet(my_BEV_model, pt_model='pointnet', grid_size=self.grid_size, fea_dim=self.fea_dim,
                                 max_pt_per_encode=256,
                                 out_pt_fea_dim=512, kernal_size=1, pt_selection='random',
                                 fea_compre=self.compression_model)
        if os.path.exists(model_save_path):
            self.my_model.load_state_dict(torch.load(model_save_path))
        self.my_model.to(self.pytorch_device)
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
                result = removeObject(result, 'road')
                # prepare and send new labeled pointcloud
                pc2 = point_cloud2.create_cloud(self.header, self.fields, result.tolist())
                self.publisher.publish(pc2)

                # empty message for callback for new data
                self.curr_msg = None

    # output semantically segmented point clouds
    def output(self, data):
        # load data
        test_pt_dataset = Custom(data, return_ref=True)
        test_dataset = spherical_dataset(test_pt_dataset, grid_size=self.grid_size, ignore_label=0,
                                         fixed_volume_space=True,
                                         return_test=True)
        test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=1,
                                                          collate_fn=collate_fn_BEV_test,
                                                          shuffle=False,
                                                          num_workers=4)

        with torch.no_grad():
            for i_iter_test, (_, _, test_grid, _, test_pt_fea, test_index) in enumerate(test_dataset_loader):
                # predict
                test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                   test_pt_fea]
                test_grid_ten = [torch.from_numpy(i[:, :2]).to(self.pytorch_device) for i in test_grid]

                predict_labels = self.my_model(test_pt_fea_ten, test_grid_ten)
                predict_labels = torch.argmax(predict_labels, 1).type(torch.uint8)
                predict_labels = predict_labels.cpu().detach().numpy()

                for count, i_test_grid in enumerate(test_grid):
                    test_pred_label = predict_labels[
                        count, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]]
                    test_pred_label = train2SemKITTI(test_pred_label)
                    test_pred_label = test_pred_label.astype(np.uint32)

        del test_grid, test_pt_fea, test_index

        visual = genColors(test_pred_label, self.learning_map_inv, self.color_map, self.label_map)
        return visual

    # get new messages and deal with threading
    def callback(self, msg):
        if self.curr_msg is None:
            self.curr_msg = msg

# a launch test of the model
def main():
    rospy.init_node('polarnet_ros')
    model = PolarNetSK()
    model.run()


if __name__ == "__main__":
    main()

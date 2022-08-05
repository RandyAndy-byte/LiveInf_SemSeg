# -*- coding: utf-8 -*-
import os
import struct

import numpy as np
import rospy
import ros_numpy

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import torch

from dataloader import load_save_util
from dataloader.inference_dataloader_C3D import collate_fn_BEV, SemKITTI_demo, cylinder_dataset
from dataloader.data_adjustment import genColors, train2SemKITTI
import yaml
from network.cylinder_spconv_3d import cylinder_asym
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea

# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")


class CylNet:
    # initialize model
    def __init__(self, model_save_path='pretrained_weight/cyl_nusc_0.5x_76_15.pt', grid_size=[480, 360, 32],
                 pytorch_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), circular_padding=True,
                 fea_dim=9, out_fea_dim=256, num_input_fea=16, use_norm=True, init_size=32, data="sk"):

        # set class variables
        self.grid_size = grid_size
        self.circular_padding = circular_padding
        self.compression_model = self.grid_size[2]
        self.fea_dim = fea_dim
        self.out_fea_dim = out_fea_dim
        self.num_input_fea = num_input_fea
        self.use_norm = use_norm
        self.init_size = init_size
        self.pytorch_device = pytorch_device
        self.curr_msg = None
        self.pcd = []
        self.data = data

        # get labels
        self.semkittiyaml = None
        with open("semantic-kitti.yaml", 'r') as stream:
            self.semkittiyaml = yaml.safe_load(stream)
        stream.close()

        '''
        with open('', 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        '''

        # load Semantic KITTI class info
        SemKITTI_label_name = dict()
        for i in sorted(list(self.semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[self.semkittiyaml['learning_map'][i]] = self.semkittiyaml['labels'][i]

        self.unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        self.num_class = len(self.unique_label)

        # create ros subscriber to os1 pointcloud2 msg
        self.subscriber = rospy.Subscriber("/os1_cloud_node/points", PointCloud2, self.callback, queue_size=1)
        self.publisher = rospy.Publisher("labeled/points", PointCloud2)
        # parameters for publishing pc2 messages
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

        # create and prepare model
        cylinder_3d_spconv_seg = Asymm_3d_spconv(
            output_shape=self.grid_size,
            use_norm=self.use_norm,
            num_input_features=self.num_input_fea,
            init_size=self.init_size,
            nclasses=self.num_class)

        cy_fea_net = cylinder_fea(grid_size=self.grid_size,
                                  fea_dim=self.fea_dim,
                                  out_pt_fea_dim=self.out_fea_dim,
                                  fea_compre=self.num_input_fea)

        self.my_model = cylinder_asym(
            cylin_model=cy_fea_net,
            segmentator_spconv=cylinder_3d_spconv_seg,
            sparse_shape=self.grid_size
        )

        if os.path.exists(model_save_path):
            #self.my_model.load_state_dict(torch.load(model_save_path))
            self.my_model = load_save_util.load_checkpoint(model_save_path, self.my_model)

        self.my_model.to(pytorch_device)

        self.my_model.eval()

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
        test_pt_dataset = SemKITTI_demo(self.semkittiyaml, data, return_ref=True)

        # You should play around with these parameters
        test_dataset = cylinder_dataset(
            test_pt_dataset,
            grid_size=self.grid_size,
            fixed_volume_space=True,
            max_volume_space=[50, 3.1415926, 2],
            min_volume_space=[0, -3.1415926, -4],
            ignore_label=0,
        )

        test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=1,
                                                          collate_fn=collate_fn_BEV,
                                                          shuffle=False,
                                                          num_workers=4)

        with torch.no_grad():
            for i_iter_demo, (_, test_vox_label, test_grid, test, test_pt_fea) in enumerate(test_dataset_loader):
                test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in test_pt_fea]
                test_grid_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in test_grid]
                test_batch_size = test_vox_label.shape[0]

                predict_labels = self.my_model(test_pt_fea_ten, test_grid_ten, test_batch_size)
                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()
                for count, i_test_grid in enumerate(test_grid):
                    test_pred_label = predict_labels[count, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]]
                    test_pred_label = train2SemKITTI(test_pred_label)
                    test_pred_label = test_pred_label.astype(np.uint32)

        del test_grid, test_pt_fea, test_vox_label

        visual = genColors(test_pred_label, test_pt_dataset.learning_map_inv, test_pt_dataset.color_map)

        return visual

    # get new messages and deal with threading
    def callback(self, msg):
        if self.curr_msg is None:
            self.curr_msg = msg


# a launch test of the model
def main():
    rospy.init_node('cylinder3D_ros')
    model = CylNet()
    model.run()


if __name__ == "__main__":
    main()

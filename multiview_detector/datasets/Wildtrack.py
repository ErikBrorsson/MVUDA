# code from https://github.com/hou-yz/MVDet/tree/master
# modified by Erik Brorsson
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset
import json

intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']


class Wildtrack(VisionDataset):
    def __init__(self, root, cameras=None, camera_orient="wildtrack"):#[1,2,3,4,5,6,7]):
        super().__init__(root)
        self.root = root
        self.gt_fname = os.path.join(self.root,'gt.txt')
        self.__name__ = 'Wildtrack'
        with open(os.path.join(self.root,'config.json'), 'r') as f:
            config = json.load(f)
        self.num_cam, self.num_frames = config['num_cam'], config['num_frames']
        self.dataset_name = config['Dataset']
        self.img_shape, self.world_grid_shape = config['img_shape'], config['grid_shape']
        self.grid_cell, self.origin = config['grid_cell'], config['origin']
        self.region_size = config['region_size'] 
        self.indexing = 'ij'
        self.camera_orient = camera_orient
        self.worldgrid2worldcoord_mat = np.array([[self.grid_cell, 0, self.origin[0]], [0, self.grid_cell, self.origin[1]], [0, 0, 1]])
        self.unit = 100
        self.n_pos = self.world_grid_shape[0] * self.world_grid_shape[1]

        if cameras is not None:
            self.cameras = [x - 1 for x in cameras] # in the code, the camera index is sometimes used to reference the position in a list => need range 0-6 instead of 1-7
            self.num_cam = len(self.cameras)
        else:
            self.cameras = [x for x in range(self.num_cam)]

        self.intrinsic_matrices, self.extrinsic_matrices = {}, {}
        for cam in self.cameras:
            self.intrinsic_matrices[cam], self.extrinsic_matrices[cam] = self.get_intrinsic_extrinsic_matrix(cam)

        print(self.root)
        print(f'Dataset Name : {self.dataset_name}')
        print(f'Cameras : {self.cameras}, Frames : {self.num_frames}')
        print(f'Image Shape(H,W) : {self.img_shape}')
        print(f'Grid Shape(rows,cols) : {self.world_grid_shape}')
        print(f'Grid Cell(in cm) : {self.grid_cell}cm i.e {self.grid_cell/100}m')
        print(f'Grid Origin(x,y) : {self.origin}')
        print(f'Area/Region size(in m) : {self.region_size[0]}m x {self.region_size[1]}m')

        self.bbox_by_pos_cam = self.read_pom()
        
    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in self.cameras}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam not in self.cameras:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 480
        grid_y = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 480

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord
        grid_x = (coord_x + 300) / 2.5
        grid_y = (coord_y + 900) / 2.5
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam


def test():
    from multiview_detector.utils.projection import get_imagecoord_from_worldcoord
    dataset = Wildtrack(os.path.expanduser('~/Data/Wildtrack'), )
    pom = dataset.read_pom()

    foot_3ds = dataset.get_worldcoord_from_pos(np.arange(np.product(dataset.worldgrid_shape)))
    errors = []
    for cam in dataset.cameras:
        projected_foot_2d = get_imagecoord_from_worldcoord(foot_3ds, dataset.intrinsic_matrices[cam],
                                                           dataset.extrinsic_matrices[cam])
        for pos in range(np.product(dataset.worldgrid_shape)):
            bbox = pom[pos][cam]
            foot_3d = dataset.get_worldcoord_from_pos(pos)
            if bbox is None:
                continue
            foot_2d = [(bbox[0] + bbox[2]) / 2, bbox[3]]
            p_foot_2d = projected_foot_2d[:, pos]
            p_foot_2d = np.maximum(p_foot_2d, 0)
            p_foot_2d = np.minimum(p_foot_2d, [1920, 1080])
            errors.append(np.linalg.norm(p_foot_2d - foot_2d))

    print(f'average error in image pixels: {np.average(errors)}')
    pass


if __name__ == '__main__':
    test()

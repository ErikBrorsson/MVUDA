# code from https://github.com/jeetv/GMVD/tree/main
# modified by Erik Brorsson
import os
import json
import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image
from PIL import ImageDraw
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.random_occlusion import generate_occlusion

class GetDataset3DROM(VisionDataset):
    def __init__(self, base, reID=False, train=True, transform=ToTensor(), target_transform=ToTensor(),grid_reduce=4, img_reduce=4, train_ratio=0.9, sample_require=0):
        # parameters in super can be accessed as (self.param) eg:- self.transform, self.target_transform
        super().__init__(base.root, transform=transform, target_transform=target_transform)
        self.trainstat = train
        self.base = base
        self.reID =  reID
        self.root, self.num_cam, self.num_frames, self.cameras, self.indexing, self.camera_orient = base.root, base.num_cam, base.num_frames, base.cameras, base.indexing, base.camera_orient
        self.dataset_name = base.dataset_name
        self.img_shape, self.world_grid_shape = base.img_shape, base.world_grid_shape  # H,W; N_row,N_col
        if sample_require:
           self.skip_frame_ratio = int(self.num_frames//sample_require)
        else:
           self.skip_frame_ratio = 1
        
        # Reduce grid [480,1440] by factor of 4.
        self.grid_reduce = grid_reduce
        #[480,1440]/4 --> [120,360] i.e [1200cm,3600cm] and 10cm is pixel size, therefore [1200/10,3600/10]
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.world_grid_shape)) 

        self.img_reduce = img_reduce
        
        # Map kernel size = 41*41
        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        # normalizing the kernel with max value.
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        
        # Split train/test data

        if train:
            frame_range = range(0,int(train_ratio*self.num_frames), self.skip_frame_ratio)
            print(f"frame range: range(0, {int(train_ratio*self.num_frames)},{self.skip_frame_ratio})")
        else:
            frame_range = range(int(train_ratio*self.num_frames), self.num_frames, self.skip_frame_ratio)
            print(f"frame range: range({int(train_ratio*self.num_frames)}, {self.num_frames} ,{self.skip_frame_ratio})")
        ############ Cam set Selection ############
        #frame_range = range(0,self.num_frames) 
        ###########################################
        self.img_fpath = self.base.get_image_fpaths(frame_range)
  
        # gt_map initialization 
        self.gt_map = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)


        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(self.base.intrinsic_matrices,
                                                                           self.base.extrinsic_matrices,
                                                                           self.base.worldgrid2worldcoord_mat)
        map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))
        img_zoom_mat = np.diag(np.array([self.img_reduce, self.img_reduce, 1]))        
        self.proj_mats_mvaug_features = {cam: torch.from_numpy(np.linalg.inv(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat))
                          for cam in self.cameras}

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        """
        returns:
            projection matrices between image pixels position and bev grid position.
            Here, the image size is determined by the intrinsic_matrices. 
            While the bev grid size is determined by  worldgrid2worldcoord_mat
        """
        projection_matrices = {}
        for cam in self.cameras:
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = {cam:[] for cam in self.cameras}, \
                                                 {cam:[] for cam in self.cameras}
                foot_row_cam_s, foot_col_cam_s, v_cam_s = {cam:[] for cam in self.cameras}, \
                                                          {cam:[] for cam in self.cameras}, \
                                                          {cam:[] for cam in self.cameras}
                for single_pedestrian in all_pedestrians:
                    if single_pedestrian is None:
                        continue
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in self.cameras:
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)

                # print("v_s", v_s)
                # print("i_s", i_s)
                # print("j_s", j_s)
                # print("self.reducedgrid_shape", self.reducedgrid_shape)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.gt_map[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in self.cameras:
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def download_old(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = {cam:[] for cam in self.cameras}, \
                                                 {cam:[] for cam in self.cameras}
                foot_row_cam_s, foot_col_cam_s, v_cam_s = {cam:[] for cam in self.cameras}, \
                                                          {cam:[] for cam in self.cameras}, \
                                                          {cam:[] for cam in self.cameras}
                for single_pedestrian in all_pedestrians:
                    if single_pedestrian is None:
                        continue
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in self.cameras:
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in self.cameras:
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):
        frame = list(self.gt_map.keys())[index]
        imgs = []
        imgs_3drom = []
        # generate random occlusions
        if self.transform is not None and self.trainstat is True:
            # if self.base.__name__ == "Wildtrack":
            #     random_list, _ = generate_occlusion(self.base, 0, 691199, 25)
            # elif self.base.__name__ == "MultiviewX":
            #     random_list, _ = generate_occlusion(self.base, 0, 639999, 25)
                
            random_list, _ = generate_occlusion(self.base, 0, int(self.base.n_pos) - 1, 25)

            bbox_by_pos_cam = self.base.bbox_by_pos_cam

        for cam in range(self.num_cam):
            fpath = self.img_fpath[cam][frame]
            img = Image.open(fpath).convert('RGB')
            img_3drom = img.copy()
            imga = ImageDraw.ImageDraw(img_3drom)

            if self.transform is not None and self.trainstat is True:
                for posID in random_list:
                    bbox = bbox_by_pos_cam[posID][cam]
                    if bbox is not None:
                        imga.rectangle((tuple(bbox[:2]), tuple(bbox[2:])), fill='gray', outline=None, width=1)
            if self.transform is not None:
                img = self.transform(img)
                img_3drom = self.transform(img_3drom)

            imgs.append(img)
            imgs_3drom.append(img_3drom)

        imgs = torch.stack(imgs)
        imgs_3drom = torch.stack(imgs_3drom)

        map_gt = self.gt_map[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        imgs_gt = []
        for cam in self.cameras:
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())

        proj_mats_mvaug_features = []
        for cam in self.cameras:
            proj_mats_mvaug_features.append(self.proj_mats_mvaug_features[cam])
            
        return imgs_3drom, map_gt.float(), imgs_gt, frame, frame, imgs, frame, frame, proj_mats_mvaug_features, self.root
    

    def __len__(self):
        # length of dataset
        return len(self.gt_map.keys())

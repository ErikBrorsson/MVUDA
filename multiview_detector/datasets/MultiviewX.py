# code from https://github.com/hou-yz/MVDet/tree/master
# modified by Erik Brorsson
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
import json
from torchvision.datasets import VisionDataset
from scipy.sparse import coo_matrix


intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']


class MultiviewX(VisionDataset):
    def __init__(self, root, cameras=None, camera_orient="multiviewx"):#[1,2,3,4,5,6]):
        super().__init__(root)

        self.root = root
        self.gt_fname = os.path.join(self.root,'gt.txt')
        self.__name__ = 'MultiviewX'
        with open(os.path.join(self.root,'config.json'), 'r') as f:
            config = json.load(f)
        self.num_cam, self.num_frames = config['num_cam'], config['num_frames']
        self.dataset_name = config['Dataset']
        self.img_shape, self.world_grid_shape = config['img_shape'], config['grid_shape']
        self.grid_cell, self.origin = config['grid_cell'], config['origin']
        self.region_size = config['region_size'] 
        self.indexing = 'xy'
        self.camera_orient = camera_orient
        self.unit = 1
        self.n_pos = self.world_grid_shape[0] * self.world_grid_shape[1]

        self.worldgrid2worldcoord_mat = np.array([[0,self.grid_cell, self.origin[0]], [self.grid_cell, 0, self.origin[1]], [0, 0, 1]])

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
        print(f'Cameras : {self.num_cam}, Frames : {self.num_frames}')
        print(f'Image Shape(H,W) : {self.img_shape}')
        print(f'Grid Shape(rows,cols) : {self.world_grid_shape}')
        print(f'Grid Cell(in cm) : {self.grid_cell}cm i.e {self.grid_cell/100}m')
        print(f'Grid Origin(x,y) : {self.origin}')
        print(f'Area/Region size(in m) : {self.region_size[0]}m x {self.region_size[1]}m')
        
        self.bbox_by_pos_cam = self.read_pom()
        # self.bbox_by_pos_cam = self.read_POM2()
        # self.cam_layout_list = []
        # for cam in self.cameras:
        #     cam_i = self.display_cam_layout([cam])
        #     self.cam_layout_list.append(cam_i)
        # self.pom = self.read_pom()
        # from PIL import Image
        # temp = self.display_cam_layout([2])
        # temp2 = self.draw_cameras(temp)
        # img = Image.fromarray(temp2)
        # img.save("cam_layout_contour.png")

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in self.cameras}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam not in self.cameras:
                continue
            if os.path.isdir(os.path.join(self.root, 'Image_subsets', camera_folder)):
                for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                    frame = int(fname.split('.')[0])
                    if frame in frame_range:
                        img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        R,C = self.world_grid_shape
        grid_x = pos % C
        grid_y = pos // C
        # [0,0]...[479,0],[0,1]..[479,1]...
        return np.array([grid_x, grid_y], dtype=int)
    
    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        coord_x = self.origin[0] + self.grid_cell * grid_x  # -300 + 2.5 * x
        coord_y = self.origin[1] + self.grid_cell * grid_y  # -900 + 2.5 * x
        return np.array([coord_x, coord_y])
    
    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)
    
    def get_pos_from_worldgrid(self, worldgrid):
        R,C = self.world_grid_shape
        grid_x, grid_y = worldgrid
        pos = grid_x + grid_y * C
        return pos
    
    def get_worldgrid_from_worldcoord(self, worldcoord):
        coord_x, coord_y = worldcoord
        grid_x = (coord_x - self.origin[0]) / self.grid_cell  # (cx + 300) / 2.5 
        grid_y = (coord_y - self.origin[1]) / self.grid_cell  # (cy + 900) / 2.5 
        return np.array([grid_x, grid_y], dtype=int)
    
    def get_pos_from_worldcoord(self, worldcoord):
        grid = self.get_worldgrid_from_worldcoord(worldcoord)
        return self.get_pos_from_worldgrid(grid)
    

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                      intrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                      extrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

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
    
    def final_overlap_pos(self):
        train = self.display_cam_layout(self.train_cam)
        mask_train = self.convex_hull(train)
        
        test = self.display_cam_layout(self.test_cam)
        mask_test = self.convex_hull(test)
        
        final_mask = mask_train & mask_test
        final_mask = final_mask.astype(np.uint8)*255.0
        
        coord = np.array(np.where(final_mask==255.0)).T[:,:2]
        coord = np.unique(coord, axis=0)
        print('overlap coord :', coord.shape)
        pos = []
        for p in coord:
            pos.append(self.get_pos_from_worldgrid(p[[1,0]]))
        pos = np.asarray(pos)
        print('overlap pos :', pos.shape)
        return pos
        
    def display_cam_layout(self, cam_selected):
        tmap_final = np.zeros(self.world_grid_shape).astype(int)
        for cam in cam_selected:
            i_s, j_s, v_s = [],[],[]
            for i in range(np.product(self.world_grid_shape)):
                grid_x, grid_y = self.get_worldgrid_from_pos(i)
                if i in self.bbox_by_pos_cam:
                    if self.bbox_by_pos_cam[i][cam] > 0:
                        i_s.append(grid_y)
                        j_s.append(grid_x)
                        v_s.append(1)
            tmap = coo_matrix((v_s, (i_s, j_s)), shape=self.world_grid_shape).toarray()
            
            tmap_final+=tmap
            
            '''
            plt.figure(figsize=(10,10))
            plt.subplot(w.num_cam,1,cam+1)
            plt.title('cam_'+str(cam))
            plt.axis('off')
            #plt.imshow(tmap)
            plt.imshow(tmap)
        
        
        plt.figure(figsize=(10,10))
        plt.title('final')
        plt.axis('off')
        #plt.imshow(tmap)
        plt.imshow(tmap_final, cmap='gray')
        #plt.colorbar(tmap_final)
        plt.show()
        '''
        return tmap_final
    
    def convex_hull(self, tmap):
        tmap = tmap.astype(np.uint8)*255
        '''
        cv2.imshow('ConvexHull', tmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        #gray = cv2.cvtColor(tmap, cv2.COLOR_BGR2GRAY) # convert to grayscale
        blur = cv2.blur(tmap, (3, 3)) # blur the image
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # create hull array for convex hull points
        hull = []
        
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))
            
        # create an empty black image
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        
        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - color for contours
            color = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            #cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            #cv2.drawContours(drawing, hull, i, color, 1, 8)
            cv2.fillPoly(drawing , contours, (255, 255, 255))
            
        mask = drawing == 255
        return mask
    
    def draw_cameras(self, tmap):
        # tmap = tmap.astype(np.uint8)*255
        # ret, thresh = cv2.threshold(tmap, 50, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(tmap.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros((tmap.shape[0], tmap.shape[1], 3), np.uint8)
        for i, contour in enumerate(contours):
            print(contour)
            drawing = cv2.drawContours(drawing, contour, -1, (0,255,0), 5)

        return drawing

    
    def read_POM2(self):
        filename = 'rectangles.pom'
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root,filename),'r') as f:
            for line in f:
                if 'RECTANGLE' in line:
                    cam, pos = map(int,cam_pos_pattern.search(line).groups())
                    if cam != 9 :
                        if pos not in bbox_by_pos_cam:
                            bbox_by_pos_cam[pos] = {}
                            #bbox_by_pos_cam[pos] = 0
                        if 'notvisible' in line:
                            bbox_by_pos_cam[pos][cam] = 0
                            #pass
                        else:
                            bbox_by_pos_cam[pos][cam] = 1  
                        #cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        #grid_x, grid_y = self.get_worldgrid_from_pos(pos)
                        #bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0), min(right, 1920 - 1), min(bottom, 1080 - 1)]
                        #bbox_by_pos_cam[pos][cam] = [grid_x, grid_y]
                        
        return bbox_by_pos_cam

def test():
    from multiview_detector.utils.projection import get_imagecoord_from_worldcoord
    dataset = MultiviewX(os.path.expanduser('~/Data/MultiviewX'), )
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

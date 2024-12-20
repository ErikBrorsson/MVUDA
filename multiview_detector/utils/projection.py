# code from https://github.com/hou-yz/MVDet/tree/master

import numpy as np

def get_worldcoord_from_imagecoord_w_projmat(image_coord, proj_mat):
    image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
    world_coord = proj_mat @ image_coord
    world_coord = world_coord[:2, :] / world_coord[2, :]
    return world_coord

def get_worldgrid_from_worldcoord(world_coord):
    # datasets default unit: centimeter & origin: (-300,-900)
    coord_x, coord_y = world_coord
    grid_x = (coord_x + 300) / 2.5
    grid_y = (coord_y + 900) / 2.5
    return np.array([grid_x, grid_y], dtype=int)

def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
    image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
    world_coord = project_mat @ image_coord
    world_coord = world_coord[:2, :] / world_coord[2, :]
    return world_coord


def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    project_mat = np.delete(project_mat, 2, 1)
    world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    image_coord = project_mat @ world_coord
    image_coord = image_coord[:2, :] / image_coord[2, :]
    return image_coord

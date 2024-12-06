# Author Erik Brorsson
import torch

def warp_features_pytorch(features, proj_mat, reducedgrid_shape, camera_orient):

    bev_w, bev_h = reducedgrid_shape
    x = torch.linspace(0, bev_h-1, bev_h)
    y = torch.linspace(0, bev_w-1, bev_w)
    mesh = torch.meshgrid([x,y], indexing="xy")


    grid = torch.concat([mesh[0].unsqueeze(0), mesh[1].unsqueeze(0)])
    grid = grid.reshape((2, -1))
    grid_homo = torch.ones((3, grid.shape[1]))
    grid_homo[0:2, :] = grid
    grid_homo = grid_homo.unsqueeze(0)
    grid_persp = torch.bmm(proj_mat.float().to('cuda:0'), grid_homo.to('cuda:0')).squeeze()#.cpu().numpy().squeeze()
    z = grid_persp[2, :]
    grid_persp = grid_persp / z

    grid_persp = (grid_persp[0:2,:] / torch.tensor([features.shape[-1]-1, features.shape[-2]-1], device="cuda:0").reshape((2, 1)))*2 - 1
    
    if camera_orient == "multiviewx":
        grid_persp[0:2, z > 0] = -10 # remove all points that are behind the camera
    elif camera_orient == "wildtrack":
        grid_persp[0:2, z < 0] = -10 # remove all points that are behind the camera
    else:
        raise Exception("camera_orient must be one of [multiviewx, wildtrack]")

    grid_persp = grid_persp.reshape((2, y.shape[0], x.shape[0])).unsqueeze(0)
    grid_persp = grid_persp.permute(0,2,3,1)

    world_feature = torch.nn.functional.grid_sample(features, grid_persp, mode='bilinear', align_corners=True, padding_mode="zeros")

    world_feature[torch.isnan(world_feature)] = 0

    return world_feature


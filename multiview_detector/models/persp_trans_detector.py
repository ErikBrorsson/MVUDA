# code from https://github.com/hou-yz/MVDet/tree/master
# modified by Erik Brorsson
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.geometry import warp_features_pytorch

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, arch='resnet18', pretrained=False, avgpool=False, num_cam = None, warp_kornia=False):
        super().__init__()

        self.warp_kornia = warp_kornia
        if num_cam is None:
            self.num_cam = 8
        else:
            self.num_cam = num_cam

        self.avgpool = avgpool

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(pretrained=pretrained, replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        
        if self.avgpool:
            n_inputs_channels = out_channel + 2
        else:
            n_inputs_channels = out_channel * self.num_cam + 2

        self.map_classifier = nn.Sequential(nn.Conv2d(n_inputs_channels, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        pass


    def forward(self, imgs, proj_mats, config_dict, visualize=False):
        B, N, C, H, W = imgs.shape
        
        upsample_shape = config_dict['upsample_shape']
        reducedgrid_shape = config_dict['reducedgrid_shape']
        coord_map = config_dict['coord_map']
        camera_orient = config_dict['camera_orient']

        if not self.avgpool:
            assert N == self.num_cam

        world_features = []
        imgs_result = []


        view_indicator_viz = []
        img_feature_viz = []
        world_feature_viz = []


        for i in range(N):
            img_feature = self.base_pt1(imgs[:, i].to('cuda:0'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            imgs_result.append(img_res)
            proj_mat = proj_mats[i].repeat([B, 1, 1]).float().to('cuda:0')

            # TODO CAUTION! kornia warp_perspective doesn't handle points that are behind the camera, leading to undesirable artifacts in the output.
            if self.warp_kornia:
                world_feature = kornia.geometry.transform.warp_perspective(img_feature.to('cuda:0'), proj_mat, reducedgrid_shape) # reducedgrid_shape=[480/4, 1440/4]
            else:
                world_feature = warp_features_pytorch(img_feature.to('cuda:0'), torch.linalg.inv(proj_mat), reducedgrid_shape, camera_orient)
            if visualize:
                fig = plt.figure(figsize=(16,9))
                subplt0 = fig.add_subplot(211, title="img_features")
                subplt1 = fig.add_subplot(212, title="bev_features")                
                subplt0.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                subplt1.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.savefig(f"img_and_bev_features_{i}.jpg")
                plt.close(fig)

            view_indicator = torch.ones_like(img_feature)
            # TODO CAUTION! kornia warp_perspective doesn't handle points that are behind the camera, leading to undesirable artifacts in the output.
            if self.warp_kornia:
                view_indicator = kornia.geometry.transform.warp_perspective(view_indicator.to('cuda:0'), proj_mat, reducedgrid_shape) # reducedgrid_shape=[480/4, 1440/4]
            else:
                view_indicator = warp_features_pytorch(view_indicator.to('cuda:0'), torch.linalg.inv(proj_mat), reducedgrid_shape, camera_orient)
            view_indicator_viz.append(view_indicator.detach().cpu())
            img_feature_viz.append(img_feature.detach().cpu())
            world_feature_viz.append(world_feature.detach().cpu())

            world_features.append(world_feature.to('cuda:0'))

        if self.avgpool:
            world_features = [x.unsqueeze(0) for x in world_features]
            world_features = torch.cat(world_features, dim=1)
            world_features = torch.mean(world_features, dim=1)    
            world_features = torch.cat([world_features] + [coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        else:
            world_features = torch.cat(world_features + [coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)

        if visualize:
            fig = plt.figure(figsize=(16,9))
            subplt0 = fig.add_subplot(111, title="concat_world_features")
            subplt0.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            plt.savefig(f"iall_bev_features.jpg")
            plt.close(fig)

            view_indicators = torch.cat(view_indicator_viz + [coord_map.repeat([B, 1, 1, 1]).cpu()], dim=1)

            fig = plt.figure(
                
            )
            subplt0 = fig.add_subplot(111, title="view_indicators")
            subplt0.imshow(torch.norm(view_indicators[0].detach(), dim=0).cpu().numpy())
            plt.savefig(f"iview_indicators.jpg")
            plt.close(fig)
            
        map_result = self.map_classifier(world_features.to('cuda:0'))
        map_result = F.interpolate(map_result, reducedgrid_shape, mode='bilinear')

        if visualize:
            fig = plt.figure(figsize=(16,9))
            subplt0 = fig.add_subplot(111, title="map_result")
            subplt0.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
            plt.savefig(f"imap_res.jpg")
            plt.close(fig)
        return map_result, imgs_result, (world_feature_viz, img_feature_viz, view_indicator_viz)

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()

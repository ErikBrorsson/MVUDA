# code from https://github.com/hou-yz/MVDet/tree/master
# modified by Erik Brorsson
import os

os.environ['OMP_NUM_THREADS'] = '1'
from PIL import Image, ImageDraw
import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from multiview_detector.datasets import frameDataset, Wildtrack, MultiviewX
from multiview_detector.utils.geometry import warp_features_pytorch

def _traget_transform(target, kernel):
    with torch.no_grad():
        target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
    return target

def get_start_end_line(line1, proj_mats, dataset_name):
    grid = line1.reshape((2, -1))
    grid_homo = torch.ones((3, grid.shape[1]))
    grid_homo[0:2, :] = grid
    grid_homo = grid_homo.unsqueeze(0)
    grid_persp = torch.bmm(proj_mats, grid_homo.to('cuda:0')).cpu().numpy().squeeze()
    z = grid_persp[2,:].copy()

    grid_persp = grid_persp / z *270/720

    # Image.fromarray((img * 255).astype('uint8'))
    # img = np.array(img)
    # for p in grid_persp.transpose():
    #     img = cv2.circle(img, (int(p[0]), int(p[1])), 1, (0,0,255), -1)
    grid_persp = grid_persp.reshape(3, len(z))

    start=None
    end = None
    for i in range(len(z)):
        if dataset_name == "wildtrack":
            if z[i] <= 0:
                continue
        elif dataset_name == "multiviewx":
            if z[i] >= 0:
                continue
        if start is None:
            start =  [int(x) for x in (grid_persp[0,i],grid_persp[1,i])]
        else:
            end = [int(x) for x in (grid_persp[0,i],grid_persp[1,i])]

    return start, end

def draw_bev_region_on_image(img, reducedgrid_shape, projm_img2bevred, dataset_name):
    bev_w, bev_h = reducedgrid_shape

    # x = torch.linspace(0, bev_h-1, bev_h)
    x = torch.linspace(0, bev_h-1, int(bev_h))
    # y = torch.linspace(0, bev_w-1, bev_w)
    y = torch.linspace(0, bev_w-1, int(bev_w))
    mesh = torch.meshgrid([x,y], indexing="xy")
    grid = torch.concat([mesh[0].unsqueeze(0), mesh[1].unsqueeze(0)])

    line1 = grid[:,:,-1]
    start, end = get_start_end_line(line1, projm_img2bevred, dataset_name)
    if start is not None and end is not None:
        img = cv2.line(img, tuple(start), tuple(end),(255,0,0))

    line1 = grid[:,:,0]
    start, end = get_start_end_line(line1, projm_img2bevred, dataset_name)
    if start is not None and end is not None:
        img = cv2.line(img, tuple(start), tuple(end),(255,0,0))

    line1 = grid[:,0,:]
    start, end = get_start_end_line(line1, projm_img2bevred, dataset_name)
    if start is not None and end is not None:
        img = cv2.line(img, tuple(start), tuple(end),(255,0,0))

    line1 = grid[:,-1,:]
    start, end = get_start_end_line(line1, projm_img2bevred, dataset_name)
    if start is not None and end is not None:
        img = cv2.line(img, tuple(start), tuple(end),(255,0,0))

    return img
def display_cam_layout(img, view_indicator_list, dataset, dataset_name):
    temp = 255*img
    temp = np.maximum(temp, 0)
    temp = np.minimum(255, temp)
    temp = temp.astype(np.uint8)
    # drawing = np.zeros((temp.shape[0]+2, temp.shape[1]+2, 3))
    # drawing[1:-1, 1:-1, :] = np.repeat(np.expand_dims(temp, axis=2), 3, axis=2)
    drawing = np.zeros((temp.shape[0], temp.shape[1], 3))
    drawing = np.repeat(np.expand_dims(temp, axis=2), 3, axis=2)
    color_list = [
        (0,255,0),
        (0,0,255),
        (255,127,0),
        (0,255,127),
        (127,0,255),
        (255,255,0),
        (255,0,255)
    ]

    extrinsics = dataset.base.extrinsic_matrices

    # for view_index, view in enumerate(view_indicator_list):
    for cam_i, cam in enumerate(extrinsics.keys()): 
        view = view_indicator_list[cam_i]      

        temp = (255*view[0][0,:,:].detach().cpu().numpy()).astype(np.uint8)
        # temp = cv2.resize(temp, (drawing.shape[1]+2, drawing.shape[0]+2), cv2.INTER_NEAREST)
        temp = cv2.resize(temp, (drawing.shape[1], drawing.shape[0]), cv2.INTER_NEAREST)

        contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = (contours[0] - 1)

        drawing=cv2.drawContours(drawing, contours, -1, color_list[cam], 1)
    # drawing = drawing[1:-1, 1:-1, :]
    print(drawing.shape)

    if dataset_name == "wildtrack":
        c_grid = [
            [170, 330],
            [930, 100],
            [700, 340],
            [120, 290],
            [520, 50],
            [10, 80],
            [350, 358],
        ]
    elif dataset_name == "multiviewx":
        c_grid = [
            [170, 450],
            [130, 60],
            [500, 80],
            [690, 475],
            [680, 280],
            [35, 250],
            [350, 358],
        ]

    for i, ext in extrinsics.items():       

        drawing = cv2.putText(drawing, f'C{i+1}', (c_grid[i][0], c_grid[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                              1, color_list[i], 2, cv2.LINE_AA)        
    
    drawing[0, :, 0] = 255
    drawing[0, :, 1] = 0
    drawing[0, :, 2] = 0
    drawing[-1, :, 0] = 255
    drawing[-1, :, 1] = 0
    drawing[-1, :, 2] = 0
    drawing[:, 0, 0] = 255
    drawing[:, 0, 1] = 0
    drawing[:, 0, 2] = 0
    drawing[:, -1, 0] = 255
    drawing[:, -1, 1] = 0
    drawing[:, -1, 2] = 0
    return drawing

def test(dataset_name='multiviewx', pred=False):
    if dataset_name == 'multiviewx':

        result_fpath = "/mnt/2024-09-30_15-13-10-806184/test_0/test_0.30_uda.txt"
        dataset = frameDataset(MultiviewX(os.path.expanduser('/data/MultiviewX'), cameras=[3,4,5]), False,
                               T.Compose([T.Resize([270, 480]), T.ToTensor(), ]))

    elif dataset_name == 'wildtrack':

        result_fpath = "/mnt/2024-11-01_13-18-59-051720/test_0/test_0.35.txt"

        dataset = frameDataset(Wildtrack('/data/Wildtrack'), False, T.Compose([T.Resize([270, 480]), T.ToTensor(), ]))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    grid_size = list(map(lambda x: x * 3, dataset.reducedgrid_shape))
    bbox_by_pos_cam = dataset.base.read_pom()
    results = np.loadtxt(result_fpath)

    video = cv2.VideoWriter(f'{dataset_name}_test_pred_100.avi', cv2.VideoWriter_fourcc(*"MJPG"), 2, (1580, 1060))
    view_indicator_list = []
    for index in tqdm.tqdm(range(len(dataset))):
        img_comb = np.zeros([1060, 1580, 3]).astype('uint8')
        map_res = np.zeros(dataset.reducedgrid_shape)
        imgs, map_gt, imgs_gt, frame,proj_mats, _, projm_img2bevred, projm_imgred2bevred, _, _ = dataset.__getitem__(index)

        if not view_indicator_list:
            for cam in range(dataset.num_cam):
                view_indicator = torch.ones_like(imgs[cam])
                view_indicator = warp_features_pytorch(view_indicator.to('cuda:0').unsqueeze(0), torch.linalg.inv(proj_mats[cam]).unsqueeze(0), dataset.reducedgrid_shape, dataset.camera_orient)
                view_indicator_list.append(view_indicator.detach().cpu())
        
        print("frame ", frame)

        if pred:
            res_map_grid = results[results[:, 0]*5 == frame, 1:]
            temp = np.zeros_like(res_map_grid)
            temp[:, 0] = res_map_grid[:, 1]
            temp[:, 1] = res_map_grid[:, 0]
            res_map_grid = temp.astype(np.int64)
        else:
            res_map_grid = results[results[:, 0] == frame, 1:]



        # print("res_map_grid ", res_map_grid)
        for ij in res_map_grid:
            # print("ij", ij)
            # j, i = (ij / dataset.grid_reduce).astype(int)
            i,j = (ij / dataset.grid_reduce).astype(int)
            # print("i, j", i, j)
            # if i>= res_map_grid.shape[0]:
            #     continue
            # if j >= res_map_grid.shape[1]:
            #     continue
            if dataset.base.indexing == 'xy':
                i, j = j, i
                map_res[i, j] = 1
            else:
                map_res[i, j] = 1
        map_res = _traget_transform(torch.from_numpy(map_res).unsqueeze(0).unsqueeze(0).float(),
                                    dataset.map_kernel)
        map_res = F.interpolate(map_res, grid_size).squeeze().numpy()

        map_res = display_cam_layout(map_res, view_indicator_list, dataset, dataset_name)


        map_res = np.uint8(map_res)
        # map_res = cv2.applyColorMap(map_res, cv2.COLORMAP_JET)
        map_res = cv2.putText(map_res, 'Ground Plane', (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (87, 59, 233), 2, cv2.LINE_AA)

        img_comb[580:580 + grid_size[0], 500:500 + grid_size[1]] = map_res
        # plt.imshow(cv2.cvtColor(img_comb.astype('uint8'), cv2.COLOR_BGR2RGB))
        # plt.show()

        print("res_map_grid: ", res_map_grid)
        res_posID = dataset.base.get_pos_from_worldgrid(res_map_grid.transpose())
        gt_map_grid = map_gt[0].nonzero().cpu().numpy() * dataset.grid_reduce
        gt_posID = dataset.base.get_pos_from_worldgrid(gt_map_grid.transpose())

        for cam_i, cam in enumerate(dataset.cameras):

            img = (imgs[cam_i].cpu().numpy().transpose([1, 2, 0]) * 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            proj_mat = torch.linalg.inv(projm_img2bevred[cam_i]).float().to('cuda:0').unsqueeze(0)
            img = draw_bev_region_on_image(img, dataset.reducedgrid_shape, proj_mat, dataset_name)

            for posID in res_posID:
                bbox = bbox_by_pos_cam[posID][cam]
                if bbox is not None:
                    bbox = tuple(map(lambda x: int(x / 4), bbox))
                    cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 1)
            pass

            img = cv2.putText(img, f'Camera {cam + 1}', (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (87, 59, 233), 2, cv2.LINE_AA)
            i, j = cam // 3, cam % 3
            img_comb[i * 290:i * 290 + 270, j * 500:j * 500 + 480] = img

        video.write(img_comb)
        # plt.imshow(cv2.cvtColor(img_comb.astype('uint8'), cv2.COLOR_BGR2RGB))
        # plt.show()
        pass
    video.release()


if __name__ == '__main__':
    test('multiviewx', pred=False)
    # test('wildtrack', pred=False)

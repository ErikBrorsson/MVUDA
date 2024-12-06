import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image
import torchvision
from multiview_detector.augmentation.homographyaugmentation import HomographyDataAugmentation




def display_cam_layout(img, view_indicator_list):
    temp = 255*img
    temp = np.maximum(temp, 0)
    temp = np.minimum(255, temp)
    temp = temp.astype(np.uint8)
    drawing = np.repeat(np.expand_dims(temp, axis=2), 3, axis=2)

    color_list = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (127,127,0),
        (0,127,127),
        (127,0,127),
        (255,255,0)
    ]

    for view_index, view in enumerate(view_indicator_list):
        temp = (255*view[0][0,:,:].detach().cpu().numpy()).astype(np.uint8)
        contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        drawing=cv2.drawContours(drawing, contours, -1, color_list[view_index], 1)
    return drawing

class Augmentation:
    def __init__(self, dropview=False, mvaug=False, rom3d=False,
                 grid_reduce=4, img_reduce=4, img_shape=[1080, 1920] , worldgrid_shape = [480, 1440]  # H,W; N_row,N_col
) -> None:
        self.dropview = dropview
        self.mvaug = mvaug
        self.rom3d = rom3d
        self.img_reduce = img_reduce
        self.img_shape = img_shape
        self.reducedgrid_shape = list(map(lambda x: int(x / grid_reduce), worldgrid_shape))

    def dropview_augment(self, imgs, map_label, imgs_labels, proj_mats):
        # imgs.shape = (1, 4, 3, 720, 1280) = (batch_size, n_cams, RGB, height, width)

        r = np.random.rand()
        if r >= 0.5: # drop one image with 50% probability if dropview is activated
            drop_indx = np.random.choice(np.arange(imgs.shape[1]))

            select_indx = [i for i in range(imgs.shape[1]) if i!=drop_indx]
            imgs = imgs[:, select_indx, :, :, :]
            imgs_labels_new = []
            proj_mats_new = []
            for i in select_indx:
                imgs_labels_new.append(imgs_labels[i])
                proj_mats_new.append(proj_mats[i])

            proj_mats = proj_mats_new
            imgs_labels = imgs_labels_new

        return imgs, map_label, imgs_labels, proj_mats
    
    def mvaug_augmentation(self, data, map_gt, imgs_gt, proj_mats_mvaug_features, weak=False):
        
        if weak:
            # half all parameters to raff
            scene_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(
                    degrees = 22, translate = (0.1, 0.1), scale = (0.9,1.1), shear = 5))
        else:
            scene_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(
                    degrees = 45, translate = (0.2, 0.2), scale = (0.8,1.2), shear = 10)) # parameters set according to MVAug's proposal

        # TODO of unknown reason, there is a slight difference between map_gt_aug_temp and map_gt_aug.
        # augment the map_label
        map_gt_aug_temp = scene_aug(torch.clone(map_gt))

        # TODO assuming batch_size 1
        map_gt_aug = torch.zeros_like(map_gt)
        foot_gt = map_gt[0, 0]
        foot_points = (foot_gt == 1).nonzero().float()
        temp = torch.zeros_like(foot_points)
        temp[:, 0] = foot_points[:, 1]
        temp[:, 1] = foot_points[:, 0]
        foot_points = temp
        foot_points_aug, pedestrian_ids = scene_aug.augment_gt_point_view_based(foot_points.detach().cpu(), gt_person_ids=None, filter_out_of_frame=True, frame_size=map_gt.shape[-2:])
        for pos in foot_points_aug:
            map_gt_aug[:,0,int(pos[1].item()), int(pos[0].item())] = 1

        data_aug = torch.zeros_like(data)
        proj_mat_aug_list = []
        img_gt_aug_list = []

        # proj_mat_aug_list_without_scene = []

        # loop over the images and apply augmentation
        for i, img_gt in enumerate(imgs_gt):
            if weak:
                # half all parameters to raff
                persp_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(
                        degrees = 22, translate = (0.1, 0.1), scale = (0.9,1.1), shear = 5))
            else:
                persp_aug = HomographyDataAugmentation(torchvision.transforms.RandomAffine(
                        degrees = 45, translate = (0.2, 0.2), scale = (0.8,1.2), shear = 10)) # parameters set according to MVAug's proposal
                
            img = torch.clone(data[0, i, :, :])
            # proj_mat_aug = proj_mats_mvaug[i] # bev-grid reduced -> image (720x1280)

            # augment the image
            data_aug[0, i] = persp_aug(img)

            # augment the image label
            # TODO assuming batch_size 1
            if img_gt is not None: # may be None in case of using pseudo-labels
                img_gt_aug = torch.zeros_like(img_gt)
                foot_gt = img_gt[0, 1]
                foot_points = (foot_gt == 1).nonzero().float()
                temp = torch.zeros_like(foot_points)
                temp[:, 0] = foot_points[:, 1]
                temp[:, 1] = foot_points[:, 0]
                foot_points = temp
                foot_points_aug, pedestrian_ids = persp_aug.augment_gt_point_view_based(foot_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=img_gt.shape[-2:])
                for pos in foot_points_aug:
                    img_gt_aug[:,1,int(pos[1].item()), int(pos[0].item())] = 1
                head_gt = img_gt[0, 0]
                head_points = (head_gt == 1).nonzero().float()
                temp = torch.zeros_like(head_points)
                temp[:, 0] = head_points[:, 1]
                temp[:, 1] = head_points[:, 0]
                head_points = temp
                head_points_aug, pedestrian_ids = persp_aug.augment_gt_point_view_based(head_points, gt_person_ids=None, filter_out_of_frame=True, frame_size=img_gt.shape[-2:])
                for pos in head_points_aug:
                    img_gt_aug[:,0,int(pos[1].item()), int(pos[0].item())] = 1
                img_gt_aug_list.append(img_gt_aug)
            else:
                img_gt_aug_list.append(None)

            # augment the projection matrix to account for persp aug
            temp = torch.tensor([int(x / self.img_reduce) for x in self.img_shape])
            proj_mat_aug_f = persp_aug.augment_homography_view_based(proj_mats_mvaug_features[i].float(), (temp).float()) # bev-grid reduced -> warped image (720x1280)

            temp = torch.tensor(self.reducedgrid_shape)
            proj_mat_aug_f = scene_aug.augment_homography_scene_based(proj_mat_aug_f, [int(x) for x in temp])

            proj_mat_aug_list.append(torch.linalg.inv(proj_mat_aug_f))

        return data_aug, map_gt_aug, img_gt_aug_list, proj_mat_aug_list

    def strong_augmentation(self, imgs, map_label, imgs_labels, proj_mats):
        """
        Args:
            imgs
            map_label
            imgs_labels
            proj_mats: input is mvaug standard (bev->image)
        
        returns:
            imgs
            maps_label
            imgs_labels
            proj_mats: output is MVDet standard (image->bev)
        """
        if self.mvaug:
            r = np.random.rand() # augment 50% of data with mvaug
            if r >= 0.5:
                imgs, map_label, imgs_labels, proj_mats = self.mvaug_augmentation(imgs, map_label, imgs_labels, proj_mats)
            else:
                proj_mats = [torch.linalg.inv(m) for m in proj_mats]
        else:
            proj_mats = [torch.linalg.inv(m) for m in proj_mats]

        if self.dropview:
            imgs, map_label, imgs_labels, proj_mats = self.dropview_augment(imgs, map_label, imgs_labels, proj_mats)
        return imgs, map_label, imgs_labels, proj_mats

    def weak_augmentation(self, imgs, map_label, imgs_labels, proj_mats):
        """
        Args:
            imgs
            map_label
            imgs_labels
            proj_mats: input is mvaug standard (bev->image)
        
        returns:
            imgs
            maps_label
            imgs_labels
            proj_mats: output is MVDet standard (image->bev)
        """
        # not using any augmentation for the teacher

        proj_mats = [torch.linalg.inv(m) for m in proj_mats]

        return imgs, map_label, imgs_labels, proj_mats


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, ema_model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0,
                 augmentation_module: Augmentation=Augmentation(), persp_sup=True, alpha_teacher=0.99, visualize_train=False):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha

        self.augmentation = augmentation_module
        self.persp_sup = persp_sup
        self.ema_model = ema_model
        self.alpha_teacher = alpha_teacher

        self.visualize_train = visualize_train

    def visualize_grid_and_bev(self, proj_mat, img, bev, f_name, foot_points, reducedgrid_shape, map_label=None):
        # bev_h = 360
        # bev_w = 120
        bev_w, bev_h = reducedgrid_shape

        # x = torch.linspace(0, bev_h-1, bev_h)
        x = torch.linspace(0, bev_h-1, int(bev_h/4))
        # y = torch.linspace(0, bev_w-1, bev_w)
        y = torch.linspace(0, bev_w-1, int(bev_w/4))
        mesh = torch.meshgrid([x,y], indexing="xy")
        grid = torch.concat([mesh[0].unsqueeze(0), mesh[1].unsqueeze(0)])
        grid = grid.reshape((2, -1))
        grid_homo = torch.ones((3, grid.shape[1]))
        grid_homo[0:2, :] = grid
        grid_homo = grid_homo.unsqueeze(0)
        grid_persp = torch.bmm(proj_mat.float().to('cuda:0'), grid_homo.to('cuda:0')).cpu().numpy().squeeze()

        # grid_persp = grid_persp[:, grid_persp[2, :] > 0] # remove all points that are behind the camera
        
        grid_persp = grid_persp / grid_persp[2, :]
        img = img.cpu().numpy().squeeze().transpose([1, 2, 0])
        img = Image.fromarray((img * 255).astype('uint8'))
        img = np.array(img)
        for p in grid_persp.transpose():
            img = cv2.circle(img, (int(p[0]), int(p[1])), 1, (0,0,255), -1)
        for p in foot_points:
            img = cv2.circle(img, (int(p[0]), int(p[1])), 4, (255,0,0), -1)

        fig = plt.figure(figsize=(16,9))
        subplt0 = fig.add_subplot(211, title="img")
        subplt1 = fig.add_subplot(212, title="bev_img")                
        subplt0.imshow(img)
        
        if map_label is not None:
            img0 = Image.fromarray((bev.cpu().numpy().squeeze().transpose([1, 2, 0]) * 255).astype('uint8'))
            bev_with_label = add_heatmap_to_image(map_label.cpu().detach().numpy().squeeze(), img0)
            subplt1.imshow(np.array(bev_with_label))
        else:
            subplt1.imshow(bev.cpu().numpy().squeeze().transpose([1, 2, 0]))

        plt.savefig(f_name)
        plt.close(fig)

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, map_gt, imgs_gt, _, proj_mats, proj_mats_mvaug, projm_img2bevred, projm_imgred2bevred, proj_mats_mvaug_features, dataset_name) in enumerate(data_loader):
            optimizer.zero_grad()

            data, map_gt, imgs_gt, proj_mats = self.augmentation.strong_augmentation(data, map_gt, imgs_gt, proj_mats_mvaug_features) 

            config_dict = data_loader.dataset.dicts[dataset_name[0]]
            map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.model(data, proj_mats, config_dict)
            
            t_f = time.time()
            t_forward += t_f - t_b
            loss = 0
            if self.persp_sup:
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    if not img_gt is None: # may be none after data augmentation
                        loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = loss / len([x for x in imgs_gt if x is not None]) * self.alpha

            loss += self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
            loss.backward()

            # update ema model
            alpha_teacher = self.alpha_teacher
            iteration = (epoch - 1) * len(data_loader.dataset) + batch_idx
            self.ema_model = self.update_ema_variables(self.ema_model, self.model, alpha_teacher=alpha_teacher, iteration=iteration)


            optimizer.step()
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                if self.visualize_train:
                    epoch_dir = os.path.join(self.logdir, f'epoch_{epoch}')
                    if not os.path.exists(epoch_dir):
                        os.mkdir(epoch_dir)

                    fig = plt.figure(dpi=800)
                    n_col = 5
                    subplt0 = fig.add_subplot(4, n_col, 1, title="student output")
                    subplt1 = fig.add_subplot(4, n_col, n_col*1 + 1, title="label")
                    subplt2 = fig.add_subplot(4, n_col, n_col*2 + 1, title="view indicators")
                    subplt3 = fig.add_subplot(4, n_col, n_col*3 + 1, title="world_features")

                    subplt4 = fig.add_subplot(4, n_col,  2, title="img_feature_i")
                    subplt5 = fig.add_subplot(4, n_col, n_col + 2, title="world_feature_i")
                    subplt6 = fig.add_subplot(4, n_col,  n_col*2 + 2, title="img_feature_i")
                    subplt7 = fig.add_subplot(4, n_col, n_col*3 + 2, title="world_feature_i")

                    subplt8 = fig.add_subplot(4, n_col,  3, title="img_feature_i")
                    subplt9 = fig.add_subplot(4, n_col, n_col + 3, title="world_feature_i")
                    subplt10 = fig.add_subplot(4, n_col,  n_col*2 + 3, title="img_feature_i")
                    subplt11 = fig.add_subplot(4, n_col, n_col*3 + 3, title="world_feature_i")

                    subplt12 = fig.add_subplot(4, n_col,  4, title="img_feature_i")
                    subplt13 = fig.add_subplot(4, n_col, n_col + 4, title="world_feature_i")
                    subplt14 = fig.add_subplot(4, n_col,  n_col*2 + 4, title="img_feature_i")
                    subplt15 = fig.add_subplot(4, n_col, n_col*3 + 4, title="world_feature_i")

                    subplt16 = fig.add_subplot(4, n_col,  5, title="img_feature_i")
                    subplt17 = fig.add_subplot(4, n_col, n_col + 5, title="world_feature_i")

                    subplt18 = fig.add_subplot(4, n_col, 2*n_col + 5, title="example_image")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    img0 = (255*self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])).astype(np.uint8)


                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)
                    subplt18.imshow(img0)

                    if len(world_features) >= 1:
                        img_feature_i = torch.norm(img_features[0][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[0][0].detach(), dim=0).cpu().numpy()
                        subplt4.imshow(img_feature_i)
                        subplt5.imshow(w_feature_i)

                    if len(world_features) >= 2:
                        img_feature_i = torch.norm(img_features[1][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[1][0].detach(), dim=0).cpu().numpy()
                        subplt6.imshow(img_feature_i)
                        subplt7.imshow(w_feature_i)

                    if len(world_features) >= 3:
                        img_feature_i = torch.norm(img_features[2][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[2][0].detach(), dim=0).cpu().numpy()
                        subplt8.imshow(img_feature_i)
                        subplt9.imshow(w_feature_i)

                    if len(world_features) >= 4:
                        img_feature_i = torch.norm(img_features[3][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[3][0].detach(), dim=0).cpu().numpy()
                        subplt10.imshow(img_feature_i)
                        subplt11.imshow(w_feature_i)

                    if len(world_features) >= 5:
                        img_feature_i = torch.norm(img_features[4][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[4][0].detach(), dim=0).cpu().numpy()
                        subplt12.imshow(img_feature_i)
                        subplt13.imshow(w_feature_i)

                    if len(world_features) >= 6:
                        img_feature_i = torch.norm(img_features[5][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[5][0].detach(), dim=0).cpu().numpy()
                        subplt14.imshow(img_feature_i)
                        subplt15.imshow(w_feature_i)

                    if len(world_features) >= 7:
                        img_feature_i = torch.norm(img_features[6][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[6][0].detach(), dim=0).cpu().numpy()
                        subplt16.imshow(img_feature_i)
                        subplt17.imshow(w_feature_i)

                    plt.savefig(os.path.join(epoch_dir, f'train_source_features_{batch_idx}.jpg'))
                    plt.close(fig)

                    fig = plt.figure(dpi=500)
                    n_col = 1
                    subplt0 = fig.add_subplot(4, n_col, 1, title="student output")
                    subplt1 = fig.add_subplot(4, n_col, n_col*1 + 1, title="label")
                    subplt2 = fig.add_subplot(4, n_col, n_col*2 + 1, title="view indicators")
                    subplt3 = fig.add_subplot(4, n_col, n_col*3 + 1, title="world_features")

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)
                    plt.savefig(os.path.join(epoch_dir, f'train_source_{batch_idx}.jpg'))
                    plt.close(fig)


                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
                      'prec: {:.1f}%, recall: {:.1f}%, Time: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / (batch_idx + 1), t_backward / (batch_idx + 1), map_res.max()))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, persp_map=False, test_time_aug=False, varying_cls_thres=False):
        if varying_cls_thres:
            cls_thres_array = np.arange(0.05, 0.95, 0.05)
            all_res_list = {str(x): [] for x in cls_thres_array}

            self.model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = {str(x): [] for x in cls_thres_array}
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]

                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.model(data, proj_mats, config_dict)
                if res_fpath is not None:
                    for cls_thres in cls_thres_array:
                        map_grid_res = map_res.detach().cpu().squeeze()
                        v_s = map_grid_res[map_grid_res > cls_thres].unsqueeze(1)
                        grid_ij = (map_grid_res > cls_thres).nonzero()
                        if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                            grid_xy = grid_ij[:, [1, 0]]
                        else:
                            grid_xy = grid_ij
                        all_res_list[str(cls_thres)].append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                        loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(411, title="output")
                    subplt1 = fig.add_subplot(412, title="target")
                    subplt2 = fig.add_subplot(413, title="view indicators")
                    subplt3 = fig.add_subplot(414, title="world features")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)

                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                    heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'cam1_foot_{batch_idx}.jpg'))


            moda = 0
            moda_list = []
            precision_list = []
            recall_list = []
            modp_list = []
            moda_04 = 0
            modp_04 = 0
            precision_04 = 0
            recall_04 = 0
            if res_fpath is not None:
                for i, cls_thres in enumerate(cls_thres_array):
                    all_res_list_thres = all_res_list[str(cls_thres)]
                    all_res_list_thres = torch.cat(all_res_list_thres, dim=0)
                    np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list_thres.numpy(), '%.8f')
                    res_list = []
                    for frame in np.unique(all_res_list_thres[:, 0]):
                        res = all_res_list_thres[all_res_list_thres[:, 0] == frame, :]
                        positions, scores = res[:, 1:3], res[:, 3]
                        ids, count = nms(positions, scores, 20, np.inf)
                        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                    np.savetxt(res_fpath, res_list, '%d')

                    recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                                data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)

                    moda_list.append(moda)
                    modp_list.append(modp)
                    precision_list.append(precision)
                    recall_list.append(recall)

                    if cls_thres == 0.4:
                        moda_04 = moda
                        modp_04 = modp
                        precision_04 = precision
                        recall_04 = recall


                max_indx = np.argmax(moda_list)
                moda = moda_list[max_indx]
                modp = modp_list[max_indx]
                precision = precision_list[max_indx]
                recall = recall_list[max_indx]
                max_cls_thres = cls_thres_array[max_indx]
                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%, cls_thres: {:.2f}'.
                        format(moda, modp, precision, recall, max_cls_thres))

            t1 = time.time()
            t_epoch = t1 - t0
            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, max_cls_thres), (moda_04, modp_04, precision_04, recall_04, 0.4)


        else:
            # self.model.configure_model_for_dataset(data_loader.dataset)
            self.model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = []
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                if test_time_aug:
                    data, map_gt, imgs_gt, proj_mats = self.augmentation.strong_augmentation(data, map_gt, imgs_gt, proj_mats)

                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]
                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.model(data, proj_mats, config_dict, visualize=True)
                if res_fpath is not None:
                    map_grid_res = map_res.detach().cpu().squeeze()
                    v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                    grid_ij = (map_grid_res > self.cls_thres).nonzero()
                    if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                        grid_xy = grid_ij[:, [1, 0]]
                    else:
                        grid_xy = grid_ij
                    all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))
                    
                    # do NMS and create actual preditions (post nms)
                    temp = map_grid_res
                    scores = temp[temp > self.cls_thres]
                    positions = (temp > self.cls_thres).nonzero().float()
                    # if data_loader.dataset.base.indexing == 'xy':
                    #     positions = positions[:, [1, 0]]
                    # else:
                    #     positions = positions
                    if not torch.numel(positions) == 0:
                        ids, count = nms(positions.float(), scores, 20 /  data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, np.inf)
                        positions = positions[ids[:count], :]
                        scores = scores[ids[:count]]
                    map_pseudo_label = torch.zeros_like(map_res)
                    for pos in positions:
                        map_pseudo_label[:,:,int(pos[0].item()), int(pos[1].item())] = 1

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    if img_gt is not None:
                        loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                    loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > self.cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    if persp_map:
                        # initialize perspective view bev predictions
                        map_res_from_perspective = torch.zeros_like(map_res).detach().cpu()
                        map_res_from_perspective_scores = -1e8*torch.ones_like(map_res).detach().cpu()

                    for cam_indx, _ in enumerate(imgs_res):
                        # cam_number = data_loader.dataset.cameras[cam_indx]
                        cam_number = cam_indx

                        pred_view1 = imgs_res[cam_indx]
                        heatmap0_head = pred_view1[0, 0].detach().cpu().numpy().squeeze()
                        heatmap0_foot = pred_view1[0, 1].detach().cpu().numpy().squeeze()

                        if persp_map: # TODO doesn't work if n_views_train > n_views_test, since we don't have the projection matrices for the duplicate views
                            foot_coords = (heatmap0_foot > self.cls_thres).nonzero()
                            foot_scores = heatmap0_foot[heatmap0_foot > self.cls_thres]
                            if not foot_coords[0].size == 0:
                                # temp = np.zeros((2, len(foot_coords[0])))
                                temp = np.ones((3, len(foot_coords[0])))
                                temp[0,:] = foot_coords[1]
                                temp[1,:] = foot_coords[0]

                                world_grid = self.model.proj_mats[cam_number] @ temp  
                                world_grid = (world_grid/world_grid[2,:]).detach().cpu().numpy()

                                for coord_indx, p in enumerate(world_grid.transpose()):
                                    if p[0]>=0 and p[1] >= 0 and p[0]<map_res_from_perspective.shape[3] and p[1]<map_res_from_perspective.shape[2]:
                                        map_res_from_perspective[0, 0, int(p[1]), int(p[0])] = 1

                                        prev_val = map_res_from_perspective_scores[0, 0, int(p[1]), int(p[0])]
                                        map_res_from_perspective_scores[0, 0, int(p[1]), int(p[0])] = max(float(foot_scores[coord_indx]), prev_val.item())
                                        # print(p)
                            else:
                                print("No preds from perspective view: ", cam_number+1)

                        img0 = self.denormalize(data[0, cam_indx]).cpu().numpy().squeeze().transpose([1, 2, 0])
                        img0 = Image.fromarray((img0 * 255).astype('uint8'))
                        # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                        # head_cam_result.save(os.path.join(self.logdir, f'output_cam{cam_number+ 1}_head_{batch_idx}.jpg'))
                        foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                        foot_cam_result.save(os.path.join(self.logdir, f'output_cam{cam_number+ 1}_foot_{batch_idx}.jpg'))



                    if persp_map:
                        # do NMS and create actual preditions (post nms)
                        perspective_all_res_list = []
                        temp = map_res_from_perspective_scores.squeeze()
                        scores = temp[temp > self.cls_thres].unsqueeze(1)
                        positions = (temp > self.cls_thres).nonzero().float()
                        if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                            positions = positions[:, [1, 0]]
                        else:
                            positions = positions

                        perspective_all_res_list.append(torch.cat([torch.ones_like(scores) * frame, positions.float() *
                                                    data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, scores], dim=1))

                        scores = scores.squeeze()
                        if not torch.numel(positions) == 0:
                            ids, count = nms(positions.float(), scores, 20 /  data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, np.inf)
                            positions = positions[ids[:count], :]
                            scores = scores[ids[:count]]
                        map_perspective_pseudo_label = torch.zeros_like(map_res)
                        for pos in positions:
                            map_perspective_pseudo_label[:,:,int(pos[0].item()), int(pos[1].item())] = 1


                        fig = plt.figure()
                        subplt0 = fig.add_subplot(321, title="scores")
                        subplt1 = fig.add_subplot(322, title="prediction")
                        subplt2 = fig.add_subplot(323, title="persp. scores")
                        subplt3 = fig.add_subplot(324, title="persp. prediction")
                        subplt4 = fig.add_subplot(325, title="label")
                        subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
                        subplt1.imshow(self.criterion._traget_transform(map_res, map_pseudo_label, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                    .cpu().detach().numpy().squeeze())
                        subplt2.imshow(self.criterion._traget_transform(map_res, map_res_from_perspective, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                    .cpu().detach().numpy().squeeze())
                        subplt3.imshow(self.criterion._traget_transform(map_res, map_perspective_pseudo_label, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                    .cpu().detach().numpy().squeeze())
                        subplt4.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                    .cpu().detach().numpy().squeeze())
                        plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                        plt.close(fig)

                    else:
                        fig = plt.figure()
                        subplt0 = fig.add_subplot(411, title="output")
                        subplt1 = fig.add_subplot(412, title="target")
                        subplt2 = fig.add_subplot(413, title="view indicators")
                        subplt3 = fig.add_subplot(414, title="world features")

                        map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                        label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                    .cpu().detach().numpy().squeeze(), view_indicator_list)
                        all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                        all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                        subplt0.imshow(map_res_view)
                        subplt1.imshow(label_view)
                        subplt2.imshow(all_views)
                        subplt3.imshow(all_world_features)

                        plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                        plt.close(fig)



            t1 = time.time()
            t_epoch = t1 - t0

            # evaluate perspective view preds
            if persp_map:
                moda = 0
                if res_fpath is not None:
                    res_fpath = os.path.abspath(os.path.dirname(res_fpath)) + '/test_perspective.txt'
                    all_res_list = torch.cat(perspective_all_res_list, dim=0)
                    np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res_perspective.txt', all_res_list.numpy(), '%.8f')
                    res_list = []
                    for frame in np.unique(all_res_list[:, 0]):
                        res = all_res_list[all_res_list[:, 0] == frame, :]
                        positions, scores = res[:, 1:3], res[:, 3]
                        ids, count = nms(positions, scores, 20, np.inf)
                        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                    np.savetxt(res_fpath, res_list, '%d')

                    recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                            data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)

                    # If you want to use the unofiicial python evaluation tool for convenient purposes.
                    # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                    #                                             data_loader.dataset.base.__name__)
                    print("########### perspective results ####################")
                    print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                        format(moda, modp, precision, recall))
                    

            moda = 0
            if res_fpath is not None:
                all_res_list = torch.cat(all_res_list, dim=0)
                np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
                res_list = []
                for frame in np.unique(all_res_list[:, 0]):
                    res = all_res_list[all_res_list[:, 0] == frame, :]
                    positions, scores = res[:, 1:3], res[:, 3]
                    ids, count = nms(positions, scores, 20, np.inf)
                    res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                np.savetxt(res_fpath, res_list, '%d')

                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)


                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                    format(moda, modp, precision, recall))

            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, self.cls_thres), (moda, modp, precision, recall, self.cls_thres)

    def test_ema(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, varying_cls_thres=False):
        if varying_cls_thres:
            cls_thres_array = np.arange(0.05, 0.95, 0.05)
            all_res_list = {str(x): [] for x in cls_thres_array}

            self.ema_model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = {str(x): [] for x in cls_thres_array}
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]

                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.ema_model(data, proj_mats, config_dict)
                if res_fpath is not None:
                    for cls_thres in cls_thres_array:
                        map_grid_res = map_res.detach().cpu().squeeze()
                        v_s = map_grid_res[map_grid_res > cls_thres].unsqueeze(1)
                        grid_ij = (map_grid_res > cls_thres).nonzero()
                        if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                            grid_xy = grid_ij[:, [1, 0]]
                        else:
                            grid_xy = grid_ij
                        all_res_list[str(cls_thres)].append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                        loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    fig = plt.figure(dpi=500)
                    subplt0 = fig.add_subplot(411, title="output")
                    subplt1 = fig.add_subplot(412, title="target")
                    subplt2 = fig.add_subplot(413, title="view indicators")
                    subplt3 = fig.add_subplot(414, title="world features")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)

                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                    heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'cam1_foot_{batch_idx}.jpg'))

            moda = 0
            moda_04 = 0
            modp_04 = 0
            precision_04 = 0
            recall_04 = 0
            moda_list = []
            precision_list = []
            recall_list = []
            modp_list = []
            if res_fpath is not None:
                for i, cls_thres in enumerate(cls_thres_array):
                    all_res_list_thres = all_res_list[str(cls_thres)]
                    all_res_list_thres = torch.cat(all_res_list_thres, dim=0)
                    np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list_thres.numpy(), '%.8f')
                    res_list = []
                    for frame in np.unique(all_res_list_thres[:, 0]):
                        res = all_res_list_thres[all_res_list_thres[:, 0] == frame, :]
                        positions, scores = res[:, 1:3], res[:, 3]
                        ids, count = nms(positions, scores, 20, np.inf)
                        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                    np.savetxt(res_fpath, res_list, '%d')

                    recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                                data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)

                    moda_list.append(moda)
                    modp_list.append(modp)
                    precision_list.append(precision)
                    recall_list.append(recall)

                    if cls_thres == 0.4:
                        moda_04 = moda
                        modp_04 = modp
                        precision_04 = precision
                        recall_04 = recall

                max_indx = np.argmax(moda_list)
                moda = moda_list[max_indx]
                modp = modp_list[max_indx]
                precision = precision_list[max_indx]
                recall = recall_list[max_indx]
                max_cls_thres = cls_thres_array[max_indx]
                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%, cls_thres: {:.2f}'.
                        format(moda, modp, precision, recall, max_cls_thres))

            t1 = time.time()
            t_epoch = t1 - t0
            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, max_cls_thres), (moda_04, modp_04, precision_04, recall_04, 0.4)

        else:
            self.ema_model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = []
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]
                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.ema_model(data, proj_mats, config_dict)
                if res_fpath is not None:
                    map_grid_res = map_res.detach().cpu().squeeze()
                    v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                    grid_ij = (map_grid_res > self.cls_thres).nonzero()
                    if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                        grid_xy = grid_ij[:, [1, 0]]
                    else:
                        grid_xy = grid_ij
                    all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                    loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > self.cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(411, title="output")
                    subplt1 = fig.add_subplot(412, title="target")
                    subplt2 = fig.add_subplot(413, title="view indicators")
                    subplt3 = fig.add_subplot(414, title="world features")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)

                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                    # visualizing the heatmap for per-view estimation
                    # heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
                    heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                    # head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'cam1_foot_{batch_idx}.jpg'))

            t1 = time.time()
            t_epoch = t1 - t0

            moda = 0
            if res_fpath is not None:
                all_res_list = torch.cat(all_res_list, dim=0)
                np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
                res_list = []
                for frame in np.unique(all_res_list[:, 0]):
                    res = all_res_list[all_res_list[:, 0] == frame, :]
                    positions, scores = res[:, 1:3], res[:, 3]
                    ids, count = nms(positions, scores, 20, np.inf)
                    res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                np.savetxt(res_fpath, res_list, '%d')

                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)

                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                    format(moda, modp, precision, recall))

            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, self.cls_thres), (moda, modp, precision, recall, self.cls_thres)
    
    @staticmethod
    def update_ema_variables(ema_model, model, alpha_teacher, iteration):

        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
    

class UDATrainer(BaseTrainer):
    def __init__(self, model, ema_model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0,
                 visualize_train=False, target_cameras=None, alpha_teacher=0.99,
                 augmentation_module: Augmentation=Augmentation(),
                 persp_sup=True, uda_persp_sup=False,
                 uda_nms_th=20, augmentation_uda: Augmentation=Augmentation(), max_pseudo=False, max_pseudo_th=11):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.teacher = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha

        self.visualize_train = visualize_train
        self.ema_model = ema_model

        assert target_cameras is not None, "target_cameras must be set in UDATrainer"
        self.target_cameras = target_cameras

        self.alpha_teacher = alpha_teacher

        self.augmentation = augmentation_module
        self.augmentation_uda = augmentation_uda

        self.uda_persp_sup = uda_persp_sup
        self.persp_sup = persp_sup
        self.uda_nms_th = uda_nms_th

        self.max_pseudo = max_pseudo
        self.k_size = max_pseudo_th

    def train(self, epoch, data_loader, data_loader_target, optimizer, log_interval=100, cyclic_scheduler=None, target_weight=0., pseudo_label_th=0.2):

        self.model.train()
        self.ema_model.train()
        losses = 0
        losses_target = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, ((data, map_gt, imgs_gt, _, _, _, _, _, proj_mats_mvaug_features_src, dataset_name),
                        (data_target, map_gt_target, imgs_gt_target, _, _, data_no3drom_target, _, _, proj_mats_mvaug_features_trg, dataset_name_trg)) in enumerate(zip(data_loader, data_loader_target)):

            img_gt_shape = imgs_gt[0].shape


            # train on source data
            optimizer.zero_grad()
            data, map_gt, imgs_gt, proj_mats_source = self.augmentation.strong_augmentation(data, map_gt, imgs_gt, proj_mats_mvaug_features_src)

            config_dict = data_loader.dataset.dicts[dataset_name[0]]
            map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.model(data, proj_mats_source, config_dict)
            t_f = time.time()
            t_forward += t_f - t_b
            loss = 0
            if self.persp_sup:
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    if img_gt is not None: # may be none if using dropview augmentation
                        loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = loss / len([x for x in imgs_gt if x is not None]) * self.alpha

            loss += self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
            loss.backward()
            # optimizer.step()
            losses += loss.item()

            # logging
            map_res_max = map_res.max()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            if (batch_idx + 1) % log_interval == 0:
                if self.visualize_train:
                    epoch_dir = os.path.join(self.logdir, f'epoch_{epoch}')
                    if not os.path.exists(epoch_dir):
                        os.mkdir(epoch_dir)

                    fig = plt.figure(dpi=800)
                    n_col = 5
                    subplt0 = fig.add_subplot(4, n_col, 1, title="student output")
                    subplt1 = fig.add_subplot(4, n_col, n_col*1 + 1, title="label")
                    subplt2 = fig.add_subplot(4, n_col, n_col*2 + 1, title="view indicators")
                    subplt3 = fig.add_subplot(4, n_col, n_col*3 + 1, title="world_features")

                    subplt4 = fig.add_subplot(4, n_col,  2, title="img_feature_i")
                    subplt5 = fig.add_subplot(4, n_col, n_col + 2, title="world_feature_i")
                    subplt6 = fig.add_subplot(4, n_col,  n_col*2 + 2, title="img_feature_i")
                    subplt7 = fig.add_subplot(4, n_col, n_col*3 + 2, title="world_feature_i")

                    subplt8 = fig.add_subplot(4, n_col,  3, title="img_feature_i")
                    subplt9 = fig.add_subplot(4, n_col, n_col + 3, title="world_feature_i")
                    subplt10 = fig.add_subplot(4, n_col,  n_col*2 + 3, title="img_feature_i")
                    subplt11 = fig.add_subplot(4, n_col, n_col*3 + 3, title="world_feature_i")

                    subplt12 = fig.add_subplot(4, n_col,  4, title="img_feature_i")
                    subplt13 = fig.add_subplot(4, n_col, n_col + 4, title="world_feature_i")
                    subplt14 = fig.add_subplot(4, n_col,  n_col*2 + 4, title="img_feature_i")
                    subplt15 = fig.add_subplot(4, n_col, n_col*3 + 4, title="world_feature_i")

                    subplt16 = fig.add_subplot(4, n_col,  5, title="img_feature_i")
                    subplt17 = fig.add_subplot(4, n_col, n_col + 5, title="world_feature_i")

                    subplt18 = fig.add_subplot(4, n_col, 2*n_col + 5, title="example_image")

                    img0 = (255*self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])).astype(np.uint8)


                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)
                    subplt18.imshow(img0)

                    if len(world_features) >= 1:
                        img_feature_i = torch.norm(img_features[0][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[0][0].detach(), dim=0).cpu().numpy()
                        subplt4.imshow(img_feature_i)
                        subplt5.imshow(w_feature_i)

                    if len(world_features) >= 2:
                        img_feature_i = torch.norm(img_features[1][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[1][0].detach(), dim=0).cpu().numpy()
                        subplt6.imshow(img_feature_i)
                        subplt7.imshow(w_feature_i)

                    if len(world_features) >= 3:
                        img_feature_i = torch.norm(img_features[2][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[2][0].detach(), dim=0).cpu().numpy()
                        subplt8.imshow(img_feature_i)
                        subplt9.imshow(w_feature_i)

                    if len(world_features) >= 4:
                        img_feature_i = torch.norm(img_features[3][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[3][0].detach(), dim=0).cpu().numpy()
                        subplt10.imshow(img_feature_i)
                        subplt11.imshow(w_feature_i)

                    if len(world_features) >= 5:
                        img_feature_i = torch.norm(img_features[4][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[4][0].detach(), dim=0).cpu().numpy()
                        subplt12.imshow(img_feature_i)
                        subplt13.imshow(w_feature_i)

                    if len(world_features) >= 6:
                        img_feature_i = torch.norm(img_features[5][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[5][0].detach(), dim=0).cpu().numpy()
                        subplt14.imshow(img_feature_i)
                        subplt15.imshow(w_feature_i)

                    if len(world_features) >= 7:
                        img_feature_i = torch.norm(img_features[6][0].detach(), dim=0).cpu().numpy()
                        w_feature_i = torch.norm(world_features[6][0].detach(), dim=0).cpu().numpy()
                        subplt16.imshow(img_feature_i)
                        subplt17.imshow(w_feature_i)

                    plt.savefig(os.path.join(epoch_dir, f'train_source_features_{batch_idx}.jpg'))
                    plt.close(fig)

                    fig = plt.figure(dpi=500)
                    n_col = 1
                    subplt0 = fig.add_subplot(4, n_col, 1, title="student output")
                    subplt1 = fig.add_subplot(4, n_col, n_col*1 + 1, title="label")
                    subplt2 = fig.add_subplot(4, n_col, n_col*2 + 1, title="view indicators")
                    subplt3 = fig.add_subplot(4, n_col, n_col*3 + 1, title="world_features")

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)
                    plt.savefig(os.path.join(epoch_dir, f'train_source_{batch_idx}.jpg'))
                    plt.close(fig)

            del imgs_res, imgs_gt, map_res, map_gt, data

            # train on target data
            # optimizer.zero_grad()

            # create bev pseudo-labels
            if target_weight != 0:
                with torch.no_grad():
                    if self.augmentation_uda.rom3d: # the teacher is fed data that is not augmneted with 3DROM
                        data_teacher, _, _, proj_mats_teacher = self.augmentation_uda.weak_augmentation(data_no3drom_target, map_gt_target, imgs_gt_target, proj_mats_mvaug_features_trg)
                    else:
                        data_teacher, _, _, proj_mats_teacher = self.augmentation_uda.weak_augmentation(data_target, map_gt_target, imgs_gt_target, proj_mats_mvaug_features_trg)
                    
                    config_dict = data_loader_target.dataset.dicts[dataset_name_trg[0]]

                    if self.alpha_teacher == 0: # if alpha_teacher == 0, use student model for pseudo-labelling
                        map_pred_teacher, imgs_teacher_pred, (world_features, img_features, view_indicator_list_teacher)  = self.model(data_teacher, proj_mats_teacher, config_dict)
                    else:
                        if self.alpha_teacher == 1.0: # alpha_teacher == 1.0 means that the pretrained model should be used as is. use .eval() to avoid batch_norm updates
                            self.ema_model.eval()
                        map_pred_teacher, imgs_teacher_pred, (world_features, img_features, view_indicator_list_teacher)  = self.ema_model(data_teacher, proj_mats_teacher, config_dict)
                temp = map_pred_teacher.detach().cpu().squeeze()

                if self.max_pseudo:
                    k_size = self.k_size
                    pad_size = int(k_size/2)

                    map_pred_teacher_pooled = torch.max_pool2d(map_pred_teacher, k_size, stride=1,padding=(pad_size, pad_size))
                    maximas = map_pred_teacher_pooled == map_pred_teacher
                    maximas = maximas & (map_pred_teacher >= pseudo_label_th)
                    map_pseudo_label = maximas.float()
                    scores = map_pred_teacher[maximas].cpu()
                    maximas = maximas.float()
                    positions = (maximas.squeeze() > 0).nonzero().float().cpu()
                else:
                    scores = temp[temp > pseudo_label_th]
                    positions = (temp > pseudo_label_th).nonzero().float()
                    if not torch.numel(positions) == 0:
                        ids, count = nms(positions.float(), scores, self.uda_nms_th / data_loader_target.dataset.dicts[dataset_name_trg[0]]['base'].grid_reduce, np.inf)
                        positions = positions[ids[:count], :]
                        scores = scores[ids[:count]]
                    map_pseudo_label = torch.zeros_like(map_pred_teacher)
                    for pos in positions:
                        map_pseudo_label[:,:,int(pos[0].item()), int(pos[1].item())] = 1
                

                # create perspective view pseudo-labels by projecting bev pseudo-labels into camera
                if data_loader_target.dataset.dicts[dataset_name_trg[0]]["base"].indexing == 'xy':
                    positions = positions[:, [1, 0]]
                else:
                    positions = positions
                imgs_pseudo_labels = []
                if self.uda_persp_sup:
                    for cam in self.target_cameras:
                        img_pseudo_label = torch.zeros(img_gt_shape)

                        for grid_pos in positions:
                            pos = data_loader_target.dataset.dicts[dataset_name_trg[0]]["base"].base.get_pos_from_worldgrid(grid_pos * data_loader_target.dataset.dicts[dataset_name_trg[0]]["base"].grid_reduce)
                            bbox = data_loader_target.dataset.dicts[dataset_name_trg[0]]["base"].base.bbox_by_pos_cam[pos.item()][cam]
                            if bbox is None:
                                continue                    
                            foot_2d = [int((bbox[0] + bbox[2]) / 2), int(bbox[3])]
                            head_2d = [int((bbox[0] + bbox[2]) / 2), int(bbox[1])]
                            img_pseudo_label[:,0,head_2d[1], head_2d[0]] = 1
                            img_pseudo_label[:,1,foot_2d[1],foot_2d[0]] = 1

                        imgs_pseudo_labels.append(img_pseudo_label)
                else:
                    for cam in self.target_cameras:
                        imgs_pseudo_labels.append(None)

                # apply augmentation to target images and pseudo-labels prior to student training
                map_pseudo_label_unaug = torch.clone(map_pseudo_label)


                data_student, map_pseudo_label, imgs_pseudo_labels, proj_mats_student = self.augmentation_uda.strong_augmentation(data_target,
                                                                                                            map_pseudo_label, imgs_pseudo_labels, proj_mats_mvaug_features_trg)
                
                # student predict and compute loss
                config_dict = data_loader_target.dataset.dicts[dataset_name_trg[0]]
                map_res_target, imgs_res_target, (world_features, img_features, view_indicator_list)  = self.model(data_student, proj_mats_student, config_dict)
                loss = 0
                for img_res_target, img_pseudo_label in zip(imgs_res_target, imgs_pseudo_labels):
                    if not img_pseudo_label is None:
                        loss += self.criterion(img_res_target, img_pseudo_label.to(img_res_target.device), data_loader_target.dataset.dicts[dataset_name_trg[0]]["base"].img_kernel)
                if len([x for x in imgs_pseudo_labels if x is not None]) > 0:
                    loss = loss / len([x for x in imgs_pseudo_labels if x is not None]) * self.alpha

                loss += self.criterion(map_res_target, map_pseudo_label.to(map_res_target.device), data_loader_target.dataset.dicts[dataset_name_trg[0]]["base"].map_kernel)

            # update student
                loss = loss * target_weight # weight the target loss with a weight that grows with increased confidence of pseudo-labels
                loss.backward()
                losses_target += loss.item()

            # update ema model
            alpha_teacher = self.alpha_teacher
            iteration = (epoch - 1) * len(data_loader.dataset) + batch_idx
            if alpha_teacher != 1.0: # alpha_teacher==1 means no update
                self.ema_model = self.update_ema_variables(self.ema_model, self.model, alpha_teacher=alpha_teacher, iteration=iteration)


            optimizer.step()
            # update learning rate
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()

            # logging
            t_b = time.time()
            t_backward += t_b - t_f
            if (batch_idx + 1) % log_interval == 0:
                if target_weight != 0:
                    if self.visualize_train:

                        fig = plt.figure(dpi=800)
                        n_col = 5
                        subplt0 = fig.add_subplot(4, n_col, 1, title="student output")
                        subplt1 = fig.add_subplot(4, n_col, n_col*1 + 1, title="label")
                        subplt2 = fig.add_subplot(4, n_col, n_col*2 + 1, title="pseudo-label")
                        subplt3 = fig.add_subplot(4, n_col, n_col*3 + 1, title="teacher output")

                        subplt4 = fig.add_subplot(4, n_col,  2, title="img_feature_i")
                        subplt5 = fig.add_subplot(4, n_col, n_col + 2, title="world_feature_i")
                        subplt6 = fig.add_subplot(4, n_col,  n_col*2 + 2, title="img_feature_i")
                        subplt7 = fig.add_subplot(4, n_col, n_col*3 + 2, title="world_feature_i")

                        subplt8 = fig.add_subplot(4, n_col,  3, title="img_feature_i")
                        subplt9 = fig.add_subplot(4, n_col, n_col + 3, title="world_feature_i")
                        subplt10 = fig.add_subplot(4, n_col,  n_col*2 + 3, title="img_feature_i")
                        subplt11 = fig.add_subplot(4, n_col, n_col*3 + 3, title="world_feature_i")

                        subplt12 = fig.add_subplot(4, n_col,  4, title="img_feature_i")
                        subplt13 = fig.add_subplot(4, n_col, n_col + 4, title="world_feature_i")
                        subplt14 = fig.add_subplot(4, n_col,  n_col*2 + 4, title="img_feature_i")
                        subplt15 = fig.add_subplot(4, n_col, n_col*3 + 4, title="world_feature_i")

                        subplt16 = fig.add_subplot(4, n_col,  5, title="student_img_example")
                        subplt17 = fig.add_subplot(4, n_col, n_col + 5, title="teacher_img_example")
                        subplt18 = fig.add_subplot(4, n_col,  n_col*2 + 5, title="view indicator")
                        subplt19 = fig.add_subplot(4, n_col, n_col*3 + 5, title="all world features")


                        img0 = (255*self.denormalize(data_student[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])).astype(np.uint8)
                        img1 = (255*self.denormalize(data_teacher[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])).astype(np.uint8)

                        student_res_view = display_cam_layout(map_res_target.cpu().detach().numpy().squeeze(), view_indicator_list)
                        label_view = display_cam_layout(self.criterion._traget_transform(map_res_target, map_gt_target, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                    .cpu().detach().numpy().squeeze(), view_indicator_list_teacher)

                        pseudo_label_view = display_cam_layout(self.criterion._traget_transform(map_res_target, map_pseudo_label_unaug,
                                                                                data_loader_target.dataset.dicts[dataset_name_trg[0]]['base'].map_kernel).cpu().detach().numpy().squeeze(), view_indicator_list_teacher)

                        pseudo_label_view_aug = display_cam_layout(self.criterion._traget_transform(map_res_target, map_pseudo_label,
                                                                                data_loader_target.dataset.dicts[dataset_name_trg[0]]['base'].map_kernel).cpu().detach().numpy().squeeze(), view_indicator_list)
                        teacher_res_view = display_cam_layout(map_pred_teacher.cpu().detach().numpy().squeeze(), view_indicator_list_teacher)

                        subplt0.imshow(student_res_view)
                        subplt1.imshow(label_view)
                        subplt2.imshow(pseudo_label_view)
                        subplt3.imshow(teacher_res_view)

                        if len(world_features) >= 1:
                            img_feature_i = torch.norm(img_features[0][0].detach(), dim=0).cpu().numpy()
                            w_feature_i = torch.norm(world_features[0][0].detach(), dim=0).cpu().numpy()
                            subplt4.imshow(img_feature_i)
                            subplt5.imshow(w_feature_i)

                        if len(world_features) >= 2:
                            img_feature_i = torch.norm(img_features[1][0].detach(), dim=0).cpu().numpy()
                            w_feature_i = torch.norm(world_features[1][0].detach(), dim=0).cpu().numpy()
                            subplt6.imshow(img_feature_i)
                            subplt7.imshow(w_feature_i)

                        if len(world_features) >= 3:
                            img_feature_i = torch.norm(img_features[2][0].detach(), dim=0).cpu().numpy()
                            w_feature_i = torch.norm(world_features[2][0].detach(), dim=0).cpu().numpy()
                            subplt8.imshow(img_feature_i)
                            subplt9.imshow(w_feature_i)

                        if len(world_features) >= 4:
                            img_feature_i = torch.norm(img_features[3][0].detach(), dim=0).cpu().numpy()
                            w_feature_i = torch.norm(world_features[3][0].detach(), dim=0).cpu().numpy()
                            subplt10.imshow(img_feature_i)
                            subplt11.imshow(w_feature_i)

                        if len(world_features) >= 5:
                            img_feature_i = torch.norm(img_features[4][0].detach(), dim=0).cpu().numpy()
                            w_feature_i = torch.norm(world_features[4][0].detach(), dim=0).cpu().numpy()
                            subplt12.imshow(img_feature_i)
                            subplt13.imshow(w_feature_i)

                        if len(world_features) >= 6:
                            img_feature_i = torch.norm(img_features[5][0].detach(), dim=0).cpu().numpy()
                            w_feature_i = torch.norm(world_features[5][0].detach(), dim=0).cpu().numpy()
                            subplt14.imshow(img_feature_i)
                            subplt15.imshow(w_feature_i)

                        if len(world_features) >= 7:
                            img_feature_i = torch.norm(img_features[6][0].detach(), dim=0).cpu().numpy()
                            w_feature_i = torch.norm(world_features[6][0].detach(), dim=0).cpu().numpy()
                            subplt16.imshow(img_feature_i)
                            subplt17.imshow(w_feature_i)

                        subplt16.imshow(img0)
                        subplt17.imshow(img1)


                        all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                        all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()
                        subplt18.imshow(all_views)
                        subplt19.imshow(all_world_features)

                        epoch_dir = os.path.join(self.logdir, f'epoch_{epoch}')
                        if not os.path.exists(epoch_dir):
                            os.mkdir(epoch_dir)
                        plt.savefig(os.path.join(epoch_dir, f'train_target_features_{batch_idx}.jpg'))
                        plt.close(fig)

                        fig = plt.figure(dpi=500)
                        n_col = 2
                        subplt0 = fig.add_subplot(4, n_col, 1, title="student output")
                        subplt1 = fig.add_subplot(4, n_col, n_col*1 + 1, title="label")
                        subplt2 = fig.add_subplot(4, n_col, n_col*2 + 1, title="pseudo-label")
                        subplt3 = fig.add_subplot(4, n_col, n_col*3 + 1, title="teacher output")
                        subplt4 = fig.add_subplot(4, n_col,  2, title="pseudo-label augmented")
                        subplt5 = fig.add_subplot(4, n_col,  n_col + 2, title="view indicator")
                        subplt6 = fig.add_subplot(4, n_col, n_col*2 + 2, title="all world features")
                        subplt0.imshow(student_res_view)
                        subplt1.imshow(label_view)
                        subplt2.imshow(pseudo_label_view)
                        subplt3.imshow(teacher_res_view)
                        subplt4.imshow(pseudo_label_view_aug)
                        subplt5.imshow(all_views)
                        subplt6.imshow(all_world_features)
                        plt.savefig(os.path.join(epoch_dir, f'train_target_{batch_idx}.jpg'))
                        plt.close(fig)

                        # visualize pseudo-label of perspective view
                        for cam_indx, img_pseudo_label in enumerate(imgs_pseudo_labels):
                            if img_pseudo_label is None:
                                continue

                            pseudo_view1 = img_pseudo_label
                            pred_view1 = imgs_res_target[cam_indx]
                            pseudo_view1 = self.criterion._traget_transform(pred_view1, pseudo_view1, data_loader_target.dataset.dicts[dataset_name_trg[0]]['base'].img_kernel).cpu().detach().numpy().squeeze()
                            pseudo_view1_head = pseudo_view1[0]
                            pseudo_view1_foot = pseudo_view1[1]

                            cam_num = self.target_cameras[cam_indx]
                            img0 = self.denormalize(data_student[0, cam_indx]).cpu().numpy().squeeze().transpose([1, 2, 0])
                            img0 = Image.fromarray((img0 * 255).astype('uint8'))
                            # head_cam_result = add_heatmap_to_image(pseudo_view1_head, img0)
                            # head_cam_result.save(os.path.join(epoch_dir, f'head_pseudo_label_cam{cam_num}_{batch_idx}.jpg'))
                            foot_cam_result = add_heatmap_to_image(pseudo_view1_foot, img0)
                            foot_cam_result.save(os.path.join(epoch_dir, f'foot_pseudo_label_cam{cam_num+1}_{batch_idx}.jpg'))

                            # visualizing the heatmap for per-view estimation
                            heatmap0_head = pred_view1[0, 0].detach().cpu().numpy().squeeze()
                            heatmap0_foot = pred_view1[0, 1].detach().cpu().numpy().squeeze()
                            # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                            # head_cam_result.save(os.path.join(epoch_dir, f'output_cam{cam_num+1}_head_{batch_idx}.jpg'))
                            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                            foot_cam_result.save(os.path.join(epoch_dir, f'student_output_cam{cam_num+1}_foot_{batch_idx}.jpg'))
                    

                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss_source: {:.6f}, Loss_target: {:.6f}, target_weight: {:.2f}'
                      'prec: {:.1f}%, recall: {:.1f}%, Time: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), losses_target / (batch_idx + 1), target_weight, precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / (batch_idx + 1), t_backward / (batch_idx + 1), map_res_max))

                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss_source: {:.6f}, Loss_target: {:.6f},'
              'Precision: {:.1f}%, Recall: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader),losses_target / len(data_loader_target), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, varying_cls_thres=False):
        if varying_cls_thres:
            cls_thres_array = np.arange(0.05, 0.95, 0.05)
            all_res_list = {str(x): [] for x in cls_thres_array}

            self.model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = {str(x): [] for x in cls_thres_array}
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]

                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.model(data, proj_mats, config_dict)
                if res_fpath is not None:
                    for cls_thres in cls_thres_array:
                        map_grid_res = map_res.detach().cpu().squeeze()
                        v_s = map_grid_res[map_grid_res > cls_thres].unsqueeze(1)
                        grid_ij = (map_grid_res > cls_thres).nonzero()
                        if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                            grid_xy = grid_ij[:, [1, 0]]
                        else:
                            grid_xy = grid_ij
                        all_res_list[str(cls_thres)].append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                        loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    fig = plt.figure(dpi=500)
                    subplt0 = fig.add_subplot(411, title="output")
                    subplt1 = fig.add_subplot(412, title="target")
                    subplt2 = fig.add_subplot(413, title="view indicators")
                    subplt3 = fig.add_subplot(414, title="world features")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)

                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                    heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'cam1_foot_{batch_idx}.jpg'))

            moda = 0
            moda_04 = 0
            modp_04 = 0
            precision_04 = 0
            recall_04 = 0
            moda_list = []
            precision_list = []
            recall_list = []
            modp_list = []
            if res_fpath is not None:
                for i, cls_thres in enumerate(cls_thres_array):
                    all_res_list_thres = all_res_list[str(cls_thres)]
                    all_res_list_thres = torch.cat(all_res_list_thres, dim=0)
                    np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list_thres.numpy(), '%.8f')
                    res_list = []
                    for frame in np.unique(all_res_list_thres[:, 0]):
                        res = all_res_list_thres[all_res_list_thres[:, 0] == frame, :]
                        positions, scores = res[:, 1:3], res[:, 3]
                        ids, count = nms(positions, scores, 20, np.inf)
                        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                    np.savetxt(res_fpath, res_list, '%d')

                    recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                                data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)

                    moda_list.append(moda)
                    modp_list.append(modp)
                    precision_list.append(precision)
                    recall_list.append(recall)

                    if cls_thres == 0.4:
                        moda_04 = moda
                        modp_04 = modp
                        precision_04 = precision
                        recall_04 = recall

                max_indx = np.argmax(moda_list)
                moda = moda_list[max_indx]
                modp = modp_list[max_indx]
                precision = precision_list[max_indx]
                recall = recall_list[max_indx]
                max_cls_thres = cls_thres_array[max_indx]
                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%, cls_thres: {:.2f}'.
                        format(moda, modp, precision, recall, max_cls_thres))

            t1 = time.time()
            t_epoch = t1 - t0
            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, max_cls_thres), (moda_04, modp_04, precision_04, recall_04, 0.4)

        else:
            self.model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = []
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]
                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.model(data, proj_mats, config_dict)
                if res_fpath is not None:
                    map_grid_res = map_res.detach().cpu().squeeze()
                    v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                    grid_ij = (map_grid_res > self.cls_thres).nonzero()
                    if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                        grid_xy = grid_ij[:, [1, 0]]
                    else:
                        grid_xy = grid_ij
                    all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                    loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > self.cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(411, title="output")
                    subplt1 = fig.add_subplot(412, title="target")
                    subplt2 = fig.add_subplot(413, title="view indicators")
                    subplt3 = fig.add_subplot(414, title="world features")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)

                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                    # visualizing the heatmap for per-view estimation
                    # heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
                    heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                    # head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'cam1_foot_{batch_idx}.jpg'))

            t1 = time.time()
            t_epoch = t1 - t0

            moda = 0
            if res_fpath is not None:
                all_res_list = torch.cat(all_res_list, dim=0)
                np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
                res_list = []
                for frame in np.unique(all_res_list[:, 0]):
                    res = all_res_list[all_res_list[:, 0] == frame, :]
                    positions, scores = res[:, 1:3], res[:, 3]
                    ids, count = nms(positions, scores, 20, np.inf)
                    res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                np.savetxt(res_fpath, res_list, '%d')

                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)


                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                    format(moda, modp, precision, recall))

            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, self.cls_thres), (moda, modp, precision, recall, self.cls_thres)
    
    def test_ema(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, varying_cls_thres=False):
        if varying_cls_thres:
            cls_thres_array = np.arange(0.05, 0.95, 0.05)
            all_res_list = {str(x): [] for x in cls_thres_array}

            self.ema_model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = {str(x): [] for x in cls_thres_array}
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]

                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.ema_model(data, proj_mats, config_dict)
                if res_fpath is not None:
                    for cls_thres in cls_thres_array:
                        map_grid_res = map_res.detach().cpu().squeeze()
                        v_s = map_grid_res[map_grid_res > cls_thres].unsqueeze(1)
                        grid_ij = (map_grid_res > cls_thres).nonzero()
                        if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                            grid_xy = grid_ij[:, [1, 0]]
                        else:
                            grid_xy = grid_ij
                        all_res_list[str(cls_thres)].append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                        loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    fig = plt.figure(dpi=500)
                    subplt0 = fig.add_subplot(411, title="output")
                    subplt1 = fig.add_subplot(412, title="target")
                    subplt2 = fig.add_subplot(413, title="view indicators")
                    subplt3 = fig.add_subplot(414, title="world features")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)

                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                    heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'cam1_foot_{batch_idx}.jpg'))

            moda = 0
            moda_04 = 0
            modp_04 = 0
            precision_04 = 0
            recall_04 = 0
            moda_list = []
            precision_list = []
            recall_list = []
            modp_list = []
            if res_fpath is not None:
                for i, cls_thres in enumerate(cls_thres_array):
                    all_res_list_thres = all_res_list[str(cls_thres)]
                    all_res_list_thres = torch.cat(all_res_list_thres, dim=0)
                    np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list_thres.numpy(), '%.8f')
                    res_list = []
                    for frame in np.unique(all_res_list_thres[:, 0]):
                        res = all_res_list_thres[all_res_list_thres[:, 0] == frame, :]
                        positions, scores = res[:, 1:3], res[:, 3]
                        ids, count = nms(positions, scores, 20, np.inf)
                        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                    np.savetxt(res_fpath, res_list, '%d')

                    recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                                data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)

                    moda_list.append(moda)
                    modp_list.append(modp)
                    precision_list.append(precision)
                    recall_list.append(recall)

                    if cls_thres == 0.4:
                        moda_04 = moda
                        modp_04 = modp
                        precision_04 = precision
                        recall_04 = recall

                max_indx = np.argmax(moda_list)
                moda = moda_list[max_indx]
                modp = modp_list[max_indx]
                precision = precision_list[max_indx]
                recall = recall_list[max_indx]
                max_cls_thres = cls_thres_array[max_indx]
                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%, cls_thres: {:.2f}'.
                        format(moda, modp, precision, recall, max_cls_thres))

            t1 = time.time()
            t_epoch = t1 - t0
            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, max_cls_thres), (moda_04, modp_04, precision_04, recall_04, 0.4)

        else:
            self.ema_model.eval()
            losses = 0
            precision_s, recall_s = AverageMeter(), AverageMeter()
            all_res_list = []
            t0 = time.time()
            if res_fpath is not None:
                assert gt_fpath is not None
            for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, _, dataset_name) in enumerate(data_loader):
                with torch.no_grad():
                    config_dict = data_loader.dataset.dicts[dataset_name[0]]
                    map_res, imgs_res, (world_features, img_features, view_indicator_list) = self.ema_model(data, proj_mats, config_dict)
                if res_fpath is not None:
                    map_grid_res = map_res.detach().cpu().squeeze()
                    v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                    grid_ij = (map_grid_res > self.cls_thres).nonzero()
                    if data_loader.dataset.dicts[dataset_name[0]]['base'].indexing == 'xy':
                        grid_xy = grid_ij[:, [1, 0]]
                    else:
                        grid_xy = grid_ij
                    all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                data_loader.dataset.dicts[dataset_name[0]]['base'].grid_reduce, v_s], dim=1))

                loss = 0
                for img_res, img_gt in zip(imgs_res, imgs_gt):
                    loss += self.criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
                loss = self.criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                    loss / len(imgs_gt) * self.alpha
                losses += loss.item()
                pred = (map_res > self.cls_thres).int().to(map_gt.device)
                true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
                false_positive = pred.sum().item() - true_positive
                false_negative = map_gt.sum().item() - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-4)
                recall = true_positive / (true_positive + false_negative + 1e-4)
                precision_s.update(precision)
                recall_s.update(recall)

                if visualize:
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(411, title="output")
                    subplt1 = fig.add_subplot(412, title="target")
                    subplt2 = fig.add_subplot(413, title="view indicators")
                    subplt3 = fig.add_subplot(414, title="world features")

                    map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
                    label_view = display_cam_layout(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                                .cpu().detach().numpy().squeeze(), view_indicator_list)
                    all_views = torch.norm(torch.cat(view_indicator_list, dim=1)[0], dim=0).numpy()
                    all_world_features = torch.norm(torch.cat(world_features, dim=1)[0], dim=0).detach().cpu().numpy()

                    subplt0.imshow(map_res_view)
                    subplt1.imshow(label_view)
                    subplt2.imshow(all_views)
                    subplt3.imshow(all_world_features)

                    plt.savefig(os.path.join(self.logdir, f'map_{batch_idx}.jpg'))
                    plt.close(fig)

                    # visualizing the heatmap for per-view estimation
                    # heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
                    heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                    # head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
                    foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                    foot_cam_result.save(os.path.join(self.logdir, f'cam1_foot_{batch_idx}.jpg'))

            t1 = time.time()
            t_epoch = t1 - t0

            moda = 0
            if res_fpath is not None:
                all_res_list = torch.cat(all_res_list, dim=0)
                np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
                res_list = []
                for frame in np.unique(all_res_list[:, 0]):
                    res = all_res_list[all_res_list[:, 0] == frame, :]
                    positions, scores = res[:, 1:3], res[:, 3]
                    ids, count = nms(positions, scores, 20, np.inf)
                    res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                np.savetxt(res_fpath, res_list, '%d')

                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)

                print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                    format(moda, modp, precision, recall))

            print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
                losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

            return losses / len(data_loader), (moda, modp, precision, recall, self.cls_thres), (moda, modp, precision, recall, self.cls_thres)
    
    @staticmethod
    def update_ema_variables(ema_model, model, alpha_teacher, iteration):

        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model




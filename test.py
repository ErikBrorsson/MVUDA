# code from https://github.com/hou-yz/MVDet/tree/master
# modified by Erik Brorsson
import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve2
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.utils.meters import AverageMeter
import time
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.datasets.concat_dataset import ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

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

    color_list = [
        (0,255,0),
        (0,0,255),
        (255,127,0),
        (0,255,127),
        (127,0,255),
        (255,255,0),
        (255,0,255)
    ]

    for view_index, view in enumerate(view_indicator_list):
        temp = (255*view[0][0,:,:].detach().cpu().numpy()).astype(np.uint8)
        contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        drawing=cv2.drawContours(drawing, contours, -1, color_list[view_index], 1)

    c_grid = [
        [170, 330],
        [930, 100],
        [700, 340],
        [120, 290],
        [520, 50],
        [10, 80],
        [350, 358],
    ]

    for i, _ in enumerate(view_indicator_list):
        drawing = cv2.putText(drawing, f'C{i+1}', (c_grid[i][0], c_grid[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                              1, color_list[i], 2, cv2.LINE_AA)  

    return drawing

def test(model, data_loader, cls_thres_array, criterion, alpha, res_fpath=None, gt_fpath=None):
    model.eval()
    losses = 0
    precision_s, recall_s = AverageMeter(), AverageMeter()
    all_res_list = {str(x): [] for x in cls_thres_array}
    t0 = time.time()
    if res_fpath is not None:
        assert gt_fpath is not None
    for batch_idx, (data, map_gt, imgs_gt, frame, proj_mats, _, _, _, proj_mats_mvaug_features, dataset_name) in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            config_dict = data_loader.dataset.dicts[dataset_name[0]]
            map_res, imgs_res, (world_features, img_features, view_indicator_list) = model(data, proj_mats, config_dict, visualize=False)
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
            loss += criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].img_kernel)
        loss = criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel) + \
                loss / len(imgs_gt) * alpha
        losses += loss.item()
        pred = (map_res > cls_thres).int().to(map_gt.device)
        true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
        false_positive = pred.sum().item() - true_positive
        false_negative = map_gt.sum().item() - true_positive
        precision = true_positive / (true_positive + false_positive + 1e-4)
        recall = true_positive / (true_positive + false_negative + 1e-4)
        precision_s.update(precision)
        recall_s.update(recall)

        fig = plt.figure(figsize=(16,9))
        map_res_view = display_cam_layout(map_res.cpu().detach().numpy().squeeze(), view_indicator_list)
        plt.imshow(map_res_view)
        plt.savefig(os.path.join(os.path.dirname(res_fpath), f'map_{batch_idx}.jpg'))
        plt.close(fig)

        fig = plt.figure(figsize=(16,9))
        label_view = display_cam_layout(criterion._traget_transform(map_res, map_gt, data_loader.dataset.dicts[dataset_name[0]]['base'].map_kernel)
                    .cpu().detach().numpy().squeeze(), view_indicator_list)
        plt.imshow(label_view)
        plt.savefig(os.path.join(os.path.dirname(res_fpath), f'label_{batch_idx}.jpg'))
        plt.close(fig)

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
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + f'/all_res.txt', all_res_list_thres.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list_thres[:, 0]):
                res = all_res_list_thres[all_res_list_thres[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])

            res_fpath_i = res_fpath.replace(".txt", "_{:.2f}.txt".format(cls_thres))
            np.savetxt(res_fpath_i, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath_i), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.dicts[dataset_name[0]]['base'].base.__name__)


            print("cls_thres: ", cls_thres)
            print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                    format(moda, modp, precision, recall))
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
        print("\nBest results ############")
        print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%, cls_thres: {:.2f}'.
                format(moda, modp, precision, recall, max_cls_thres))
        
        print("\n cls_thres=0.4 results ##################")
        print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%, cls_thres: {:.2f}'.
                format(moda_04, modp_04, precision_04, recall_04, 0.4))
        

    t1 = time.time()
    t_epoch = t1 - t0
    print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
        losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

    return losses / len(data_loader), (moda, modp, precision, recall, max_cls_thres), (moda_04, modp_04, precision_04, recall_04, 0.4)


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])



    if 'wildtrack' in args.dataset:
        # data_path = os.path.expanduser('/data/Wildtrack')
        assert args.data_path is not None, "must specify data path"
        data_path = args.data_path
        if args.cam_adapt:
            assert args.trg_cams is not None, "trg_cams must be specified in cam_adapt setting"
            trg_cams = args.trg_cams.split(",")
            trg_cams = [int(x) for x in trg_cams]
            print("trg_cams: ", trg_cams)
            test_base = Wildtrack(data_path, cameras=trg_cams)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)
            test_loader = torch.utils.data.DataLoader(ConcatDataset(test_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            train_base = Wildtrack(data_path, cameras=trg_cams)
            train_set = frameDataset(train_base, train=True, transform=train_trans, grid_reduce=4)
            train_loader = torch.utils.data.DataLoader(ConcatDataset(train_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            if args.src_cams is not None:
                src_cams = args.src_cams.split(",")
                src_cams = [int(x) for x in src_cams]
                print("src_cams: ", src_cams)
                train_base_src = Wildtrack(data_path, cameras=src_cams)
                train_set_src = frameDataset(train_base_src, train=True, transform=train_trans, grid_reduce=4)
             
        else:
            test_base = Wildtrack(data_path)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)            
            test_loader = torch.utils.data.DataLoader(ConcatDataset(test_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            train_base = Wildtrack(data_path)
            train_set = frameDataset(train_base, train=True, transform=train_trans, grid_reduce=4)
            train_loader = torch.utils.data.DataLoader(ConcatDataset(train_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
    
    elif 'multiviewx' in args.dataset:
        data_path = args.data_path
        if args.cam_adapt:
            assert args.trg_cams is not None, "src_cams and trg_cams must be specified in cam_adapt setting"
            trg_cams = args.trg_cams.split(",")
            trg_cams = [int(x) for x in trg_cams]
            print("trg_cams: ", trg_cams)
            test_base = MultiviewX(data_path, cameras=trg_cams)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)
            test_loader = torch.utils.data.DataLoader(ConcatDataset(test_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            train_base = MultiviewX(data_path, cameras=trg_cams)
            train_set = frameDataset(train_base, train=True, transform=train_trans, grid_reduce=4)
            train_loader = torch.utils.data.DataLoader(ConcatDataset(train_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            if args.src_cams is not None:
                src_cams = args.src_cams.split(",")
                src_cams = [int(x) for x in src_cams]
                print("src_cams: ", src_cams)
                train_base_src = MultiviewX(data_path, cameras=src_cams)
                train_set_src = frameDataset(train_base_src, train=True, transform=train_trans, grid_reduce=4)
             
        else:
            test_base = MultiviewX(data_path)
            test_set = frameDataset(test_base, train=False, transform=train_trans, grid_reduce=4)            
            test_loader = torch.utils.data.DataLoader(ConcatDataset(test_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
            
            train_base = MultiviewX(data_path)
            train_set = frameDataset(train_base, train=True, transform=train_trans, grid_reduce=4)
            train_loader = torch.utils.data.DataLoader(ConcatDataset(train_set), batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')



    # model
    if args.variant == 'default':
        if args.src_cams is not None:
            model = PerspTransDetector(args.arch, avgpool=args.avgpool)
        else:
            model = PerspTransDetector(args.arch, avgpool=args.avgpool)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(test_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(test_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(test_set, args.arch)
    else:
        raise Exception('no support for this variant')

    # loss
    criterion = GaussianMSE().cuda()

    # logging
    idx = 0
    while os.path.exists(os.path.join(args.log_dir, "test_"+str(idx))):
        idx += 1

    logdir=os.path.join(args.log_dir, "test_"+str(idx))
    os.makedirs(logdir, exist_ok=True)

    sys.stdout = Logger(os.path.join(logdir, 'test_log.txt'), )
    print('Settings:')
    print(vars(args))

    resume_fname = os.path.join(args.log_dir, args.model)
    print("Loading saved model from: ", resume_fname)
    model.load_state_dict(torch.load(resume_fname))


    ema_model = PerspTransDetector(args.arch, avgpool=args.avgpool)
    for param in ema_model.parameters():
        param.detach_()
    ema_model.load_state_dict(model.state_dict()) # this method correctly copies the parameters

    print('Testing...')

    print("test_set.gt_fpath: ", test_set.gt_fpath)
    cls_thres_array = np.arange(0.05, 0.95, 0.05)
    # cls_thres_array = [0.2]
    test_loss, metrics, metrics_04 = test(model, test_loader, cls_thres_array, criterion,
                                                               args.alpha,  os.path.join(logdir, 'test.txt'), test_set.gt_fpath)
    (moda, modp, precision, recall, cls_thres_var) = metrics
    (moda_04, modp_04, precision_04, recall_04, cls_thres_fix) = metrics_04

    x_epoch = []
    cls_thres_list = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []
    test_modp_s = []
    test_recall_s = []
    for i in range(1, 4):
        x_epoch.append(i)
        test_loss_s.append(test_loss)
        test_prec_s.append(precision)
        test_moda_s.append(moda)
        test_modp_s.append(modp)
        test_recall_s.append(recall)
        cls_thres_list.append(cls_thres_var)

    draw_curve2(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, test_loss_s, test_loss_s,
                test_moda_s, test_modp_s, test_prec_s, test_recall_s, cls_thres_list)
    



if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')

    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--cam_adapt', action="store_true")
    parser.add_argument('--train_set', action="store_true")
    parser.add_argument('--trg_cams', type=str, default=None)
    parser.add_argument('--src_cams', type=str, default=None,
                        help="specify src_cams if the model was trained with a different number of cameras than expected for testing")
    parser.add_argument('--model', type=str, default="MultiviewDetector.pth")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--avgpool", action="store_true")

    args = parser.parse_args()


    main(args)

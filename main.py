# code from https://github.com/hou-yz/MVDet/tree/master
# modified by Erik Brorsson
import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import WeightedGaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve2
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer, UDATrainer, Augmentation
from multiview_detector.datasets.concat_dataset import ConcatDataset
from multiview_detector.datasets.dataloader import GetDataset
from multiview_detector.datasets.dataloader_3drom import GetDataset3DROM
import csv

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

    if args.rom3d_uda is None:
        args.rom3d_uda = args.rom3d

    if args.rom3d:
        framedataset = frameDataset3DROM
        getdataset = GetDataset3DROM
    else:
        framedataset = frameDataset
        getdataset = GetDataset

    if args.rom3d_uda:
        framedataset_trg = frameDataset3DROM
    else:
        framedataset_trg = frameDataset


    if args.dataset_src == "wildtrack":
        data_path = args.data_path_src
        if args.src_cams is not None:
            src_cams = args.src_cams.split(",")
            src_cams = [int(x) for x in src_cams]
            source_base = Wildtrack(data_path, cameras=src_cams)
        else:
            source_base = Wildtrack(data_path)
        train_dataset_src_ = framedataset(source_base, train=True, transform=train_trans, grid_reduce=4, img_reduce=4)
        train_dataset_src = ConcatDataset(train_dataset_src_)
        train_loader = torch.utils.data.DataLoader(train_dataset_src, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True) 
    elif args.dataset_src == "multiviewx":
        data_path = args.data_path_src
        if args.src_cams is not None:
            src_cams = args.src_cams.split(",")
            src_cams = [int(x) for x in src_cams]
            source_base = MultiviewX(data_path, cameras=src_cams)
        else:
            source_base = MultiviewX(data_path)
        train_dataset_src_ = framedataset(source_base, train=True, transform=train_trans, grid_reduce=4, img_reduce=4)
        train_dataset_src = ConcatDataset(train_dataset_src_)
        train_loader = torch.utils.data.DataLoader(train_dataset_src, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    elif args.dataset_src == "gmvd":
        data_root = args.data_path_src
        train_dataset_list = []
        assert os.path.exists(os.path.join(data_root,args.gmvd_csv)), f"{os.path.join(data_root,args.gmvd_csv)} doesn't exist"
        f = open(os.path.join(data_root,args.gmvd_csv))
        data_path = csv.reader(f)
        for i,data_row in enumerate(data_path):
            train_ratio = float(data_row[2])
            sample_require = int(data_row[3])
            path = os.path.join(data_root, str(data_row[1]))
            if data_row[1].split('/')[-1]!='Wildtrack':
                base = MultiviewX(path, camera_orient="wildtrack")
            else:
                base = Wildtrack(path)
            if data_row[0]=='train':
                # Train data
                dataset_obj = getdataset(base, train=True, transform=train_trans, grid_reduce=4, img_reduce=4, train_ratio=train_ratio, sample_require=sample_require)
                train_dataset_list.append(dataset_obj)
        train_set_ = ConcatDataset(*train_dataset_list)

        train_loader = torch.utils.data.DataLoader(train_set_, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    else:
        raise Exception(f"args.dataset_src = {args.dataset_src} is not allowed")

    if args.dataset_trg == "wildtrack":
        data_path = args.data_path_trg
        if args.trg_cams is not None:
            trg_cams = args.trg_cams.split(",")
            trg_cams = [int(x) for x in trg_cams]
            target_base = Wildtrack(data_path, cameras=trg_cams)
        else:
            target_base = Wildtrack(data_path)
        train_dataset_trg_ = framedataset_trg(target_base, train=True, transform=train_trans, grid_reduce=4, img_reduce=4)
        train_dataset_trg = ConcatDataset(train_dataset_trg_)

        test_base0 = Wildtrack(data_path)
        test_set = framedataset(test_base0, train=False, transform=train_trans, grid_reduce=4, img_reduce=4)
        test_dataset_ = ConcatDataset(test_set)

        train_loader_target = torch.utils.data.DataLoader(train_dataset_trg, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset_, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    elif args.dataset_trg == "multiviewx":
        data_path = args.data_path_trg
        if args.trg_cams is not None:
            trg_cams = args.trg_cams.split(",")
            trg_cams = [int(x) for x in trg_cams]
            target_base = MultiviewX(data_path, cameras=trg_cams)
        else:
            target_base = MultiviewX(data_path)
        train_dataset_trg_ = framedataset_trg(target_base, train=True, transform=train_trans, grid_reduce=4, img_reduce=4)
        train_dataset_trg = ConcatDataset(train_dataset_trg_)

        test_base0 = MultiviewX(data_path)
        test_set = framedataset(test_base0, train=False, transform=train_trans, grid_reduce=4, img_reduce=4)
        test_dataset_ = ConcatDataset(test_set)
        train_loader_target = torch.utils.data.DataLoader(train_dataset_trg, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset_, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    else:
        raise Exception(f"args.dataset_trg = {args.dataset_trg} is not allowed")


    print("images in source training set: ", len(train_loader))
    print("images in target training set: ", len(train_loader_target))
    print("images in target test set: ", len(test_loader))

    # model
    if args.variant == 'default':
        model = PerspTransDetector(args.arch, pretrained=args.pretrained, avgpool=args.avgpool)

        # load pre-trained model before initializing EMA
        if args.resume_model is not None:
            resume_fname = args.resume_model
            print("Loading saved model from: ", resume_fname)
            model.load_state_dict(torch.load(resume_fname))

        # init ema model
        ema_model = PerspTransDetector(args.arch, pretrained=args.pretrained, avgpool=args.avgpool)
        ema_model.load_state_dict(model.state_dict()) # using load_state_dict here to copy parameters and buffers (buffers include e.g. batch_norm mean)
        for param in ema_model.parameters():
            param.detach_()

    else:
        raise Exception('no support for this variant')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    criterion = WeightedGaussianMSE().cuda()


    # logging
    logdir = f'logs/{args.dataset_src}_frame/{args.variant}/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')# if not args.resume else f'logs/{args.dataset}_frame/{args.variant}/{args.resume}'
    if args.log_dir is not None:
        logdir = os.path.join(args.log_dir, logdir)

    # if args.resume is None:
    os.makedirs(logdir, exist_ok=True)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )

    # draw curve
    x_epoch = []
    train_loss_s = []
    # train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []
    test_modp_s = []
    test_recall_s = []

    test_prec_s_04 = []
    test_moda_s_04 = []
    test_modp_s_04 = []
    test_recall_s_04 = []
    cls_thres_list_var= []
    cls_thres_list_fix = []

    augmentation = Augmentation(args.dropview, args.mvaug)
    if args.dropview_uda is None:
        args.dropview_uda = args.dropview
    if args.mvaug_uda is None:
        args.mvaug_uda = args.mvaug
    augmentation_uda = Augmentation(args.dropview_uda, args.mvaug_uda, rom3d=args.rom3d_uda)

    print('Settings:')
    for k, v in vars(args).items():
        print(k, ": ", v)

    print("logdir: ", logdir)

    if args.uda:
        trainer = UDATrainer(model, ema_model, criterion, logdir, denormalize, args.cls_thres, args.alpha,
                             args.train_viz, target_cameras=target_base.cameras,
                             alpha_teacher=args.alpha_teacher,
                             augmentation_module=augmentation, uda_persp_sup=args.uda_persp_sup,
                             persp_sup=args.persp_sup, uda_nms_th=args.uda_nms_th, augmentation_uda=augmentation_uda,
                             max_pseudo=args.max_pseudo, max_pseudo_th=args.max_pseudo_th)
    else:
        trainer = PerspectiveTrainer(model, ema_model, criterion, logdir, denormalize, args.cls_thres, args.alpha,
                                     augmentation_module=augmentation, persp_sup=args.persp_sup, visualize_train=args.train_viz)


    print('Testing...')
    test_loss, (moda, modp, precision, recall, cls_thres_var), (moda_04, modp_04, precision_04, recall_04, cls_thres_fix) = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                test_set.gt_fpath, True, varying_cls_thres=args.varying_cls_thres)
    trainer.test_ema(test_loader, os.path.join(logdir, 'test.txt'),
                                                        test_set.gt_fpath, True, varying_cls_thres=args.varying_cls_thres)
    max_moda = -1e10
    best_epoch = -1

    max_moda_ema = -1e10
    best_epoch_ema = -1
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        print('Training...')
        if args.uda:
            train_loss, train_prec = trainer.train(epoch, train_loader, train_loader_target, optimizer, args.log_interval, scheduler, args.lambda_weight, args.pseudo_label_th)
        else:
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
        print('Testing...')
        test_loss, (moda, modp, precision, recall, cls_thres_var), (moda_04, modp_04, precision_04, recall_04, cls_thres_fix) = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                    test_set.gt_fpath, True, varying_cls_thres=args.varying_cls_thres)
        
        if args.test_ema:
            _, (moda_ema, modp_ema, precision_ema, recall_ema, cls_thres_var_ema), (moda_04_ema, modp_04_ema, precision_04_ema, recall_04_ema, cls_thres_fix_ema) = trainer.test_ema(test_loader, os.path.join(logdir, 'test.txt'),
                                                        test_set.gt_fpath, True, varying_cls_thres=args.varying_cls_thres)

        if moda >= max_moda:
            max_modp, max_precision, max_recall = modp, precision, recall
            max_moda = moda
            best_epoch = epoch
            # save model after every epoch
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))

        if args.test_ema:
            if moda_ema >= max_moda_ema:
                max_modp_ema, max_precision_ema, max_recall_ema = modp_ema, precision_ema, recall_ema
                max_moda_ema = moda_ema
                best_epoch_ema = epoch
                # save model after every epoch
                torch.save(ema_model.state_dict(), os.path.join(logdir, 'MultiviewDetector_ema.pth'))

        torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector_latest.pth'))
        if args.uda or args.test_ema:
            torch.save(ema_model.state_dict(), os.path.join(logdir, 'MultiviewDetector_ema_latest.pth'))


        x_epoch.append(epoch)
        train_loss_s.append(train_loss)
        test_loss_s.append(test_loss)

        cls_thres_list_fix.append(cls_thres_fix)
        test_prec_s_04.append(precision_04)
        test_moda_s_04.append(moda_04)
        test_modp_s_04.append(modp_04)
        test_recall_s_04.append(recall_04)
        draw_curve2(os.path.join(logdir, 'learning_curve_fixed_cls.jpg'), x_epoch, train_loss_s, test_loss_s,
            test_moda_s_04, test_modp_s_04, test_prec_s_04, test_recall_s_04, cls_thres_list_fix)

        if args.varying_cls_thres:
            cls_thres_list_var.append(cls_thres_var)
            test_prec_s.append(precision)
            test_moda_s.append(moda)
            test_modp_s.append(modp)
            test_recall_s.append(recall)

            draw_curve2(os.path.join(logdir, 'learning_curve_varying_cls.jpg'), x_epoch, train_loss_s, test_loss_s,
                test_moda_s, test_modp_s, test_prec_s, test_recall_s, cls_thres_list_var)
        
        
        print('max_moda: {:.1f}%, max_modp: {:.1f}%, max_precision: {:.1f}%, max_recall: {:.1f}%, epoch: {:.1f}%'.
                format(max_moda, max_modp, max_precision, max_recall, best_epoch))
        
        if args.test_ema:
            print('EMA METRICS: max_moda: {:.1f}%, max_modp: {:.1f}%, max_precision: {:.1f}%, max_recall: {:.1f}%, epoch: {:.1f}%'.
                    format(max_moda_ema, max_modp_ema, max_precision_ema, max_recall_ema, best_epoch_ema))


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume_model', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--train_viz', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')

    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--avgpool', action="store_true")
    parser.add_argument('--persp_sup', action="store_true", default=False)
    parser.add_argument('--varying_cls_thres', action="store_true")

    # data augmentation
    parser.add_argument('--dropview', action="store_true")
    parser.add_argument("--mvaug", action="store_true")
    parser.add_argument("--rom3d", action="store_true")

    # datasets
    parser.add_argument('--dataset_src', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx', 'gmvd'])
    parser.add_argument('--dataset_trg', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument("--data_path_src", type=str, default=None)
    parser.add_argument("--data_path_trg", type=str, default=None)
    parser.add_argument("--gmvd_csv", type=str, default="train_datapath.csv")
    parser.add_argument('--src_cams', type=str, default=None)
    parser.add_argument('--trg_cams', type=str, default=None)

    # uda parameters
    parser.add_argument('--uda', action="store_true")
    parser.add_argument('--max_pseudo', action="store_true")
    parser.add_argument('--max_pseudo_th', type=int, default=11, help='The kernel size when finding local_maxima for max_pseudo pseudo-label creation')
    parser.add_argument('--test_ema', action="store_true")
    parser.add_argument('--uda_nms_th', type=int, default=20, help='The NMS distance threshold used when creating pseudo-labels')
    parser.add_argument('--alpha_teacher', type=float, default=0.99)
    parser.add_argument('--uda_persp_sup', action="store_true")
    parser.add_argument('--lambda_weight', type=float, default=None, help='weight lambda determining the influence of self-training')
    parser.add_argument('--pseudo_label_th', type=float, default=None, help='confidence threshold for creating pseudo-labels')
    parser.add_argument('--dropview_uda', action="store_true", default=None)
    parser.add_argument("--mvaug_uda", action="store_true", default=None)
    parser.add_argument("--rom3d_uda", action="store_true", default=None)

    args = parser.parse_args()

    if args.config is not None:
        import json
        f = open(args.config)
        data = json.load(f)
        args_d = vars(args)

        for k,v in data.items():
            args_d[k] = v

    if not args.avgpool:
        raise Exception("Our UDA trainer is only compatible with GMVD architecture using avgpool")

    main(args)

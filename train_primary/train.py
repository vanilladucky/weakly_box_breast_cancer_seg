import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from net import AncillaryNet, PrimaryNet
from torchvision.utils import make_grid
import torch.nn.functional as F
import sys
import shutil
from file_and_folder_operations import read_data_list, myMakedirs
from augment import Norm, RandomCrop, ToTensor, Projection, CorrectSeg
from data import BreastTumor, data_prefetcher, BreastTumorEval
from losses import LogBarrierLoss, CRFLoss
from val import validate

def reproduce(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)

def main():
    reproduce(args.seed)
    logging.basicConfig(filename=os.path.join(args.exp_name, 'breast_seg.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = PrimaryNet(1, args.num_classes).cuda()
    a_net = AncillaryNet(1, args.num_classes).cuda()
    a_net.load_state_dict(torch.load('/root/autodl-tmp/Kim/weakly_box_breast_cancer_seg/train_ancillary_self_training/epoch_120.pth'))
    a_net.eval()

    train_data_list = read_data_list('/root/autodl-tmp/Kim/kits23/dataset/original_train.txt')
    transform_fg_train = transforms.Compose([Norm(),
                                             RandomCrop(args.patch_size, 1., 3),  # seed = 2 bbox
                                             Projection(),
                                             CorrectSeg(),
                                             ToTensor(0)])
    train_fg_dataset = BreastTumor(train_data_list, transform=transform_fg_train)
    fg_dataloader = DataLoader(train_fg_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=False,
                               worker_init_fn=worker_init_fn,
                               drop_last=True)

    transform_bg_train = transforms.Compose([Norm(),
                                             RandomCrop(args.patch_size, 1., 0),
                                             Projection(),
                                             CorrectSeg(),
                                             ToTensor(0)])
    train_bg_dataset = BreastTumor(train_data_list, transform=transform_bg_train)
    bg_dataloader = DataLoader(train_bg_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=False,
                               worker_init_fn=worker_init_fn,
                               drop_last=True)

    eval_data_list = read_data_list('/root/autodl-tmp/Kim/kits23/dataset/original_val.txt')
    transform_eval = transforms.Compose([Norm()])
    eval_dataset = BreastTumorEval(eval_data_list, transform=transform_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.99, weight_decay=1e-4, nesterov=True)
    writer = SummaryWriter(os.path.join(args.exp_name, 'tbx'))
    weights = torch.tensor([1,10,10], dtype=torch.float32).cuda() 
    CE = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=3)
    KL = torch.nn.KLDivLoss(reduction="none")
    LogBarrier = LogBarrierLoss(t=5)
    REG = CRFLoss(alpha=15, beta=0.05, is_da=False, use_norm=False)

    best_eval_dice = 0
    best_eval_jc = 0

    iter_num = 0
    max_epoch = int(args.max_epoch)
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        epoch_num = epoch_num + 1

        fg_prefetcher = data_prefetcher(fg_dataloader)
        bg_prefetcher = data_prefetcher(bg_dataloader)
        fg_sample = fg_prefetcher.next()
        bg_sample = bg_prefetcher.next()
        while fg_sample is not None and bg_sample is not None:
            iter_num = iter_num + 1
            net.train()

            fg_img, fg_seg, fg_gt = fg_sample['image'], fg_sample['label'], fg_sample['gt']
            bg_img, bg_seg, bg_gt = bg_sample['image'], bg_sample['label'], bg_sample['gt']
            fg_seg_float = fg_seg.type(torch.FloatTensor).unsqueeze(1).cuda()
            bg_seg_float = bg_seg.type(torch.FloatTensor).unsqueeze(1).cuda()
            fg_img, fg_seg = fg_img.cuda(), fg_seg.cuda()
            bg_img, bg_seg = bg_img.cuda(), bg_seg.cuda()

            projection_0 = torch.cat((fg_sample['projection_0'].cuda(), bg_sample['projection_0'].cuda()), dim=0)
            projection_1 = torch.cat((fg_sample['projection_1'].cuda(), bg_sample['projection_1'].cuda()), dim=0)
            projection_2 = torch.cat((fg_sample['projection_2'].cuda(), bg_sample['projection_2'].cuda()), dim=0)
            cor_seg = torch.cat((fg_sample['cor_seg'].cuda(), bg_sample['cor_seg'].cuda()), dim=0)  # 1 - inside bbox, 0 - outside bbox

            with torch.no_grad():
                fg_outs_a = a_net(fg_img, fg_seg_float)
                bg_outs_a = a_net(bg_img, bg_seg_float)
                outs_a_sm = F.softmax(torch.cat((fg_outs_a, bg_outs_a), dim=0) / args.T, dim=1)

            fg_outs = net(fg_img)
            bg_outs = net(bg_img)

            outs = torch.cat((fg_outs, bg_outs), dim=0)
            segs = torch.cat((fg_seg, bg_seg), dim=0)  # 2 - inside bbox, 0 - outside bbox
            outs_log_sm = F.log_softmax(torch.cat((fg_outs, bg_outs), dim=0) / args.T, dim=1)

            if np.unique(cor_seg.data.cpu().numpy()).size == 1 and np.unique(cor_seg.data.cpu().numpy())[0] == 0:
                l_ce = CE(outs, segs)
                loss = l_ce
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(list(net.parameters()), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('loss/L_supervised', l_ce.item(), iter_num)
            else:
                outs_sm_fg = outs.softmax(1)[:, 1, ...]
                bbox_outs_sm_fg = outs_sm_fg * cor_seg    # 1 - inside bbox, 0 - outside box
                outs_sm_proj_0 = bbox_outs_sm_fg.sum((2, 3))
                outs_sm_proj_1 = bbox_outs_sm_fg.sum((1, 3))
                outs_sm_proj_2 = bbox_outs_sm_fg.sum((1, 2))
                z0 = projection_0.sum() - outs_sm_proj_0[projection_0 == 1].sum()
                z1 = projection_1.sum() - outs_sm_proj_1[projection_1 == 1].sum()
                z2 = projection_2.sum() - outs_sm_proj_2[projection_2 == 1].sum()

                l_ce = CE(outs, segs)
                l_proj = 0.01 * (LogBarrier.penalty(z0) + LogBarrier.penalty(z1) + LogBarrier.penalty(z2))

                # Knowledge distillation
                w_kl = 0.5 * np.exp(-np.power((epoch_num / max_epoch), 2) / (2 * np.power(0.3, 2)))
                if w_kl < 0.05:
                    w_kl = 0.05

                l_kl = w_kl * (KL(outs_log_sm, outs_a_sm).sum(1)[segs == 2]).mean()
                loss = l_ce + l_proj + l_kl
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(list(net.parameters()), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('loss/L_kl', l_kl.item(), iter_num)

                # finetune
                fg_outs = net(fg_img)
                bg_outs = net(bg_img)

                segs = torch.cat((fg_seg, bg_seg), dim=0)  # 2 - inside bbox, 0 - outside bbox
                outs = torch.cat((fg_outs, bg_outs), dim=0)

                outs_sm_fg = outs.softmax(1)[:, 1, ...]

                bbox_outs_sm_fg = outs_sm_fg * cor_seg  # 1 - inside bbox, 0 - outside box
                outs_sm_proj_0 = bbox_outs_sm_fg.sum((2, 3))
                outs_sm_proj_1 = bbox_outs_sm_fg.sum((1, 3))
                outs_sm_proj_2 = bbox_outs_sm_fg.sum((1, 2))
                z0 = projection_0.sum() - outs_sm_proj_0[projection_0 == 1].sum()
                z1 = projection_1.sum() - outs_sm_proj_1[projection_1 == 1].sum()
                z2 = projection_2.sum() - outs_sm_proj_2[projection_2 == 1].sum()

                l_crf = 0.001 / np.prod(args.patch_size) * REG(fg_img, fg_outs)
                l_ce = CE(outs, segs)
                l_proj = 0.01 * (LogBarrier.penalty(z0) + LogBarrier.penalty(z1) + LogBarrier.penalty(z2))

                loss = l_ce + l_proj + l_crf
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(list(net.parameters()), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar('loss/L_supervised', l_ce.item(), iter_num)
                writer.add_scalar('loss/L_proj', l_proj.item(), iter_num)
                writer.add_scalar('loss/L_crf', l_crf.item(), iter_num)

            """if iter_num % 50 == 0:
                image = fg_img[0, 0:1, 30:71:10, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                outputs_soft = F.softmax(fg_outs, 1)
                image = outputs_soft[0, 1:2, 30:71:10, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted', grid_image, iter_num)

                gt_batch = fg_gt.long()
                image = gt_batch[0, 30:71:10, :, :].unsqueeze(0).permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth', grid_image, iter_num)"""

            fg_sample = fg_prefetcher.next()
            bg_sample = bg_prefetcher.next()

        lr_ = args.base_lr * (1 - epoch_num / max_epoch) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        # save
        if epoch_num % args.save_per_epoch == 0:
            save_model_path = os.path.join(args.exp_name, f'epoch_{epoch_num}.pth')
            torch.save(net.state_dict(), save_model_path)

        # eval
        if epoch_num % args.eval_per_epoch == 0:
            writer, (dice_kidney, dice_tumor), (jc_kidney, jc_tumor), (precision_kidney, precision_tumor), (recall_kidney, recall_tumor) = validate(net, eval_dataloader,
                                                                               args.patch_size, args.num_classes,
                                                                               logging, writer, iter_num, epoch_num)
            mean_dice = (dice_kidney+dice_tumor)/2
            if mean_dice > best_eval_dice:
                best_eval_dice = mean_dice
                save_model_path = os.path.join(args.exp_name, 'epoch_best.pth')
                torch.save(net.state_dict(), save_model_path)
            logging.info(f"\nEpoch: {epoch_num} | Kidney Dice score: {dice_kidney:.3f} | Tumor Dice score: {dice_tumor:.3f}")

    writer.close()

    save_model_path = os.path.join(args.exp_name, f'epoch_{max_epoch}.pth')
    torch.save(net.state_dict(), save_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='/root/autodl-tmp/Kim/weakly_box_breast_cancer_seg/train_primary')
    # parser.add_argument('--exp_name', type=str, default='/data/zym/experiment/bbox_tmi/DEBUG')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_size', type=list, default=[96, 128, 128])
    parser.add_argument('--base_lr', type=float, default=5e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--T', type=float, default=1)
    parser.add_argument('--volume_mn', type=float, default=0.10)
    parser.add_argument('--volume_mx', type=float, default=0.60)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--save_per_epoch', type=int, default=10)
    parser.add_argument('--eval_per_epoch', type=int, default=1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    """if args.exp_name == '/data/zym/experiment/bbox_tmi/DEBUG':
        myMakedirs(args.exp_name, overwrite=True)
    else:
        myMakedirs(args.exp_name, overwrite=False)

    # save code
    py_path_old = os.path.dirname(os.path.abspath(sys.argv[0]))
    py_path_new = os.path.join(args.exp_name, 'code')
    shutil.copytree(py_path_old, py_path_new)"""

    main()


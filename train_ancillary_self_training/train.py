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
from net import Unet
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

    net = Unet(1, args.num_classes).cuda()
    net.load_state_dict(torch.load(args.checkpoint))

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
    weights = torch.tensor([1,7,7], dtype=torch.float32).cuda() 
    CE = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=3)
    LogBarrier = LogBarrierLoss(t=5)
    REG = CRFLoss(alpha=15, beta=0.05, is_da=False, use_norm=False)

    iter_num = 0
    max_epoch = int(args.max_epoch)
    for epoch_num in range(max_epoch):
        print(f"\n=====Epoch: {epoch_num}====")
        loss_1, loss_2, loss_3, count = 0, 0, 0, 0
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

            fg_outs = net(fg_img, fg_seg_float)
            bg_outs = net(bg_img, bg_seg_float)

            segs = torch.cat((fg_seg, bg_seg), dim=0)   # 2 - inside bbox, 0 - outside bbox
            outs = torch.cat((fg_outs, bg_outs), dim=0)

            if np.unique(cor_seg.data.cpu().numpy()).size == 1 and np.unique(cor_seg.data.cpu().numpy())[0] == 0:
                l_ce = CE(outs, segs)
                loss = l_ce
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(list(net.parameters()), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()
                writer.add_scalar('loss/L_supervised', l_ce.item(), iter_num)
                logging.info(f"Epoch: {epoch_num} | Iter num: {iter_num} | Loss/L_supervised: {l_ce.item():.3f}")
            else:
                outs_sm_fg = outs.softmax(1)[:, 1, ...] + outs.softmax(1)[:, 2, ...] # Accounting for 1 - kidney and 2 - tumor

                bbox_outs_sm_fg = outs_sm_fg * cor_seg    # 1 - inside bbox, 0 - outside box

                outs_sm_proj_0 = bbox_outs_sm_fg.sum((2, 3))
                outs_sm_proj_1 = bbox_outs_sm_fg.sum((1, 3))
                outs_sm_proj_2 = bbox_outs_sm_fg.sum((1, 2))
                z0 = projection_0.sum() - outs_sm_proj_0[projection_0 == 1].sum()
                z1 = projection_1.sum() - outs_sm_proj_1[projection_1 == 1].sum()
                z2 = projection_2.sum() - outs_sm_proj_2[projection_2 == 1].sum()

                l_ce = CE(outs, segs)
                l_proj = 0.05 * torch.abs(LogBarrier.penalty(z0) + LogBarrier.penalty(z1) + LogBarrier.penalty(z2))
                # pseudo label
                l_pseudo = 0.
                for i in range(args.batch_size):
                    fg_outs_one = fg_outs[i:i+1]
                    fg_seg_one = fg_seg[i:i+1]
                    seed_outs = fg_outs_one.softmax(1).permute(0, 2, 3, 4, 1).contiguous()[fg_seg_one == 1]  # after softmax
                    fg_seed_mask = seed_outs[:, 1] > 0.95
                    bg_seed_mask = seed_outs[:, 1] < 0.05

                    seed_outs = fg_outs_one.permute(0, 2, 3, 4, 1).contiguous()[fg_seg_one == 1]  # before softmax
                    fg_seed_outs = seed_outs[fg_seed_mask]
                    bg_seed_outs = seed_outs[bg_seed_mask]

                    fg_select_idx = torch.randperm(fg_seed_outs.shape[0])[:int(fg_seed_outs.shape[0] * args.ratio)]
                    bg_select_idx = torch.randperm(bg_seed_outs.shape[0])[:int(bg_seed_outs.shape[0] * args.ratio)]

                    if not fg_select_idx.shape[0] == 0:
                        fg_seed_outs = fg_seed_outs[fg_select_idx]
                        l_pseudo_fg = CE(fg_seed_outs, torch.LongTensor(np.ones(fg_seed_outs.shape[0])).cuda())
                        l_pseudo = l_pseudo + l_pseudo_fg
                    if not bg_select_idx.shape[0] == 0:
                        bg_seed_outs = bg_seed_outs[bg_select_idx]
                        l_pseudo_bg = CE(bg_seed_outs, torch.LongTensor(np.zeros(bg_seed_outs.shape[0])).cuda())
                        l_pseudo = l_pseudo + l_pseudo_bg
                l_pseudo = (1. / args.batch_size) * l_pseudo

                loss = l_ce + l_proj + l_pseudo
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(list(net.parameters()), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()

                # finetune
                fg_outs = net(fg_img, fg_seg_float)
                bg_outs = net(bg_img, bg_seg_float)

                segs = torch.cat((fg_seg, bg_seg), dim=0)  # 2 - inside bbox, 0 - outside bbox
                outs = torch.cat((fg_outs, bg_outs), dim=0)

                outs_sm_fg = outs.softmax(1)[:, 1, ...] + outs.softmax(1)[:, 2, ...] # Accounting for 1 - kidney and 2 - tumor

                bbox_outs_sm_fg = outs_sm_fg * cor_seg  # 1 - inside bbox, 0 - outside box
                outs_sm_proj_0 = bbox_outs_sm_fg.sum((2, 3))
                outs_sm_proj_1 = bbox_outs_sm_fg.sum((1, 3))
                outs_sm_proj_2 = bbox_outs_sm_fg.sum((1, 2))
                z0 = projection_0.sum() - outs_sm_proj_0[projection_0 == 1].sum()
                z1 = projection_1.sum() - outs_sm_proj_1[projection_1 == 1].sum()
                z2 = projection_2.sum() - outs_sm_proj_2[projection_2 == 1].sum()

                l_crf = 0.001 / np.prod(args.patch_size) * REG(fg_img, fg_outs)
                l_ce = CE(outs, segs)
                l_proj = 0.05 * torch.abs(LogBarrier.penalty(z0) + LogBarrier.penalty(z1) + LogBarrier.penalty(z2))

                loss = l_ce + l_proj + l_crf
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(list(net.parameters()), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar('loss/L_supervised', l_ce.item(), iter_num)
                writer.add_scalar('loss/L_proj', l_proj.item(), iter_num)
                writer.add_scalar('loss/L_crf', l_crf.item(), iter_num)
                loss_1+=l_ce.item()
                loss_2+=l_proj.item()
                loss_3+=l_crf.item()
                count+=1

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

        logging.info(f"\nEpoch: {epoch_num} | Loss/L_supervised: {loss_1/count:3f} | Loss/L_proj: {loss_2/count:.3f} | Loss/L_crf: {loss_3/count:.3f}") 
        lr_ = args.base_lr * (1 - epoch_num / max_epoch) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        """# save
        if epoch_num % args.save_per_epoch == 0:
            save_model_path = os.path.join(args.exp_name, f'epoch_{epoch_num}.pth')
            torch.save(net.state_dict(), save_model_path)"""

        # eval
        if epoch_num % args.eval_per_epoch == 0:
            writer, (dice_kidney, dice_tumor), (jc_kidney, jc_tumor), (precision_kidney, precision_tumor), (recall_kidney, recall_tumor) = validate(net, eval_dataloader,
                                                                               args.patch_size, args.num_classes,
                                                                               logging, writer, iter_num, epoch_num)
            """if eval_dice > best_eval_dice and eval_jc > best_eval_jc:
                best_eval_dice = eval_dice
                best_eval_jc = eval_jc
                writer.add_scalar('eval_best/dice', eval_dice, epoch_num)
                writer.add_scalar('eval_best/jc', eval_jc, epoch_num)
                writer.add_scalar('eval_best/precision', eval_precision, epoch_num)
                writer.add_scalar('eval_best/recall', eval_recall, epoch_num)
                save_model_path = os.path.join(args.exp_name, 'epoch_best.pth')
                torch.save(net.state_dict(), save_model_path)"""
            logging.info(f"\nEpoch: {epoch_num} | Kidney Dice score: {dice_kidney:.3f} | Tumor Dice score: {dice_tumor:.3f}")
    writer.close()

    save_model_path = os.path.join(args.exp_name, f'epoch_{max_epoch}.pth')
    torch.save(net.state_dict(), save_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='/root/autodl-tmp/Kim/weakly_box_breast_cancer_seg/train_ancillary_self_training')
    # parser.add_argument('--exp_name', type=str, default='/data/zym/experiment/bbox_tmi/DEBUG')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_size', type=list, default=[96, 128, 128])
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--volume_mn', type=float, default=0.10)
    parser.add_argument('--volume_mx', type=float, default=0.60)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--checkpoint', type=str, default='/root/autodl-tmp/Kim/weakly_box_breast_cancer_seg/train_ancillary_init/epoch_20.pth')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--save_per_epoch', type=int, default=10)
    parser.add_argument('--eval_per_epoch', type=int, default=5)
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


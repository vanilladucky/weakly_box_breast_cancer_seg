import torch
import torch.nn.functional as F
import numpy as np
from medpy import metric
import math
from tqdm import tqdm

def validate(net, eval_dataloader,patch_size, num_classes, logging, writer, iter_num, epoch_num):
    all_case_dice = []
    all_case_jc = []
    all_case_precision = []
    all_case_recall = []

    for sampled_batch in tqdm(eval_dataloader):
        net.eval()
        c1, seg = sampled_batch['image'], sampled_batch['gt']
        c1 = c1.numpy()
        seg = seg.numpy()
        c1 = np.squeeze(c1, axis=0)
        seg = np.squeeze(seg, axis=0)

        w, h, d = seg.shape

        # if the size of image is less than patch_size, then padding it
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0] - w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1] - h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2] - d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
        if add_pad:
            c1 = np.pad(c1, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        ww, hh, dd = c1.shape

        stride_x = int(patch_size[0] / 2)
        stride_y = int(patch_size[1] / 2)
        stride_z = int(patch_size[2] / 2)
        sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        score_map = np.zeros((num_classes,) + c1.shape).astype(np.float32)
        cnt = np.zeros(c1.shape).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_x * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_y * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    c1_patch = c1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    c1_patch = np.expand_dims(c1_patch, axis=(0, 1)).astype(np.float32)
                    c1_patch = torch.from_numpy(c1_patch).cuda()
                    with torch.no_grad():
                        pred = net(c1_patch)
                        y1 = F.softmax(pred, dim=1)
                    y1 = y1.cpu().data.numpy()
                    y1 = y1[0, :, :, :, :]
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y1
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
        score_map = score_map / np.expand_dims(cnt, axis=0)
        if add_pad:
            score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        label_map = np.argmax(score_map, axis=0)

        dice = metric.binary.dc(label_map, seg)
        jc = metric.binary.jc(label_map, seg)
        precision = metric.binary.precision(label_map, seg)
        recall = metric.binary.recall(label_map, seg)
        all_case_dice.append(dice)
        all_case_jc.append(jc)
        all_case_precision.append(precision)
        all_case_recall.append(recall)

    mean_dice = np.array(all_case_dice).mean()
    mean_jc = np.array(all_case_jc).mean()
    mean_precision = np.array(all_case_precision).mean()
    mean_recall = np.array(all_case_recall).mean()
    logging.info('dice: {}; jc: {}; precision: {}; recall: {};'.format(mean_dice, mean_jc, mean_precision, mean_recall))
    writer.add_scalar('eval/dice', mean_dice, epoch_num)
    writer.add_scalar('eval/jc', mean_jc, epoch_num)
    writer.add_scalar('eval/precision', mean_precision, epoch_num)
    writer.add_scalar('eval/recall', mean_recall, epoch_num)
    return writer, mean_dice, mean_jc, mean_precision, mean_recall

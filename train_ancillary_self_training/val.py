import torch
import torch.nn.functional as F
import numpy as np
from medpy import metric
import math
from tqdm import tqdm

def validate(net, eval_dataloader,patch_size, num_classes, logging, writer, iter_num, epoch_num):
    dice_kidney, dice_tumor = [], []
    jc_kidney, jc_tumor = [], []
    prec_kidney, prec_tumor = [], []
    rec_kidney, rec_tumor = [], []

    for sampled_batch in tqdm(eval_dataloader):
        net.eval()
        c1, bbox, seg = sampled_batch['image'], sampled_batch['label'], sampled_batch['gt']
        c1 = c1.numpy()
        bbox = bbox.numpy()
        seg = seg.numpy()
        c1 = np.squeeze(c1, axis=0)
        bbox = np.squeeze(bbox, axis=0)
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
            bbox = np.pad(bbox, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
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
                    bbox_patch = bbox[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    c1_patch = np.expand_dims(c1_patch, axis=(0, 1)).astype(np.float32)
                    bbox_patch = np.expand_dims(bbox_patch, axis=(0, 1)).astype(np.float32)
                    c1_patch = torch.from_numpy(c1_patch).cuda()
                    bbox_patch = torch.from_numpy(bbox_patch).cuda()
                    with torch.no_grad():
                        pred = net(c1_patch, bbox_patch)
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
        # Get value counts for model output
        unique, counts = np.unique(label_map, return_counts=True)
        value_counts = dict(zip(unique, counts))
        print(f"Value count for validation model output: {value_counts}")

        # Get value counts for ground truth
        unique, counts = np.unique(seg, return_counts=True)
        value_counts = dict(zip(unique, counts))
        print(f"Value count for validation ground truth: {value_counts}")

        # Calculating metrics
        for cls, dice_list, jc_list, p_list, r_list in [
            (1, dice_kidney, jc_kidney, prec_kidney, rec_kidney),
            (2, dice_tumor,  jc_tumor,  prec_tumor,  rec_tumor)
        ]:
            pred_bin = (label_map == cls).astype(np.uint8)
            gt_bin   = (seg      == cls).astype(np.uint8)
            dice_list.append(    metric.binary.dc(pred_bin, gt_bin)    )
            jc_list.append(      metric.binary.jc(pred_bin, gt_bin)    )
            p_list.append(       metric.binary.precision(pred_bin, gt_bin) )
            r_list.append(       metric.binary.recall(pred_bin, gt_bin)    )

    mean_dk = np.mean(dice_kidney)
    mean_dt = np.mean(dice_tumor)
    mean_jk = np.mean(jc_kidney)
    mean_jt = np.mean(jc_tumor)
    mean_pk = np.mean(prec_kidney)
    mean_pt = np.mean(prec_tumor)
    mean_rk = np.mean(rec_kidney)
    mean_rt = np.mean(rec_tumor)

    # log separately
    writer.add_scalar('eval/dice_kidney', mean_dk, epoch_num)
    writer.add_scalar('eval/dice_tumor',  mean_dt, epoch_num)
    writer.add_scalar('eval/jc_kidney',   mean_jk, epoch_num)
    writer.add_scalar('eval/jc_tumor',    mean_jt, epoch_num)
    writer.add_scalar('eval/prec_kidney', mean_pk, epoch_num)
    writer.add_scalar('eval/prec_tumor',  mean_pt, epoch_num)
    writer.add_scalar('eval/recall_kidney', mean_rk, epoch_num)
    writer.add_scalar('eval/recall_tumor',  mean_rt, epoch_num)
    return writer, (mean_dk, mean_dt), (mean_jk, mean_jt), (mean_pk, mean_pt), (mean_rk, mean_rt)
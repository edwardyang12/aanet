import torch
import numpy as np

EPSILON = 1e-8


def epe_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        epe = np.mean(np.abs(d_est - d_gt))
    else:
        epe = torch.mean(torch.abs(d_est - d_gt))

    return epe

# bad 1.0, bad 2.0
def bad(d_est, d_gt, mask, threshold=1, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    bad = []
    if use_np:
        e = np.abs(d_gt - d_est)
        err_mask = (e>threshold)
        bad = np.mean(err_mask.astype('float'))
    else:
        e = torch.abs(d_gt - d_est)
        err_mask = (e>threshold)
        bad = torch.mean(err_mask.float())
    return bad

# 2mm, 4mm, 8mm
def mm_error(depth_est, depth_gt, mask, threshold=2, use_np=False):
    d_est, d_gt = depth_est[mask], depth_gt[mask]
    bad = []
    if use_np:
        e = np.abs(d_gt - d_est)
        err_mask = (e>threshold*16./3.)
        bad = np.mean(err_mask.astype('float'))
    else:
        e = torch.abs(d_gt - d_est)
        err_mask = (e>threshold*16./3.)
        bad = torch.mean(err_mask.float())
    return bad


def d1_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = (e > 3) & (e / d_gt > 0.05)

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres # don't need 3/16 anymore?

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean

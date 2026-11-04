"""
Metrici standard KITTI pentru adâncime (ca în Monodepth / Monodepth2).

Evaluarea se face la rezoluția ORIGINALĂ (375 x 1242), față de LiDAR,
cu decuparea Garg și adâncimi între 1 mm și 80 m.
"""
import numpy as np
from PIL import Image

import config as C
from student_model import decode_disparity

NAMES = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


def depth_metrics(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return np.array([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])


def q_to_depth_full(q_small, meta):
    """Disparitate normalizată (orice rezoluție) -> adâncime în metri la rezoluția originală."""
    w, h = meta["orig_w"], meta["orig_h"]
    q = np.asarray(Image.fromarray(q_small.astype(np.float32)).resize((w, h), Image.BILINEAR))
    q = np.clip(q, C.Q_MIN, C.Q_MAX)
    return meta["fx"] * meta["baseline"] / (q * w)


def evaluate(q_list, gt_list, meta, median_scaling=False):
    """
    q_list: disparități normalizate prezise; gt_list: adâncimi LiDAR (0 = lipsă).
    median_scaling=False: scara metrică vine din geometria stereo (baseline-ul
    real). Este raportarea principală pentru noi.
    """
    errs = []
    for q, gt in zip(q_list, gt_list):
        gt = gt.astype(np.float32)
        pred = q_to_depth_full(q, meta)
        h, w = gt.shape
        crop = np.zeros_like(gt, dtype=bool)
        crop[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
        m = (gt > C.MIN_DEPTH) & (gt < C.MAX_DEPTH) & crop
        p, g = pred[m], gt[m]
        if median_scaling:
            p = p * np.median(g) / np.median(p)
        p = np.clip(p, C.MIN_DEPTH, C.MAX_DEPTH)
        errs.append(depth_metrics(g, p))
    return dict(zip(NAMES, np.mean(errs, 0).tolist()))


def s_batch_to_q(s_batch):
    """Ieșirea rețelei [N,48,160,1] -> listă de q [48,160]."""
    return [decode_disparity(s[..., 0]) for s in s_batch]


def format_row(name, m):
    return (f"| {name} | {m['abs_rel']:.3f} | {m['sq_rel']:.3f} | {m['rmse']:.3f} | "
            f"{m['rmse_log']:.3f} | {m['a1']:.3f} | {m['a2']:.3f} | {m['a3']:.3f} |")


TABLE_HEADER = ("| Model | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |\n"
                "|---|---|---|---|---|---|---|---|")

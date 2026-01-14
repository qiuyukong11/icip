import os
import cv2
import numpy as np
import random
import torch

import matplotlib.pyplot as plt
from PIL import Image


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


def save_debug_vis(img, coarse_mask, sam3_mask, query_mask, support, support_mask, save_path):
    """
    query_img_path: 原始图片路径
    coarse_mask: FSS 生成的粗略掩码 (Numpy [H, W])
    sam3_mask: SAM3 生成的精细掩码 (Numpy [H, W])
    query_mask: Ground Truth 真实掩码 (Numpy [H, W])
    """
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Query Image")
    
    axes[1].imshow(query_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    
    axes[2].imshow(coarse_mask, cmap='gray')
    axes[2].set_title("Coarse Mask (FSS)")
    
    axes[3].imshow(sam3_mask, cmap='gray')
    axes[3].set_title("SAM3 Refined Mask")
    
    axes[4].imshow(support)
    axes[4].set_title("Support Image")
    
    axes[5].imshow(support_mask, cmap='gray')
    axes[5].set_title("Support Mask")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def mask_to_boxes_and_save(
    mask,
    score_map=None,           # 同形状 (H,W) 的前景概率或 logits，建议传概率
    save_path="mask_with_boxes.png",
    min_area=20,
    erode_kernel=7,
    score_mode="mean",        # "mean" 或 "max"
    score_thresh=0.5,         # 过滤阈值（概率用 0~1，logits 需相应调节）
    top_k=None,               # 只要分数最高的 K 个
):
    """
    逻辑：mask 做种子腐蚀切断细连接 → watershed 在原 mask 上分块（不缩小区域） → 按 score_map 给每块打分过滤。
    建议 score_map 传前景概率（softmax/sigmoid 后），不要传二值。
    """

    if hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()
    mask = (mask > 0).astype(np.uint8)

    if score_map is not None and hasattr(score_map, "cpu"):
        score_map = score_map.detach().cpu().numpy()

    # 1) 种子：腐蚀切断细连接
    seeds = mask.copy()
    if erode_kernel > 1:
        k = np.ones((erode_kernel, erode_kernel), np.uint8)
        seeds = cv2.erode(seeds, k, iterations=1)
    if seeds.sum() == 0:
        seeds = mask.copy()

    # 2) watershed：mask 作为限制，保持原大小分裂大块
    num_seed, seed_labels = cv2.connectedComponents(seeds, connectivity=8)
    markers = seed_labels + 1  # 背景 0，前景标签从 1
    markers[mask == 0] = 0
    ws_img = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(ws_img, markers)

    # 3) 收集前景区域并打分
    vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    boxes_scores = []

    def comp_score(component_mask):
        if score_map is None:
            return 1.0
        vals = score_map[component_mask > 0]
        if vals.size == 0:
            return 0.0
        if score_mode == "max":
            return float(vals.max())
        return float(vals.mean())

    for label_id in np.unique(markers):
        if label_id <= 1:  # 0 背景，-1 边界
            continue
        component = (markers == label_id).astype(np.uint8)
        area = int(component.sum())
        if area < min_area:
            continue

        s = comp_score(component)
        if s < score_thresh:
            continue

        ys, xs = np.where(component > 0)
        x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        boxes_scores.append(([x, y, w, h], s))

    # 4) top-k
    if top_k is not None and len(boxes_scores) > top_k:
        boxes_scores = sorted(boxes_scores, key=lambda x: x[1], reverse=True)[:top_k]

    boxes = []
    for (x, y, w, h), s in boxes_scores:
        boxes.append([x, y, w, h])
        cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 2)
        cv2.putText(vis, f"{s:.2f}", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, vis)

    return boxes#, [s for _, s in boxes_scores]
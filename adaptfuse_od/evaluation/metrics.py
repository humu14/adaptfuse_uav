"""
Evaluation Metrics — mAP computation and model comparison
==========================================================
Computes COCO-style mAP@50 and mAP@50:95 from predictions and ground truth.
Used by TorchVision and EfficientDet wrappers (Ultralytics has built-in mAP).
"""

import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    Boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


def compute_iou_matrix(boxes1, boxes2):
    """Compute IoU matrix between two sets of boxes [N, 4] and [M, 4]."""
    n = len(boxes1)
    m = len(boxes2)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])

    return iou_matrix


def compute_ap(recalls, precisions):
    """Compute Average Precision using the 101-point interpolation."""
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation (COCO style)
    recall_levels = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_levels:
        mask = recalls >= r
        if mask.any():
            ap += precisions[mask].max()
    ap /= 101.0

    return ap


def compute_ap_per_class(predictions, targets, iou_threshold=0.5, class_id=1):
    """
    Compute AP for a single class at a given IoU threshold.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        targets: List of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        class_id: Which class to evaluate
    
    Returns:
        AP value (float), number of GT objects
    """
    # Collect all predictions and ground truths for this class
    all_pred_scores = []
    all_pred_matched = []
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        # Filter predictions for this class
        pred_mask = pred["labels"].numpy() == class_id if hasattr(pred["labels"], 'numpy') \
                    else np.array(pred["labels"]) == class_id
        pred_boxes = pred["boxes"].numpy()[pred_mask] if hasattr(pred["boxes"], 'numpy') \
                     else np.array(pred["boxes"])[pred_mask]
        pred_scores = pred["scores"].numpy()[pred_mask] if hasattr(pred["scores"], 'numpy') \
                      else np.array(pred["scores"])[pred_mask]

        # Filter GT for this class
        gt_mask = gt["labels"].numpy() == class_id if hasattr(gt["labels"], 'numpy') \
                  else np.array(gt["labels"]) == class_id
        gt_boxes = gt["boxes"].numpy()[gt_mask] if hasattr(gt["boxes"], 'numpy') \
                   else np.array(gt["boxes"])[gt_mask]

        num_gt = len(gt_boxes)
        total_gt += num_gt

        if len(pred_boxes) == 0:
            continue

        if num_gt == 0:
            # All predictions are false positives
            for s in pred_scores:
                all_pred_scores.append(s)
                all_pred_matched.append(False)
            continue

        # Compute IoU matrix
        iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)

        # Sort predictions by score (descending)
        sorted_idx = np.argsort(-pred_scores)
        gt_matched = np.zeros(num_gt, dtype=bool)

        for pi in sorted_idx:
            all_pred_scores.append(pred_scores[pi])

            # Find best matching GT
            best_iou = 0
            best_gt = -1
            for gi in range(num_gt):
                if gt_matched[gi]:
                    continue
                if iou_mat[pi, gi] > best_iou:
                    best_iou = iou_mat[pi, gi]
                    best_gt = gi

            if best_iou >= iou_threshold and best_gt >= 0:
                gt_matched[best_gt] = True
                all_pred_matched.append(True)
            else:
                all_pred_matched.append(False)

    if total_gt == 0:
        return 0.0, 0

    if len(all_pred_scores) == 0:
        return 0.0, total_gt

    # Sort by score
    sorted_idx = np.argsort(-np.array(all_pred_scores))
    matched = np.array(all_pred_matched)[sorted_idx]

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(matched)
    fp_cumsum = np.cumsum(~matched)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = compute_ap(recalls, precisions)
    return ap, total_gt


def compute_map(predictions, targets, iou_thresholds=None, num_classes=2):
    """
    Compute mAP at specified IoU thresholds.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        targets: List of dicts with 'boxes', 'labels'
        iou_thresholds: List of IoU thresholds (default: [0.5])
        num_classes: Number of object classes (excluding background)
    
    Returns:
        Dict with mAP@50, mAP@50:95, per-class AP
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    class_names = {1: "fire", 2: "smoke"}  # class_id → name

    results = {}

    # mAP@50
    ap_per_class_50 = {}
    for cls_id in range(1, num_classes + 1):
        ap, n_gt = compute_ap_per_class(predictions, targets,
                                         iou_threshold=0.5, class_id=cls_id)
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        ap_per_class_50[cls_name] = ap
        results[f"AP50_{cls_name}"] = ap

    results["mAP50"] = np.mean(list(ap_per_class_50.values())) if ap_per_class_50 else 0.0

    # mAP@50:95
    iou_range = np.arange(0.5, 1.0, 0.05)
    ap_per_iou = []
    for iou_thresh in iou_range:
        aps = []
        for cls_id in range(1, num_classes + 1):
            ap, _ = compute_ap_per_class(predictions, targets,
                                          iou_threshold=iou_thresh, class_id=cls_id)
            aps.append(ap)
        ap_per_iou.append(np.mean(aps))

    results["mAP50_95"] = np.mean(ap_per_iou) if ap_per_iou else 0.0

    # Per-class precision and recall at IoU=0.5 (for reporting)
    for cls_id in range(1, num_classes + 1):
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        # Quick precision/recall at conf=0.25
        tp = 0
        fp = 0
        total_gt = 0
        for pred, gt in zip(predictions, targets):
            pred_labels = pred["labels"].numpy() if hasattr(pred["labels"], 'numpy') \
                          else np.array(pred["labels"])
            gt_labels = gt["labels"].numpy() if hasattr(gt["labels"], 'numpy') \
                        else np.array(gt["labels"])
            total_gt += np.sum(gt_labels == cls_id)

        results[f"n_gt_{cls_name}"] = int(total_gt)

    return results


def compute_fps(model, img_size=640, num_warmup=10, num_test=100, device="cuda"):
    """
    Measure inference FPS.
    
    Args:
        model: PyTorch model
        img_size: Input image size
        num_warmup: Number of warmup iterations
        num_test: Number of test iterations
        device: Device to benchmark on
    
    Returns:
        FPS (float)
    """
    import torch

    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(num_test):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    fps = num_test / elapsed

    return fps

"""
IoU-Reliability NMS — Trust-Weighted Multi-Modal NMS
=====================================================
Novel Contribution #5: Extends standard Non-Maximum Suppression (NMS) to
incorporate per-detection reliability scores from D-RUE across modalities.

Standard NMS sorts boxes solely by confidence score:
    score = P(class) * obj_conf

IoU-Reliability NMS sorts boxes by effective trust score:
    trust_score = P(class) * obj_conf * (reliability ** alpha)

Where:
    - reliability \in [0, 1] comes from D-RUE
    - alpha \ge 0 controls the strength of reliability weighting

When a high-confidence RGB box has low reliability (e.g. smoke occluded by dust),
its trust score drops, allowing a high-reliability Thermal box to be retained as the primary detection.
"""

import torch
import torchvision.ops as ops
from typing import Dict, List, Tuple


def iou_reliability_nms(boxes: torch.Tensor, scores: torch.Tensor,
                        reliability: torch.Tensor, iou_threshold: float = 0.45,
                        alpha: float = 1.0) -> torch.Tensor:
    """
    Perform IoU-Reliability NMS.
    
    Args:
        boxes: (N, 4) bounding box coordinates [x1, y1, x2, y2]
        scores: (N,) confidence scores
        reliability: (N,) D-RUE per-box reliability scores in [0, 1]
        iou_threshold: IoU overlap threshold for suppression
        alpha: Exponent for reliability weighting
    
    Returns:
        keep_indices: Indices of retained bounding boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # Compute trust-weighted scores
    trust_scores = scores * (reliability.squeeze() ** alpha)

    # Perform NMS using trust scores instead of raw confidence
    keep = ops.nms(boxes, trust_scores, iou_threshold)
    return keep


if __name__ == "__main__":
    boxes = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.85, 0.7], dtype=torch.float32)
    rel = torch.tensor([0.2, 0.95, 0.8], dtype=torch.float32)  # Box 0 is high-conf but unreliable

    keep_std = ops.nms(boxes, scores, 0.45)
    keep_ir = iou_reliability_nms(boxes, scores, rel, 0.45, alpha=1.0)

    print("IoU-Reliability NMS Test:")
    print(f"  Standard NMS kept indices:        {keep_std.tolist()}")
    print(f"  IoU-Reliability NMS kept indices: {keep_ir.tolist()}")
    print("  Notice: Box 1 (reliable) suppressed Box 0 (unreliable) despite Box 0 having higher raw conf!")

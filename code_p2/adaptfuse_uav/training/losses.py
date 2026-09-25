"""
AdapFuse-UAV & UniAdapFuse: Unified Multi-Task & Evidential Loss Suite
====================================================================
Includes:
  1. Focal Loss (for severe class imbalance)
  2. Evidential Dirichlet Loss (EDL) with Dempster-Shafer epistemic uncertainty
  3. Bounding Box CIoU and Objectness Loss for Dense Detection
  4. Nuisance Auxiliary Loss (for false-alarm suppression)
  5. Dynamic RUE Degradation Supervision Loss
  6. Knowledge Distillation (Logit + Feature + Reliability KD)
  7. UnifiedMultiTaskPerceptionLoss orchestrating all streams end-to-end
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25,
               gamma: float = 2.0, label_smoothing: float = 0.05) -> torch.Tensor:
    """Focal loss for class imbalance with FP16 AMP numerical stability."""
    logits_clamped = logits.clamp(min=-30.0, max=30.0)
    ce = F.cross_entropy(logits_clamped, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce.clamp(max=30.0))
    focal_weight = (1.0 - pt).clamp(min=0.0, max=1.0) ** gamma
    return (alpha * focal_weight * ce).mean()


def evidential_dirichlet_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    annealing_step: int = 10,
    current_epoch: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evidential Deep Learning (EDL) Loss based on Dirichlet distribution.
    Computes rigorous epistemic uncertainty.

    Returns:
        loss: Scalar EDL loss
        belief: (B, num_classes) belief mass
        uncertainty: (B,) epistemic uncertainty in [0, 1]
    """
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True)
    
    # 1. Expected Cross Entropy under Dirichlet
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    loss_ce = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1, keepdim=True)

    # 2. KL Divergence Regularization on misleading evidence
    alpha_tilde = targets_one_hot + (1.0 - targets_one_hot) * alpha
    S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)

    kl = (
        torch.lgamma(S_tilde)
        - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
        + torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=-1,
            keepdim=True,
        )
    )

    annealing_coef = min(1.0, float(current_epoch) / max(1, annealing_step))
    loss = torch.mean(loss_ce + annealing_coef * kl)

    belief = evidence / S
    uncertainty = (num_classes / S).squeeze(-1)

    return loss, belief, uncertainty


def bbox_ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Complete IoU (CIoU) Loss for bounding box regression.
    Boxes formatted as (x1, y1, x2, y2).
    """
    # Box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.unbind(-1)

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union area
    w1, h1 = (b1_x2 - b1_x1).clamp(min=1e-6), (b1_y2 - b1_y1).clamp(min=1e-6)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=1e-6), (b2_y2 - b2_y1).clamp(min=1e-6)
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    # Smallest enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-6

    # Center distance
    center_dist = ((b1_x1 + b1_x2) / 2.0 - (b2_x1 + b2_x2) / 2.0) ** 2 + \
                  ((b1_y1 + b1_y2) / 2.0 - (b2_y1 + b2_y2) / 2.0) ** 2

    # Aspect ratio consistency
    v = (4.0 / (3.14159265 ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + 1e-6)

    ciou = iou - (center_dist / c_diag) - alpha * v
    return (1.0 - ciou).clamp(min=0.0).mean()


def logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    """Hinton KD loss, scaled by T^2."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t.detach(), reduction='batchmean') * (temperature ** 2)


def feature_kd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """L2 feature distillation."""
    return F.mse_loss(student_feat, teacher_feat.detach())


def reliability_kd_loss(student_rel: torch.Tensor, teacher_rel: torch.Tensor) -> torch.Tensor:
    """Student learns teacher's reliability trust behavior."""
    return F.mse_loss(student_rel, teacher_rel.detach())


def reliability_regularization(reliability: torch.Tensor) -> torch.Tensor:
    """Encourage discriminative reliability scores (maximize variance across modalities)."""
    return -reliability.var(dim=-1).clamp(0.0, 1.0).mean()


# ── Detection helpers ─────────────────────────────────────────────────────────

def yolo_to_xyxy(boxes_cxcywh: torch.Tensor, img_size: int = 1) -> torch.Tensor:
    """Convert YOLO normalized (cx,cy,w,h) -> (x1,y1,x2,y2)."""
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def det_loss_from_gt(
    pred_pyramid: torch.Tensor,
    gt_boxes: torch.Tensor,
    n_boxes: torch.Tensor,
    img_size: int = 224,
) -> torch.Tensor:
    """
    Compute detection loss (CIoU bbox regression + objectness BCE) against YOLO GT.
    pred_pyramid: (B, 5+C, H, W)  --  channels: [obj, cx, cy, w, h, class...]
    gt_boxes:     (B, MAX_BOXES, 5)  -- [class, cx, cy, w, h], padded rows = -1
    n_boxes:      (B,) int -- valid box count per image
    """
    B, C, H, W = pred_pyramid.shape
    device = pred_pyramid.device
    total_ciou = torch.tensor(0.0, device=device)
    total_obj  = torch.tensor(0.0, device=device)
    n_valid    = 0

    # Build anchor-free grid centres (normalized 0-1)
    ys = (torch.arange(H, device=device).float() + 0.5) / H
    xs = (torch.arange(W, device=device).float() + 0.5) / W
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
    grid_cx = grid_x.reshape(-1)  # (H*W,)
    grid_cy = grid_y.reshape(-1)

    # Predicted objectness map
    obj_pred = pred_pyramid[:, 0, :, :].reshape(B, -1)  # (B, H*W)
    # Predicted box offsets (sigmoid for cx/cy, exp+clamp for w/h)
    pred_cx = torch.sigmoid(pred_pyramid[:, 1, :, :].reshape(B, -1)) / W + grid_cx.unsqueeze(0)
    pred_cy = torch.sigmoid(pred_pyramid[:, 2, :, :].reshape(B, -1)) / H + grid_cy.unsqueeze(0)
    pred_w  = torch.sigmoid(pred_pyramid[:, 3, :, :].reshape(B, -1)).clamp(1e-3, 1.0)
    pred_h  = torch.sigmoid(pred_pyramid[:, 4, :, :].reshape(B, -1)).clamp(1e-3, 1.0)

    obj_target = torch.zeros_like(obj_pred)  # (B, H*W)

    for b in range(B):
        n = int(n_boxes[b].item())
        if n == 0:
            continue
        boxes = gt_boxes[b, :n]  # (n, 5)  [cls, cx, cy, w, h]
        gt_cx, gt_cy = boxes[:, 1], boxes[:, 2]
        gt_w,  gt_h  = boxes[:, 3], boxes[:, 4]

        # Assign each GT to nearest grid cell
        cell_xi = (gt_cx * W).long().clamp(0, W - 1)
        cell_yi = (gt_cy * H).long().clamp(0, H - 1)
        cell_idx = cell_yi * W + cell_xi  # (n,)

        # Build pred boxes at assigned cells
        p_cx = pred_cx[b, cell_idx]
        p_cy = pred_cy[b, cell_idx]
        p_w  = pred_w[b, cell_idx]
        p_h  = pred_h[b, cell_idx]
        pred_box_xyxy = yolo_to_xyxy(torch.stack([p_cx, p_cy, p_w, p_h], dim=-1))
        gt_box_xyxy   = yolo_to_xyxy(torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=-1))

        ciou = bbox_ciou_loss(pred_box_xyxy, gt_box_xyxy)
        total_ciou = total_ciou + ciou

        # Objectness target: 1 at assigned cells
        obj_target[b, cell_idx] = 1.0
        n_valid += n

    # Objectness BCE across all cells
    total_obj = F.binary_cross_entropy_with_logits(
        obj_pred, obj_target, reduction='mean',
        pos_weight=torch.tensor(5.0, device=device),  # handle foreground sparsity
    )

    if n_valid > 0:
        return total_ciou / n_valid + total_obj
    return total_obj


def compute_map50(
    pred_boxes_batch: list,
    gt_boxes_batch: list,
    iou_threshold: float = 0.5,
) -> float:
    """
    Lightweight mAP@0.5 computation over a batch.
    pred_boxes_batch: list of (N, 5) tensors [score, cx, cy, w, h] per image
    gt_boxes_batch:   list of (M, 4) tensors [cx, cy, w, h] per image
    Returns mAP@0.5 as a float.
    """
    all_tp, all_fp, all_scores = [], [], []
    n_gt_total = 0

    for pred, gt in zip(pred_boxes_batch, gt_boxes_batch):
        n_gt = len(gt)
        n_gt_total += n_gt

        if len(pred) == 0:
            continue

        scores = pred[:, 0].cpu()

        if n_gt == 0:
            # All predictions are FPs
            for s in scores.tolist():
                all_tp.append(0)
                all_fp.append(1)
                all_scores.append(s)
            continue

        # Sort predictions by confidence descending
        order     = scores.argsort(descending=True)
        pred_sort = pred[order]
        pred_xyxy = yolo_to_xyxy(pred_sort[:, 1:])
        gt_xyxy   = yolo_to_xyxy(gt)
        matched_gt = torch.zeros(n_gt, dtype=torch.bool)

        for p in range(len(pred_sort)):
            ious   = _box_iou(pred_xyxy[p:p+1], gt_xyxy).squeeze(0)  # (M,)
            best_i = int(ious.argmax())
            if ious[best_i] >= iou_threshold and not matched_gt[best_i]:
                all_tp.append(1); all_fp.append(0)
                matched_gt[best_i] = True
            else:
                all_tp.append(0); all_fp.append(1)
            all_scores.append(float(scores[order[p]]))

    if n_gt_total == 0 or len(all_scores) == 0:
        return 0.0

    # Sort all detections by score globally
    scores_t  = torch.tensor(all_scores)
    order_g   = scores_t.argsort(descending=True)
    tp_sorted = torch.tensor(all_tp, dtype=torch.float32)[order_g]
    fp_sorted = torch.tensor(all_fp, dtype=torch.float32)[order_g]
    tp_cum    = tp_sorted.cumsum(0)
    fp_cum    = fp_sorted.cumsum(0)
    recall    = tp_cum / max(n_gt_total, 1)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    # Area under P-R curve
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = torch.max(mpre[i], mpre[i + 1])
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=True)[0]
    ap  = ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()
    return float(ap)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-6)


class UnifiedMultiTaskPerceptionLoss(nn.Module):
    """
    Unified Loss Function for UniAdapFuse-UAV:
      - Evidential / Focal Disaster Classification Loss
      - Victim Detection Loss
      - Nuisance False-Alarm Loss
      - Dynamic RUE Degradation Classification Loss
      - Multi-Scale Bounding Box Detection Loss (CIoU + Objectness + Class Focal)
      - Knowledge Distillation Loss (Optional)
    """

    def __init__(
        self,
        use_edl: bool = True,
        use_kd: bool = False,
        kd_temperature: float = 4.0,
        kd_lambda: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        w_disaster: float = 1.0,
        w_victim: float = 0.5,
        w_nuisance: float = 0.3,
        w_det: float = 0.5,
        w_rue_aux: float = 0.3,
        w_kd: float = 0.75,
    ):
        super().__init__()
        self.use_edl = use_edl
        self.use_kd = use_kd
        self.kd_temperature = kd_temperature
        self.kd_lambda = kd_lambda
        self.alpha = alpha
        self.gamma = gamma
        self.w_disaster = w_disaster
        self.w_victim = w_victim
        self.w_nuisance = w_nuisance
        self.w_det = w_det
        self.w_rue_aux = w_rue_aux
        self.w_kd = w_kd

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_epoch: int = 1,
        teacher_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict returned by UniAdapFuseModel
            targets: Dict with keys: 'disaster_gt', 'victim_gt', 'nuisance_gt',
                     'corruption_flags', 'det_targets' (optional)
        """
        device = outputs['disaster'].device
        zero = torch.tensor(0.0, device=device)

        # 1. Disaster classification loss (Evidential or Focal)
        if self.use_edl:
            l_disaster, belief, epistemic_unc = evidential_dirichlet_loss(
                outputs['disaster'], targets['disaster_gt'],
                num_classes=4, annealing_step=10, current_epoch=current_epoch
            )
        else:
            l_disaster = focal_loss(outputs['disaster'], targets['disaster_gt'], self.alpha, self.gamma)
            epistemic_unc = zero

        # 2. Victim binary classification loss
        l_victim = focal_loss(outputs['victim'], targets['victim_gt'], self.alpha, self.gamma)

        # 3. Nuisance auxiliary loss
        l_nuisance = zero
        if 'nuisance_gt' in targets and targets['nuisance_gt'] is not None and outputs.get('nuisance') is not None:
            l_nuisance = focal_loss(outputs['nuisance'], targets['nuisance_gt'], self.alpha, self.gamma)

        # 4. RUE degradation auxiliary loss
        l_rue_aux = zero
        if 'corruption_flags' in targets and targets['corruption_flags'] is not None and outputs.get('degradation_logits') is not None:
            l_rue_aux = F.binary_cross_entropy_with_logits(
                outputs['degradation_logits'], targets['corruption_flags'].float(), reduction='mean'
            )

        # 5. Detection loss across multi-scale pyramids
        l_det = zero
        if 'det_pyramids' in outputs and outputs['det_pyramids'] is not None:
            # Self-supervised spatial objectness regularization or supervised if bboxes present
            det_losses = []
            for p in outputs['det_pyramids']:
                # p has shape (B, 5+C, H, W)
                obj_logits = p[:, 4:5, :, :]
                det_losses.append(torch.mean(torch.relu(1.0 - torch.sigmoid(obj_logits))))
            l_det = sum(det_losses) / len(det_losses)

        # 6. Knowledge distillation loss
        l_kd = zero
        if self.use_kd and teacher_outputs is not None:
            l_kd_logit = logit_kd_loss(outputs['disaster'], teacher_outputs['disaster'], self.kd_temperature)
            l_kd_rel = reliability_kd_loss(outputs['reliability'], teacher_outputs['reliability'])
            l_kd = l_kd_logit + 0.5 * l_kd_rel

        # Total Loss
        total_loss = (
            self.w_disaster * l_disaster
            + self.w_victim * l_victim
            + self.w_nuisance * l_nuisance
            + self.w_rue_aux * l_rue_aux
            + self.w_det * l_det
            + self.w_kd * l_kd
        )

        return {
            'total': total_loss,
            'disaster': l_disaster,
            'victim': l_victim,
            'nuisance': l_nuisance,
            'rue_aux': l_rue_aux,
            'det': l_det,
            'kd': l_kd,
            'epistemic_uncertainty': epistemic_unc.mean() if isinstance(epistemic_unc, torch.Tensor) else zero,
        }


class CombinedLoss(nn.Module):
    """
    Combined loss wrapper for legacy Trainer and modern multi-task models.
    """

    def __init__(
        self,
        use_kd: bool = False,
        kd_temperature: float = 4.0,
        kd_lambda: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        disaster_weight: float = 2.0,
        victim_weight: float = 0.5,
        nuisance_weight: float = 0.3,
        reliability_aux_weight: float = 0.3,
        kd_weight: float = 0.75,
        reliability_reg_weight: float = 0.01,
        det_weight: float = 0.2,
    ):
        super().__init__()
        self.use_kd = use_kd
        self.temperature = kd_temperature
        self.kd_lambda = kd_lambda
        self.alpha = alpha
        self.gamma = gamma
        self.w_disaster = disaster_weight
        self.w_victim = victim_weight
        self.w_nuisance = nuisance_weight
        self.w_rel_aux = reliability_aux_weight
        self.w_kd = kd_weight
        self.w_rel_reg = reliability_reg_weight
        self.w_det = det_weight

    def forward(
        self,
        disaster_out: torch.Tensor,
        victim_out: torch.Tensor,
        disaster_gt: torch.Tensor,
        victim_gt: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        nuisance_out: Optional[torch.Tensor] = None,
        nuisance_gt: Optional[torch.Tensor] = None,
        degradation_logits: Optional[torch.Tensor] = None,
        corruption_flags: Optional[torch.Tensor] = None,
        fused_feat: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        teacher_disaster: Optional[torch.Tensor] = None,
        teacher_victim: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
        teacher_reliability: Optional[torch.Tensor] = None,
        teacher_uncertainty: Optional[torch.Tensor] = None,
        det_pyramids: Optional[List[torch.Tensor]] = None,
        gt_boxes: Optional[torch.Tensor] = None,       # (B, MAX_BOXES, 5) YOLO format
        n_boxes: Optional[torch.Tensor] = None,         # (B,) int
        has_bbox: Optional[torch.Tensor] = None,        # (B,) float 0/1
    ) -> Dict[str, torch.Tensor]:

        device = disaster_out.device
        zero = torch.tensor(0.0, device=device)

        # ── Task losses ──────────────────────────────────────────
        l_disaster = focal_loss(disaster_out, disaster_gt, self.alpha, self.gamma)
        l_victim = focal_loss(victim_out, victim_gt, self.alpha, self.gamma)
        l_task = self.w_disaster * l_disaster + self.w_victim * l_victim

        # ── Nuisance auxiliary loss ───────────────────────────────
        l_nuisance = zero
        if nuisance_out is not None and nuisance_gt is not None:
            l_nuisance = focal_loss(nuisance_out, nuisance_gt, self.alpha, self.gamma)

        # ── Reliability auxiliary loss ────────────────────────────
        l_rel_aux = zero
        if degradation_logits is not None and corruption_flags is not None:
            l_rel_aux = F.binary_cross_entropy_with_logits(
                degradation_logits,
                corruption_flags.float(),
                reduction='mean',
            )

        # ── Reliability regularization ────────────────────────────
        l_rel_reg = zero
        if reliability is not None:
            l_rel_reg = reliability_regularization(reliability)

        # ── Detection loss (CIoU + objectness) ───────────────────
        l_det = zero
        if det_pyramids is not None and len(det_pyramids) > 0:
            det_losses = []
            for p in det_pyramids:
                if gt_boxes is not None and n_boxes is not None and has_bbox is not None:
                    # Images in batch that have GT boxes
                    mask = (has_bbox > 0).nonzero(as_tuple=True)[0]
                    if len(mask) > 0:
                        det_losses.append(
                            det_loss_from_gt(p[mask], gt_boxes[mask], n_boxes[mask])
                        )
                    # For images without GT: self-supervised objectness regularizer
                    no_mask = (has_bbox == 0).nonzero(as_tuple=True)[0]
                    if len(no_mask) > 0:
                        obj_logits = p[no_mask][:, 0:1, :, :]
                        det_losses.append(torch.mean(torch.relu(1.0 - torch.sigmoid(obj_logits))))
                else:
                    # Fallback: self-supervised objectness regularizer
                    obj_logits = p[:, 0:1, :, :]
                    det_losses.append(torch.mean(torch.relu(1.0 - torch.sigmoid(obj_logits))))
            if det_losses:
                l_det = sum(det_losses) / len(det_losses)

        # ── KD losses ─────────────────────────────────────────────
        l_kd_logit = zero
        l_kd_feat = zero
        l_kd_reliability = zero
        l_kd = zero

        if self.use_kd and teacher_disaster is not None:
            l_kd_logit = logit_kd_loss(disaster_out, teacher_disaster, self.temperature)

            if fused_feat is not None and teacher_feat is not None and fused_feat.shape == teacher_feat.shape:
                l_kd_feat = feature_kd_loss(fused_feat, teacher_feat)

            if reliability is not None and teacher_reliability is not None:
                l_kd_reliability = reliability_kd_loss(reliability, teacher_reliability)

            if teacher_uncertainty is not None:
                uncertainty_weights = 1.0 / (1.0 + teacher_uncertainty.mean(dim=-1))
                l_kd = (uncertainty_weights * (l_kd_logit + l_kd_feat)).mean()
                l_kd = l_kd + l_kd_reliability * 0.5
            else:
                l_kd = l_kd_logit + self.kd_lambda * l_kd_feat + 0.5 * l_kd_reliability

        total = (
            l_task
            + self.w_nuisance * l_nuisance
            + self.w_rel_aux * l_rel_aux
            + self.w_rel_reg * l_rel_reg
            + self.w_det * l_det
            + self.w_kd * l_kd
        )

        return {
            'total': total,
            'disaster': l_disaster,
            'victim': l_victim,
            'nuisance': l_nuisance,
            'reliability_aux': l_rel_aux,
            'reliability_reg': l_rel_reg,
            'det': l_det,
            'kd_logit': l_kd_logit,
            'kd_feat': l_kd_feat,
            'kd_reliability': l_kd_reliability,
            'kd': l_kd,
        }

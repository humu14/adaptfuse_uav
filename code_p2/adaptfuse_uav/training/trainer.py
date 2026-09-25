"""
AdapFuse-UAV: Core Trainer class.
Handles train/val loops, logging, checkpointing, and LR scheduling.
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from training.losses import CombinedLoss, compute_map50, yolo_to_xyxy


def _autocast_context(device_type: str, enabled: bool):
    try:
        return autocast(device_type, enabled=enabled)
    except TypeError:
        return autocast(enabled=enabled)


class Trainer:
    """
    Multi-modal trainer with:
      - Mixed precision training (AMP)
      - Gradient clipping
      - Warmup + CosineAnnealing LR schedule
      - Best model checkpointing
      - Detailed per-epoch logging
      - Optional knowledge distillation
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        output_dir: str,
        teacher: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.teacher = teacher
        self.config = config
        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        # Feature projection adapter: student feat_dim → teacher feat_dim for KD
        self.feat_adapter = None
        if self.teacher is not None and config.get("use_kd", False):
            t_dim = getattr(self.teacher, "proj_dim", None)
            s_dim = getattr(self.model, "proj_dim", None)
            if t_dim is not None and s_dim is not None and t_dim != s_dim:
                self.feat_adapter = nn.Linear(s_dim, t_dim).to(self.device)

        # ── Loss ─────────────────────────────────────────────────
        self.criterion = CombinedLoss(
            use_kd=config.get("use_kd", False),
            kd_temperature=config.get("kd_temperature", 4.0),
            kd_lambda=config.get("kd_lambda", 0.5),
            disaster_weight=float(config.get("disaster_weight", 2.0)),
            victim_weight=float(config.get("victim_weight", 0.5)),
            nuisance_weight=float(config.get("nuisance_weight", 0.3)),
            det_weight=float(config.get("det_weight", 0.2)),
        )

        # ── Phased Training Config ────────────────────────────────
        self.phase1_epochs = int(config.get("phase1_epochs", 0))
        self.use_phased_training = self.phase1_epochs > 0

        # ── Optimizer with Separate LR Groups ─────────────────────
        base_lr = float(config.get("lr", 1e-3))
        backbone_lr_mult = float(config.get("backbone_lr_mult", 0.1))
        det_lr_mult = float(config.get("det_lr_mult", 0.3))

        # Build parameter groups if model supports it (unified model)
        if hasattr(self.model, 'set_training_phase'):
            backbone_params = []
            classification_params = []
            detection_params = []
            other_params = []

            # Categorize parameters by component
            backbone_names = {'rgb_encoder', 'thermal_encoder', 'audio_encoder'}
            det_names = {'det_neck', 'det_head', 'pyramid_fusers'}

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                top_module = name.split('.')[0]
                if top_module in backbone_names:
                    backbone_params.append(param)
                elif top_module in det_names:
                    detection_params.append(param)
                else:
                    classification_params.append(param)

            param_groups = [
                {'params': backbone_params, 'lr': base_lr * backbone_lr_mult, 'name': 'backbone'},
                {'params': classification_params, 'lr': base_lr, 'name': 'classification'},
                {'params': detection_params, 'lr': base_lr * det_lr_mult, 'name': 'detection'},
            ]
            # Filter out empty groups
            param_groups = [g for g in param_groups if len(g['params']) > 0]
        else:
            param_groups = [{'params': list(self.model.parameters()), 'lr': base_lr}]

        if self.feat_adapter is not None:
            param_groups.append({'params': list(self.feat_adapter.parameters()), 'lr': base_lr, 'name': 'feat_adapter'})

        self.optimizer = AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )

        # ── LR Scheduler (warmup + cosine) ───────────────────────
        epochs = int(config.get("epochs", 50))
        warmup = int(config.get("warmup_epochs", 5))
        min_lr = float(config.get("min_lr", 1e-5))

        warmup_sched = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup,
            eta_min=min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )

        # ── Mixed precision ────────────────────────────────────────
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── State ─────────────────────────────────────────────────
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history: List[Dict] = []
        self.exp_name = config.get("experiment_name", "experiment")
        self.dry_run_batches = config.get("dry_run_batches", None)

    # ─────────────────────────────────────────────────────────────

    def _batch_to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _forward(self, batch: Dict):
        """Forward pass, returns (disaster_out, victim_out, reliability)."""
        out = self.model(
            rgb=batch["rgb"],
            thermal=batch["thermal"],
            audio=batch["audio"],
            has_rgb=batch["has_rgb"],
            has_thermal=batch["has_thermal"],
            has_audio=batch["has_audio"],
        )
        if isinstance(out, dict):
            self.model._last_outputs = out
            return out["disaster"], out["victim"], out["reliability"]
        return out

    # ─────────────────────────────────────────────────────────────

    def train_epoch(self, loader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        loss_components = {"disaster": 0.0, "victim": 0.0, "kd_logit": 0.0, "kd_feat": 0.0}
        correct_d, correct_v, n_total = 0, 0, 0

        pbar = tqdm(loader, desc=f"[Train E{epoch}]", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            if self.dry_run_batches is not None and step >= self.dry_run_batches:
                break
            batch = self._batch_to_device(batch)
            disaster_gt = batch["disaster_label"]
            victim_gt = batch["victim_label"]

            # ── Forward ───────────────────────────────────────────
            with _autocast_context(self.device.type, self.use_amp):
                d_out, v_out, reliability = self._forward(batch)

                # Teacher for KD
                t_disaster = t_victim = t_feat = None
                if self.teacher is not None:
                    with torch.no_grad():
                        t_disaster, t_victim, _ = self.teacher(
                            rgb=batch["rgb"], thermal=batch["thermal"], audio=batch["audio"],
                            has_rgb=batch["has_rgb"], has_thermal=batch["has_thermal"],
                            has_audio=batch["has_audio"],
                        )
                    t_full = getattr(self.teacher, "get_full_outputs", lambda: {})()
                    t_feat = t_full.get("fused_feat")

                # Extra outputs for AdapFuseV1
                full_outs = getattr(self.model, "get_full_outputs", lambda: {})()
                student_feat = full_outs.get("fused_feat")
                if self.feat_adapter is not None and student_feat is not None:
                    student_feat = self.feat_adapter(student_feat)

                losses = self.criterion(
                    disaster_out=d_out, victim_out=v_out,
                    disaster_gt=disaster_gt, victim_gt=victim_gt,
                    reliability=reliability,
                    nuisance_out=full_outs.get("nuisance"),
                    nuisance_gt=None,
                    degradation_logits=full_outs.get("degradation_logits"),
                    corruption_flags=None,
                    fused_feat=student_feat,
                    uncertainty=full_outs.get("uncertainty"),
                    teacher_disaster=t_disaster, teacher_victim=t_victim,
                    teacher_feat=t_feat,
                    det_pyramids=full_outs.get("det_pyramids"),
                    gt_boxes=batch.get("bboxes"),
                    n_boxes=batch.get("n_boxes"),
                    has_bbox=batch.get("has_bbox"),
                )
                loss = losses["total"]

            # ── Backward with NaN/Inf Guard ─────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── Metrics ───────────────────────────────────────────
            bs = disaster_gt.size(0)
            total_loss += loss.item() * bs
            loss_components["disaster"] += losses["disaster"].item() * bs
            loss_components["victim"] += losses["victim"].item() * bs
            loss_components["kd_logit"] += losses["kd_logit"].item() * bs
            loss_components["kd_feat"] += losses["kd_feat"].item() * bs
            correct_d += (d_out.argmax(1) == disaster_gt).sum().item()
            correct_v += (v_out.argmax(1) == victim_gt).sum().item()
            n_total += bs

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "d_acc": f"{correct_d / n_total:.3f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        n = max(n_total, 1)
        return {
            "loss": total_loss / n,
            "disaster_loss": loss_components["disaster"] / n,
            "victim_loss": loss_components["victim"] / n,
            "kd_logit_loss": loss_components["kd_logit"] / n,
            "kd_feat_loss": loss_components["kd_feat"] / n,
            "disaster_acc": correct_d / n,
            "victim_acc": correct_v / n,
        }
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_epoch(
        self,
        loader,
        epoch: int,
        split: str = "val",
        return_predictions: bool = False,
    ):
        from sklearn.metrics import f1_score, accuracy_score

        self.model.eval()
        all_d_preds, all_d_gt = [], []
        all_v_preds, all_v_gt = [], []
        prediction_rows = []
        total_loss = 0.0
        valid_loss_samples = 0
        n_total = 0

        # mAP accumulation lists
        all_pred_boxes_batch = []  # list of (N,5) tensors per image
        all_gt_boxes_batch   = []  # list of (M,4) tensors per image

        pbar = tqdm(loader, desc=f"[{split.upper()} E{epoch}]", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            if self.dry_run_batches is not None and step >= self.dry_run_batches:
                break
            batch = self._batch_to_device(batch)
            disaster_gt = batch["disaster_label"]
            victim_gt = batch["victim_label"]

            with _autocast_context(self.device.type, self.use_amp):
                d_out, v_out, reliability = self._forward(batch)
                full_outs = getattr(self.model, "get_full_outputs", lambda: {})()
                losses = self.criterion(
                    disaster_out=d_out, victim_out=v_out,
                    disaster_gt=disaster_gt, victim_gt=victim_gt,
                    reliability=reliability,
                    nuisance_out=full_outs.get("nuisance"),
                    degradation_logits=full_outs.get("degradation_logits"),
                    fused_feat=full_outs.get("fused_feat"),
                    uncertainty=full_outs.get("uncertainty"),
                    det_pyramids=full_outs.get("det_pyramids"),
                    gt_boxes=batch.get("bboxes"),
                    n_boxes=batch.get("n_boxes"),
                    has_bbox=batch.get("has_bbox"),
                )

            bs = disaster_gt.size(0)
            batch_loss = losses["total"].item()
            if not (math.isnan(batch_loss) or math.isinf(batch_loss)):
                total_loss += batch_loss * bs
                valid_loss_samples += bs
            n_total += bs

            all_d_preds.extend(d_out.argmax(1).cpu().numpy())
            all_d_gt.extend(disaster_gt.cpu().numpy())
            all_v_preds.extend(v_out.argmax(1).cpu().numpy())
            all_v_gt.extend(victim_gt.cpu().numpy())

            if return_predictions:
                d_prob = torch.softmax(d_out.float(), dim=1).cpu().numpy()
                v_prob = torch.softmax(v_out.float(), dim=1).cpu().numpy()
                rel = reliability.float().cpu().numpy() if reliability is not None else None
                d_pred = d_out.argmax(1).cpu().numpy()
                v_pred = v_out.argmax(1).cpu().numpy()
                d_true = disaster_gt.cpu().numpy()
                v_true = victim_gt.cpu().numpy()
                for bi in range(bs):
                    row = {
                        "sample_id": str(batch["sample_id"][bi]),
                        "source": str(batch["source"][bi]),
                        "disaster_gt": int(d_true[bi]),
                        "disaster_pred": int(d_pred[bi]),
                        "victim_gt": int(v_true[bi]),
                        "victim_pred": int(v_pred[bi]),
                    }
                    for ci, value in enumerate(d_prob[bi]):
                        row[f"disaster_prob_{ci}"] = float(value)
                    for ci, value in enumerate(v_prob[bi]):
                        row[f"victim_prob_{ci}"] = float(value)
                    if rel is not None:
                        for mi, name in enumerate(("rgb", "pseudo_thermal", "audio")):
                            if mi < rel.shape[1]:
                                row[f"reliability_{name}"] = float(rel[bi, mi])
                    prediction_rows.append(row)

            # mAP: extract predictions from highest-res det pyramid
            det_pyramids = full_outs.get("det_pyramids")
            gt_boxes_b   = batch.get("bboxes")
            n_boxes_b    = batch.get("n_boxes")
            has_bbox_b   = batch.get("has_bbox")
            if det_pyramids is not None and gt_boxes_b is not None:
                p = det_pyramids[0]  # highest res: (B, 5+C, H, W)
                B2, C2, H2, W2 = p.shape
                obj_scores = torch.sigmoid(p[:, 0, :, :]).reshape(B2, -1)  # (B, H*W)
                ys = (torch.arange(H2, device=p.device).float() + 0.5) / H2
                xs = (torch.arange(W2, device=p.device).float() + 0.5) / W2
                gy, gx = torch.meshgrid(ys, xs, indexing='ij')
                gcx = gx.reshape(-1); gcy = gy.reshape(-1)
                pred_cx = torch.sigmoid(p[:, 1, :, :].reshape(B2, -1)) / W2 + gcx
                pred_cy = torch.sigmoid(p[:, 2, :, :].reshape(B2, -1)) / H2 + gcy
                pred_w  = torch.sigmoid(p[:, 3, :, :].reshape(B2, -1)).clamp(1e-3, 1.0)
                pred_h  = torch.sigmoid(p[:, 4, :, :].reshape(B2, -1)).clamp(1e-3, 1.0)

                for bi in range(B2):
                    # Keep top-30 predictions by objectness score
                    scores_i = obj_scores[bi]
                    topk_idx = scores_i.topk(min(30, len(scores_i))).indices
                    pred_entry = torch.stack([
                        scores_i[topk_idx],
                        pred_cx[bi][topk_idx],
                        pred_cy[bi][topk_idx],
                        pred_w[bi][topk_idx],
                        pred_h[bi][topk_idx],
                    ], dim=-1).cpu()
                    all_pred_boxes_batch.append(pred_entry)

                    # GT boxes for this image
                    n_gt = int(n_boxes_b[bi].item()) if n_boxes_b is not None else 0
                    if n_gt > 0 and has_bbox_b is not None and has_bbox_b[bi] > 0:
                        gt_entry = gt_boxes_b[bi, :n_gt, 1:].cpu()  # [cx,cy,w,h]
                        all_gt_boxes_batch.append(gt_entry)
                    else:
                        all_gt_boxes_batch.append(torch.zeros(0, 4))

        n = max(n_total, 1)
        d_f1 = f1_score(all_d_gt, all_d_preds, average="macro", zero_division=0)
        v_f1 = f1_score(all_v_gt, all_v_preds, average="macro", zero_division=0)
        d_acc = accuracy_score(all_d_gt, all_d_preds)
        v_acc = accuracy_score(all_v_gt, all_v_preds)

        avg_loss = total_loss / max(valid_loss_samples, 1) if valid_loss_samples > 0 else 0.0

        # mAP@0.5
        map50 = 0.0
        if all_pred_boxes_batch:
            map50 = compute_map50(all_pred_boxes_batch, all_gt_boxes_batch, iou_threshold=0.5)

        metrics = {
            "loss": avg_loss,
            "disaster_f1": d_f1,
            "victim_f1": v_f1,
            "disaster_acc": d_acc,
            "victim_acc": v_acc,
            "combined_f1": (d_f1 + v_f1) / 2,
            "map50": map50,
        }
        if return_predictions:
            return metrics, prediction_rows
        return metrics

    # ─────────────────────────────────────────────────────────────

    def _checkpoint_payload(self, epoch: int, metrics: Dict) -> Dict:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        if self.feat_adapter is not None:
            ckpt["feat_adapter_state"] = self.feat_adapter.state_dict()
        return ckpt

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        ckpt = self._checkpoint_payload(epoch, metrics)
        # Latest checkpoint
        torch.save(ckpt, self.ckpt_dir / f"{self.exp_name}_latest.pth")
        # Best checkpoint
        if is_best:
            torch.save(ckpt, self.ckpt_dir / f"{self.exp_name}_best.pth")
            print(f"  [SAVED] Best checkpoint saved (epoch {epoch}, f1={metrics.get('combined_f1', 0):.4f})")

    def save_final_checkpoint(self, epoch: int, metrics: Dict):
        path = self.ckpt_dir / f"{self.exp_name}_final.pth"
        torch.save(self._checkpoint_payload(epoch, metrics), path)
        print(f"  [SAVED] Final checkpoint: {path.name}")
        return path

    def _save_history(self):
        """Persist epoch logs incrementally so interrupted runs retain completed epochs."""
        history_path = self.log_dir / f"{self.exp_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        return history_path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        
        # Helper to recursively convert string parameters to floats
        def clean_state_dict_floats(state_dict):
            if isinstance(state_dict, dict):
                for k, v in list(state_dict.items()):
                    if k in ("eta_min", "min_lr", "lr", "initial_lr", "weight_decay", "factor", "start_factor", "end_factor"):
                        if isinstance(v, str):
                            try:
                                state_dict[k] = float(v)
                            except ValueError:
                                pass
                    elif k in ("base_lrs", "_last_lr") and isinstance(v, list):
                        state_dict[k] = [float(x) if isinstance(x, str) else x for x in v]
                    else:
                        clean_state_dict_floats(v)
            elif isinstance(state_dict, list):
                for item in state_dict:
                    clean_state_dict_floats(item)

        if "optimizer_state" in ckpt:
            clean_state_dict_floats(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            clean_state_dict_floats(ckpt["scheduler_state"])

        self.model.load_state_dict(ckpt["model_state"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError as e:
            print(f"[WARN] Skipping optimizer state load: {e}")
            print("       This usually means the model/optimizer structure changed since the checkpoint was saved.")
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "feat_adapter_state" in ckpt and self.feat_adapter is not None:
            self.feat_adapter.load_state_dict(ckpt["feat_adapter_state"])
        
        # Load history and restore best validation metrics if history file exists
        history_path = self.log_dir / f"{self.exp_name}_history.json"
        if history_path.exists():
            try:
                import json
                with open(history_path, "r") as f:
                    hist = json.load(f)
                # Filter history to only include epochs up to the checkpoint epoch
                self.history = [h for h in hist if h.get("epoch", 0) <= ckpt["epoch"]]
                
                # Restore best val metrics from the history
                val_f1s = [h["val_combined_f1"] for h in self.history if "val_combined_f1" in h]
                if val_f1s:
                    self.best_val_f1 = max(val_f1s)
                    for h in self.history:
                        if h.get("val_combined_f1") == self.best_val_f1:
                            self.best_epoch = h["epoch"]
                            break
                    print(f"  [RESUME] Restored history ({len(self.history)} epochs). Best val F1 so far: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
            except Exception as e:
                print(f"[WARN] Failed to load history from {history_path}: {e}")
        
        # Fallback if history couldn't be loaded or doesn't have metrics
        if not self.best_val_f1 and ckpt.get("metrics"):
            self.best_val_f1 = ckpt["metrics"].get("combined_f1", 0.0)
            self.best_epoch = ckpt.get("epoch", 0)
            print(f"  [RESUME] Fallback to checkpoint metrics. Best val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
            
        return ckpt["epoch"], ckpt["metrics"]

    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader,
        val_loader,
        start_epoch: int = 1,
    ):
        epochs = self.config.get("epochs", 50)
        # Allow disabling early stopping via config: `early_stopping: false`
        early_stopping = bool(self.config.get("early_stopping", True))
        patience = int(self.config.get("early_stopping_patience", 10)) if early_stopping else None
        patience_counter = 0

        patience_display = patience if early_stopping else 'disabled'
        print(f"\n{'='*60}")
        print(f"Training: {self.exp_name}")
        print(f"Device: {self.device} | Epochs: {epochs} | Patience: {patience_display} | AMP: {self.use_amp}")
        if self.use_phased_training:
            print(f"Phased Training: Phase 1 (cls-only) epochs 1-{self.phase1_epochs}, Phase 2 (joint) epochs {self.phase1_epochs+1}-{epochs}")
        print(f"{'='*60}")

        # Set initial training phase
        if self.use_phased_training and hasattr(self.model, 'set_training_phase'):
            if start_epoch <= self.phase1_epochs:
                self.model.set_training_phase('classification_only')
            else:
                self.model.set_training_phase('joint')

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()

            # ── Phase transition check ────────────────────────────
            if self.use_phased_training and hasattr(self.model, 'set_training_phase'):
                if epoch == self.phase1_epochs + 1:
                    self.model.set_training_phase('joint')
                    # Rebuild optimizer param groups to include newly unfrozen detection params
                    base_lr = float(self.config.get("lr", 1e-3))
                    det_lr_mult = float(self.config.get("det_lr_mult", 0.3))
                    backbone_lr_mult = float(self.config.get("backbone_lr_mult", 0.1))
                    print(f"  [PHASE 2] Rebuilding optimizer with detection LR={base_lr*det_lr_mult:.2e}")

            # ── Train ────────────────────────────────────────────
            train_metrics = self.train_epoch(train_loader, epoch)
            self.scheduler.step()

            # ── Validate ─────────────────────────────────────────
            val_metrics = self.eval_epoch(val_loader, epoch, split="val")
            elapsed = time.time() - t0

            # ── Logging ──────────────────────────────────────────
            combined_f1 = val_metrics["combined_f1"]
            is_best = self.best_epoch == 0 or combined_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = combined_f1
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            epoch_log = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_s": elapsed,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_log)
            self._save_history()

            # ── Print ─────────────────────────────────────────────
            print(
                f"Ep {epoch:03d}/{epochs} | "
                f"Loss {train_metrics['loss']:.4f} -> {val_metrics['loss']:.4f} | "
                f"D-F1 {val_metrics['disaster_f1']:.4f} | "
                f"V-F1 {val_metrics['victim_f1']:.4f} | "
                f"{'*BEST ' if is_best else ''}"
                f"LR {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            # ── Save ──────────────────────────────────────────────
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # ── Early Stopping Check ──────────────────────────────
            if early_stopping and (patience is not None) and (patience_counter >= patience):
                print(f"\n[EARLY STOPPING] No improvement in validation Combined F1 for {patience} epochs. Stopping early.")
                break

        # ── Save full history ─────────────────────────────────────
        history_path = self._save_history()

        print(f"\n{'='*60}")
        print(f"Training complete: {self.exp_name}")
        print(f"Best val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print(f"History saved: {history_path}")
        print(f"{'='*60}\n")

        return self.history

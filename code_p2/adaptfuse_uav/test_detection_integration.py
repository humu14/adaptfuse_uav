import torch, sys
sys.path.insert(0, '.')

# 1. Test loss imports
from training.losses import CombinedLoss, compute_map50, yolo_to_xyxy, det_loss_from_gt
print('[OK] losses imports')

# 2. Test dataset bbox loading
from datasets.multimodal_dataset import load_yolo_bboxes, bbox_collate_fn, MAX_BOXES
boxes, n = load_yolo_bboxes('datasets/raw/SARD/test/labels/gss1006_jpg.rf.32aa7f99ebd9eed1e6777ff5a3b64517.txt')
print('[OK] load_yolo_bboxes: n=%d, shape=%s, first box=%s' % (n, str(boxes.shape), str(boxes[0])))

# 3. Test CombinedLoss with detection
B = 2
loss_fn = CombinedLoss(det_weight=0.5)
d_out = torch.randn(B, 4)
v_out = torch.randn(B, 2)
d_gt  = torch.tensor([0, 1])
v_gt  = torch.tensor([0, 1])

det_pyr  = [torch.randn(B, 7, 32, 32), torch.randn(B, 7, 16, 16)]
gt_boxes = torch.zeros(B, MAX_BOXES, 5)
gt_boxes[0, 0] = torch.tensor([0, 0.5, 0.5, 0.2, 0.3])  # one box in image 0
n_boxes  = torch.tensor([1, 0])
has_bbox = torch.tensor([1.0, 0.0])

losses = loss_fn(d_out, v_out, d_gt, v_gt,
                 det_pyramids=det_pyr,
                 gt_boxes=gt_boxes, n_boxes=n_boxes, has_bbox=has_bbox)
print('[OK] CombinedLoss with detection: total=%.4f  det=%.4f' % (losses['total'].item(), losses['det'].item()))

# 4. Test compute_map50
pred_batch = [torch.rand(10, 5), torch.rand(5, 5)]
gt_batch   = [torch.rand(3, 4),  torch.zeros(0, 4)]
map50 = compute_map50(pred_batch, gt_batch)
print('[OK] compute_map50=%.4f' % map50)

print('\nALL INTEGRATION TESTS PASSED')

"""YOLO-style single-stage detector completed in the UAM object detection lab.

Boxes use (x_min, y_min, x_max, y_max) in activation-map coordinates.
Public names retain the coursework API so notebook sanity checks still apply.
See docs/ATTRIBUTION.md for the teaching material credits.
"""

import torch
import torchvision
from torch import nn

from a5_helper import (
    FeatureExtractor,
    GenerateGrid,
    ObjectClassification,
    ReferenceOnActivatedAnchors,
)


def GenerateAnchor(anc, grid):
    """
    Anchor generator.

    Inputs:
    - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
      each point in the grid. anc[a] = (w, h) gives the width and height of the
      a'th anchor shape.
    - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
      center of each feature from the backbone feature map. This is the tensor
      returned from GenerateGrid.

    Outputs:
    - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
      anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
      centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
      boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
      and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
      corners of the box.
    """
    # Broadcast the same anchor shapes over every image and grid position.
    half_sizes = anc.to(device=grid.device, dtype=grid.dtype).view(1, -1, 1, 1, 2) / 2
    centers = grid.unsqueeze(1)
    return torch.cat((centers - half_sizes, centers + half_sizes), dim=-1)


def GenerateProposal(anchors, offsets, method="YOLO"):
    """
    Proposal generator.

    Inputs:
    - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
      by the coordinates of their top-left and bottom-right corners.
    - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
      convert anchor boxes into region proposals. The transformation
      offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
      anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
      (-0.5, 0.5).
    - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'

    Outputs:
    - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
      coordinates of their top-left and bottom-right corners. Applying the
      transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
      proposal proposals[b, a, h, w].

    """
    if method not in ("YOLO", "FasterRCNN"):
        raise ValueError("method must be 'YOLO' or 'FasterRCNN'")
    centers = (anchors[..., :2] + anchors[..., 2:]) / 2
    sizes = anchors[..., 2:] - anchors[..., :2]
    # YOLO translates in grid units; Faster R-CNN scales translations by anchor size.
    translations = offsets[..., :2]
    if method == "FasterRCNN":
        translations = translations * sizes
    new_centers = centers + translations
    new_sizes = sizes * torch.exp(offsets[..., 2:])
    return torch.cat((new_centers - new_sizes / 2, new_centers + new_sizes / 2), dim=-1)


def IoU(proposals, bboxes):
    """
    Compute intersection over union between sets of bounding boxes.

    Inputs:
    - proposals: Proposals of shape (B, A, H', W', 4)
    - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
      Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
      If image i has fewer than N boxes, then bboxes[i] will be padded with extra
      rows of -1.

    Outputs:
    - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
      the IoU between one element of proposals[b] and bboxes[b, n].

    Padded and degenerate boxes have zero area and yield zero IoU.
    """
    batch_size = proposals.shape[0]
    proposals = proposals.reshape(batch_size, -1, 4).unsqueeze(2)
    boxes = bboxes[:, None, :, :4]
    # Broadcasting avoids materializing a copy for every proposal/ground-truth pair.
    intersection_sizes = (
        torch.minimum(proposals[..., 2:], boxes[..., 2:])
        - torch.maximum(proposals[..., :2], boxes[..., :2])
    ).clamp(min=0)
    intersection = intersection_sizes.prod(dim=-1)
    proposal_area = (proposals[..., 2:] - proposals[..., :2]).clamp(min=0).prod(dim=-1)
    box_area = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0).prod(dim=-1)
    union = proposal_area + box_area - intersection
    # Degenerate boxes and -1 padding must produce zero rather than NaN.
    return intersection / union.clamp(min=torch.finfo(proposals.dtype).eps)


class PredictionNetwork(nn.Module):
    """Predict objectness and box offsets per anchor, with class logits per cell."""

    def __init__(
        self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3
    ):
        super().__init__()

        assert num_classes != 0 and num_anchors != 0
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.Dropout(p=drop_ratio),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, 5 * num_anchors + num_classes, kernel_size=1),
        )

    def _extract_anchor_data(self, anchor_data, anchor_idx):
        """
        Inputs:
        - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
          D for each of A anchors at each point in an H x W grid.
        - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

        Returns:
        - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
          of the anchors specified by anchor_idx.
        """
        B, A, D, H, W = anchor_data.shape
        anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
        extracted_anchors = anchor_data[anchor_idx]
        return extracted_anchors

    def _extract_class_scores(self, all_scores, anchor_idx):
        """
        Inputs:
        - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
          C classes at each point in an H x W grid.
        - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
          which to extract classification scores

        Returns:
        - extracted_scores: Tensor of shape (M, C) giving the classification scores
          for each of the anchors specified by anchor_idx.
        """
        B, C, H, W = all_scores.shape
        A = self.num_anchors
        all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
        all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
        all_scores = all_scores.reshape(B * A * H * W, C)
        extracted_scores = all_scores[anchor_idx]
        return extracted_scores

    def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
        """
        Run the forward pass of the network to predict outputs given features
        from the backbone network.

        Inputs:
        - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
          by the backbone network.
        - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
          marked as positive. These are only given during training; at test-time
          this should be None.
        - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
          marked as negative. These are only given at training; at test-time this
          should be None.

        The outputs from this method are different during training and inference.

        During training, pos_anchor_idx and neg_anchor_idx are given and identify
        which anchors should be positive and negative, and this forward pass needs
        to extract only the predictions for the positive and negative anchors.

        During inference, only features are provided and this method needs to return
        predictions for all anchors.

        Outputs (During training):
        - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
          scores for positive anchors and negative anchors (in that order).
        - offsets: Tensor of shape (M, 4) giving predicted transformation for
          positive anchors.
        - class_scores: Tensor of shape (M, C) giving classification scores for
          positive anchors.

        Outputs (During inference):
        - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
          scores for all anchors.
        - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
          for all anchors.
        - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
          each spatial position.
        """
        A = self.num_anchors
        B, _, H, W = features.shape

        # use self.net to predict outputs for all anchors
        out = self.net(features)  # (B, 5A+C, 7, 7)

        # retrieve anchor predictions and reshape them
        anchor_data = out[:, : 5 * A, :, :].view(B, A, 5, H, W)

        # retrieve class scores
        class_scores = out[:, 5 * A :, :, :]  # (B, C, 7, 7)

        # retrieve objectness scores from anchor_data. Apply a sigmoid to those scores
        conf_scores = torch.sigmoid(anchor_data[:, :, 0:1, :, :])  # (B, A, 1, H, W)

        # retrieve anchor offsets from anchor_data.
        # Build a fresh tensor to preserve autograd's saved values for backpropagation.
        offsets = torch.cat(
            (
                torch.sigmoid(anchor_data[:, :, 1:3, :, :]) - 0.5,
                anchor_data[:, :, 3:5, :, :],
            ),
            dim=2,
        )

        if pos_anchor_idx is None:  # inference. Use all anchors
            conf_scores = conf_scores.squeeze(2)

        else:  # training. Use positive and negative anchors
            # Use the self._extract_anchor_data and self._extract_class_scores functions
            # to obtain bbox offsets and class scores for positive anchors. Extract objectness scores
            # for positive and negative samples (pos, neg)
            offsets = self._extract_anchor_data(offsets, pos_anchor_idx)
            class_scores = self._extract_class_scores(class_scores, pos_anchor_idx)
            pos = self._extract_anchor_data(conf_scores, pos_anchor_idx)  # (M, 1)
            neg = self._extract_anchor_data(conf_scores, neg_anchor_idx)  # (M, 1)

            conf_scores = torch.cat((pos, neg), dim=0)  # (2*M, 1)

        return conf_scores, offsets, class_scores


def ConfScoreRegression(conf_scores, GT_conf_scores):
    """
    Use sum-squared error as in YOLO

    Inputs:
    - conf_scores: Predicted confidence scores
    - GT_conf_scores: GT confidence scores

    Outputs:
    - conf_score_loss
    """
    # the target conf_scores for negative samples are zeros
    GT_conf_scores = torch.cat(
        (torch.ones_like(GT_conf_scores), torch.zeros_like(GT_conf_scores)), dim=0
    ).view(-1, 1)

    conf_score_loss = (
        torch.sum((conf_scores - GT_conf_scores) ** 2) * 1.0 / GT_conf_scores.shape[0]
    )
    return conf_score_loss


def BboxRegression(offsets, GT_offsets):
    """
    Use sum-squared error as in YOLO
    For both tx,ty and tw,th

    Inputs:
    - offsets: Predicted box offsets
    - GT_offsets: GT box offsets

    Outputs:
    - bbox_reg_loss
    """

    # compute sum of squared errors of the offsets
    bbox_reg_loss = torch.sum((offsets - GT_offsets) ** 2)

    # mini-batch average
    bbox_reg_loss = bbox_reg_loss * 1.0 / GT_offsets.shape[0]
    return bbox_reg_loss


def FocalLoss(logits, target, focusing_param=2, balance_param=0.25):
    """Return unreduced multiclass focal loss for each of N positive anchors.

    logits has shape (N, C); target contains N integer class indices.
    focusing_param is gamma and balance_param is a scalar alpha weight.
    Gamma zero and alpha one recover ordinary per-anchor cross entropy.
    """

    ce = nn.functional.cross_entropy(logits, target, reduction="none")  # shape [N]
    pt = torch.exp(-ce)

    focal = balance_param * (1 - pt) ** focusing_param * ce
    return focal


class SingleStageDetector(nn.Module):
    """MobileNetV2 backbone and anchor head with the lab's cross-entropy baseline.

    Set pretrained=False for offline checks. Inference returns boxes in activation
    coordinates; use a5_helper.coord_trans to map them back to the original image.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Move anchors with the model without adding a key to existing lab checkpoints.
        self.register_buffer(
            "anchor_list",
            torch.tensor(
                [
                    [1.0, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                    [5, 5],
                    [2, 3],
                    [3, 2],
                    [3, 5],
                    [5, 3],
                ]
            ),
            persistent=False,
        )
        self.feat_extractor = FeatureExtractor(pretrained=pretrained)
        self.num_classes = 20
        self.pred_network = PredictionNetwork(
            1280, num_anchors=self.anchor_list.shape[0], num_classes=self.num_classes
        )

    def forward(self, images, bboxes):
        """
        Training-time forward pass for the single-stage detector.

        Inputs:
        - images: Input images, of shape (B, 3, 224, 224)
        - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

        Outputs:
        - total_loss: Torch scalar giving the total loss for the batch.
        """
        # Give each component of the multi-task objective equal weight.
        w_conf = 1  # for conf_scores
        w_reg = 1  # for offsets
        w_cls = 1  # for class_prob

        B = images.shape[0]

        # i) backbone CNN feature extraction
        features = self.feat_extractor(images)  # (B, 1280, 7, 7)

        # ii) create the grid using GenerateGrid (implemented in a5_helper.py)
        grid_list = GenerateGrid(
            B,
            w_amap=features.shape[3],
            h_amap=features.shape[2],
            device=features.device,
            dtype=features.dtype,
        )

        # generate the anchors using GenerateAnchor
        anc_list = GenerateAnchor(self.anchor_list, grid_list)

        # iii) compute iou:
        iou_mat = IoU(anc_list, bboxes)  # (B, A*H'*W', N)

        # match the ground-truth boxes against the anchors to determine the classification labels for the anchors
        (
            activated_anc_ind,
            negative_anc_ind,
            GT_conf_scores,
            GT_offsets,
            GT_class,
            activated_anc_coord,
            negative_anc_coord,
        ) = ReferenceOnActivatedAnchors(
            anc_list, bboxes, grid_list, iou_mat, neg_thresh=0.2, method="YOLO"
        )

        # Train objectness on positive/negative anchors; regress and classify positives.
        conf_scores, offsets, class_scores = self.pred_network(
            features, activated_anc_ind, negative_anc_ind
        )

        # Compute the objectness score and bbox regression losses. Use GT objectness scores and bbox offsets computed before
        conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
        reg_loss = BboxRegression(offsets, GT_offsets)

        # classification loss
        anc_per_img = self.anchor_list.shape[0] * features.shape[2] * features.shape[3]
        cls_loss = ObjectClassification(
            class_scores, GT_class, B, anc_per_img, activated_anc_ind
        )

        # multi-task loss. Compute a weighted multi-task loss weighted by the w_conf, w_reg and w_cls weights
        total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

        return total_loss

    def inference(self, images, thresh=0.5, nms_thresh=0.7):
        """
        Inference-time forward pass for the single stage detector.

        Inputs:
        - images: Input images; call model.eval() first to disable dropout
        - thresh: Threshold value on confidence scores
        - nms_thresh: Threshold value on NMS

        Outputs:
        - final_proposals: Retained proposals after confidence score thresholding and NMS,
                          a list of B (*x4) tensors
        - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
        - final_class: Corresponding class predictions, a list of B  (*x1) tensors
        """
        final_proposals, final_conf_scores, final_class = [], [], []
        with torch.no_grad():
            B = images.shape[0]
            A = len(self.anchor_list)

            # feature extraction
            features = self.feat_extractor(images)  # (B, 1280, 7, 7)
            _, _, H, W = features.shape

            # Grid and Anchor generation
            grid_list = GenerateGrid(
                B,
                w_amap=features.shape[3],
                h_amap=features.shape[2],
                device=features.device,
                dtype=features.dtype,
            )
            anc_list = GenerateAnchor(self.anchor_list, grid_list)

            # predict scores (inference mode - no pos/neg anchor indices)
            conf_scores, offsets, class_scores = self.pred_network(features)
            C = class_scores.shape[1]

            # Permute offsets for proposal generation
            offsets = offsets.permute(0, 1, 3, 4, 2).contiguous()  # B, A, 7, 7, 4
            # generate proposals
            proposals = GenerateProposal(anc_list, offsets, method="YOLO")

            for b in range(B):  # loop over samples in the batch
                # re-shape to find maximum class-score per grid cell
                expand_class = (
                    class_scores[b].permute(1, 2, 0).unsqueeze(0).expand(A, H, W, C)
                )  # A * 7 * 7 * C
                # find the predicted class per grid cell (and anchor)
                picked_class = torch.argmax(expand_class, dim=3)  # A * 7 * 7

                # keep objectness scores over threshold only
                thresh_index = torch.nonzero((conf_scores[b] > thresh), as_tuple=True)
                conf_scores_masked = conf_scores[b][thresh_index].view(-1)
                class_score_masked = picked_class[thresh_index].view(-1)
                proposal_masked = proposals[b][thresh_index].view(-1, 4)

                # use torchvision.ops.nms to find the indices of proposals kept after NMS
                proposals_left = torchvision.ops.nms(
                    proposal_masked, conf_scores_masked, nms_thresh
                )

                final_proposals.append(proposal_masked[proposals_left])  # (*x4)
                final_conf_scores.append(
                    conf_scores_masked[proposals_left].unsqueeze(1)
                )  # (*x1)
                final_class.append(
                    class_score_masked[proposals_left].unsqueeze(1)
                )  # (*x1)

        return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
    """Return indices kept by NMS, optionally limited to the top scoring boxes."""
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return keep if topk is None else keep[:topk]

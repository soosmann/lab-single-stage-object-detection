"""VOC data loading, coordinate transforms, and detector training utilities.

The public function names follow the original teaching material. Images are
normalized for ImageNet; annotation boxes retain original pixel coordinates
until coord_trans explicitly converts them to the activation grid.
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import eecs598


def hello_helper():
    """Print a small import check used by the lab notebooks."""
    print("Hello from a5_helper.py!")


# --- Pascal Classes ---
class_to_idx = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}
idx_to_class = {i: c for c, i in class_to_idx.items()}


def get_pascal_voc2007_data(image_root, split="train"):
    """Load a VOC 2007 split, downloading trainval only when not extracted.

    image_root contains VOCdevkit/VOC2007, not the JPEGImages directory itself.
    A previously downloaded archive at image_root is reused by torchvision.
    """
    image_root = Path(image_root).expanduser()
    image_root.mkdir(parents=True, exist_ok=True)
    download = not (image_root / "VOCdevkit" / "VOC2007").is_dir()
    return datasets.VOCDetection(
        str(image_root), year="2007", image_set=split, download=download
    )


def pascal_voc2007_loader(dataset, batch_size, num_workers=0):
    """
    Data loader for Pascal VOC 2007.
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    # turn off shuffle so we can index the original image
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=voc_collate_fn,
    )
    return train_loader


def voc_collate_fn(batch_lst, reshape_size=224):
    """Resize images and pad (x_min, y_min, x_max, y_max, class) rows with -1.

    Return images, boxes, original widths/heights, and image filenames.
    VOC annotations with one object may store it as a dict rather than a list.
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize((reshape_size, reshape_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    batch_size = len(batch_lst)

    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)

    objects = [sample[1]["annotation"]["object"] for sample in batch_lst]
    objects = [[obj] if isinstance(obj, dict) else obj for obj in objects]
    max_num_box = max(len(obj) for obj in objects)

    box_batch = torch.Tensor(batch_size, max_num_box, 5).fill_(-1.0)
    w_list = []
    h_list = []
    img_id_list = []

    for i in range(batch_size):
        img, ann = batch_lst[i]
        w_list.append(img.size[0])  # image width
        h_list.append(img.size[1])  # image height
        img_id_list.append(ann["annotation"]["filename"])
        img_batch[i] = preprocess(img)
        all_bbox = objects[i]
        for bbox_idx, one_bbox in enumerate(all_bbox):
            bbox = one_bbox["bndbox"]
            obj_cls = one_bbox["name"]
            box_batch[i][bbox_idx] = torch.Tensor(
                [
                    float(bbox["xmin"]),
                    float(bbox["ymin"]),
                    float(bbox["xmax"]),
                    float(bbox["ymax"]),
                    class_to_idx[obj_cls],
                ]
            )

    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)

    return img_batch, box_batch, w_batch, h_batch, img_id_list


def coord_trans(bbox, w_pixel, h_pixel, w_amap=7, h_amap=7, mode="a2p"):
    """
    Coordinate transformation function. It converts the box coordinate from
    the image coordinate system to the activation map coordinate system and vice versa.
    In our case, the input image will have a few hundred of pixels in
    width/height while the activation map is of size 7x7.

    Input:
    - bbox: Could be either bbox, anchor, or proposal, of shape Bx*x4
    - w_pixel: Number of pixels in the width side of the original image, of shape B
    - h_pixel: Number of pixels in the height side of the original image, of shape B
    - w_amap: Number of pixels in the width side of the activation map, scalar
    - h_amap: Number of pixels in the height side of the activation map, scalar
    - mode: Whether transfer from the original image to activation map ('p2a') or
            the opposite ('a2p')

    Output:
    - resized_bbox: Resized box coordinates, of the same shape as the input bbox
    """

    assert mode in ("p2a", "a2p"), "invalid coordinate transformation mode!"
    assert (
        bbox.shape[-1] >= 4
    ), "the transformation is applied to the first 4 values of dim -1"

    if bbox.shape[0] == 0:  # corner cases
        return bbox

    resized_bbox = bbox.clone()
    # could still work if the first dim of bbox is not batch size
    # in that case, w_pixel and h_pixel will be scalars
    resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
    invalid_bbox_mask = resized_bbox == -1  # indicating invalid bbox

    if mode == "p2a":
        # pixel to activation
        width_ratio = w_pixel * 1.0 / w_amap
        height_ratio = h_pixel * 1.0 / h_amap
        resized_bbox[:, :, [0, 2]] /= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] /= height_ratio.view(-1, 1, 1)
    else:
        # activation to pixel
        width_ratio = w_pixel * 1.0 / w_amap
        height_ratio = h_pixel * 1.0 / h_amap
        resized_bbox[:, :, [0, 2]] *= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] *= height_ratio.view(-1, 1, 1)

    resized_bbox.masked_fill_(invalid_bbox_mask, -1)
    resized_bbox.resize_as_(bbox)
    return resized_bbox


class FeatureExtractor(torch.nn.Module):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, reshape_size=224, pooling=False, verbose=False, pretrained=True):
        """Build MobileNetV2 features; pretrained=False avoids a weights download."""
        super().__init__()

        if hasattr(models, "MobileNet_V2_Weights"):
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.mobilenet = models.mobilenet_v2(weights=weights)
        else:  # Keep compatibility with the torchvision version used in the lab.
            self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.mobilenet = torch.nn.Sequential(
            *list(self.mobilenet.children())[:-1]
        )  # Remove the last classifier

        # average pooling
        if pooling:
            self.mobilenet.add_module(
                "LastAvgPool", nn.AvgPool2d(math.ceil(reshape_size / 32.0))
            )  # input: N x 1280 x 7 x 7

        for i in self.mobilenet.named_parameters():
            i[1].requires_grad = True  # fine-tune all

        if verbose:
            from torchsummary import summary

            if torch.cuda.device_count() == 0:
                summary(self.mobilenet, (3, reshape_size, reshape_size))
            else:
                summary(self.mobilenet.cuda(), (3, reshape_size, reshape_size))

    def forward(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape Nx3x224x224

        Outputs:
        - feat: Image feature, of shape Nx1280 (pooled) or Nx1280x7x7
        """
        num_img = img.shape[0]

        img_prepro = img

        feat = []
        process_batch = 500
        for b in range(math.ceil(num_img / process_batch)):
            feat.append(
                self.mobilenet(img_prepro[b * process_batch : (b + 1) * process_batch])
                .squeeze(-1)
                .squeeze(-1)
            )  # forward and squeeze
        feat = torch.cat(feat)

        if verbose:
            print("Output feature shape: ", feat.shape)

        return feat


def GenerateGrid(batch_size, w_amap=7, h_amap=7, dtype=torch.float32, device="cpu"):
    """
    Return a grid cell given a batch size (center coordinates).

    Inputs:
    - batch_size, B
    - w_amap: or W', width of the activation map (number of grids in the horizontal dimension)
    - h_amap: or H', height of the activation map (number of grids in the vertical dimension)
    - W' and H' are always 7 in our case while w and h might vary.

    Outputs:
    grid: A float32 tensor of shape (B, H', W', 2) giving the (x, y) coordinates
          of the centers of each feature for a feature map of shape (B, D, H', W')
    """
    w_range = torch.arange(0, w_amap, dtype=dtype, device=device) + 0.5
    h_range = torch.arange(0, h_amap, dtype=dtype, device=device) + 0.5

    w_grid_idx = w_range.unsqueeze(0).repeat(h_amap, 1)
    h_grid_idx = h_range.unsqueeze(1).repeat(1, w_amap)
    grid = torch.stack([w_grid_idx, h_grid_idx], dim=-1)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return grid


def ReferenceOnActivatedAnchors(
    anchors, bboxes, grid, iou_mat, pos_thresh=0.7, neg_thresh=0.3, method="FasterRCNN"
):
    """
    Determine the activated (positive) and negative anchors for model training.

    For YOLO - A grid cell is responsible for predicting a GT box if the center of
    the box falls into that cell.
    Implementation details: First compute manhattan distance between grid cell centers
    (BxH’xW’) and GT box centers (BxN). This gives us a matrix of shape Bx(H'xW')xN and
    perform torch.min(dim=1)[1] on it gives us the indexes indicating activated grids
    responsible for GT boxes (convert to x and y). Among all the anchors associated with
    the activate grids, the anchor with the largest IoU with the GT box is responsible to
    predict (regress to) the GT box.
    Note: One anchor might match multiple GT boxes.

    For Faster R-CNN - Positive anchors are defined Any of the two
    (i) the anchor/anchors with the highest IoU overlap with a GT box, or
    (ii) an anchor that has an IoU overlap higher than 0.7 with any GT box.
    Note: One anchor can match at most one GT box (the one with the largest IoU overlapping).

    For both - We assign a negative label to a anchor if its IoU ratio is lower than
    a threshold value for all GT boxes. Anchors that are neither positive nor negative
    do not contribute to the training objective.

    Main steps include:
    i) Decide activated and negative anchors based on the IoU matrix.
    ii) Compute GT confidence score/offsets/object class on the positive proposals.
    iii) Compute GT confidence score on the negative proposals.

    Inputs:
    - anchors: Anchor boxes, of shape BxAxH’xW’x4
    - bboxes: GT boxes of shape BxNx5, where N is the number of PADDED GT boxes,
              5 indicates (x_{lr}^{gt}, y_{lr}^{gt}, x_{rb}^{gt}, y_{rb}^{gt}) and class index
    - grid (float): A cell grid of shape BxH'xW'x2 where 2 indicate the (x, y) coord
    - iou_mat: IoU matrix of shape Bx(AxH’xW’)xN
    - pos_thresh: Positive threshold value
    - neg_thresh: Negative threshold value
    - method: Switch between 'YOLO' mode and 'FasterRCNN' mode

    Outputs:
    - activated_anc_ind: Index on activated anchors, of shape M, where M indicates the
                         number of activated anchors
    - negative_anc_ind: Index on negative anchors, of shape M
    - GT_conf_scores: GT IoU confidence scores on activated anchors, of shape M
    - GT_offsets: GT offsets on activated anchors, of shape Mx4. They are denoted as
                (t_x, t_y, t_w, t_h) in the formulation earlier.
    - GT_class: GT class category on activated anchors, essentially indexed from bboxes[:, :, 4],
                of shape M
    - activated_anc_coord: Coordinates on activated anchors (mainly for visualization purposes)
    - negative_anc_coord: Coordinates on negative anchors (mainly for visualization purposes)
    """

    assert method in ["FasterRCNN", "YOLO"]

    B, A, h_amap, w_amap, _ = anchors.shape
    N = bboxes.shape[1]

    # activated/positive anchors
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    if method == "FasterRCNN":
        max_iou_per_box = iou_mat.max(dim=1, keepdim=True)[0]
        activated_anc_mask = (iou_mat == max_iou_per_box) & (max_iou_per_box > 0)
        activated_anc_mask |= (
            iou_mat > pos_thresh
        )  # using the pos_thresh condition as well
        # if an anchor matches multiple GT boxes, choose the box with the largest iou
        activated_anc_mask = activated_anc_mask.max(dim=-1)[0]  # Bx(AxH’xW’)
        activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)

        # GT conf scores
        GT_conf_scores = max_iou_per_anc[activated_anc_mask]  # M

        # GT class
        box_cls = bboxes[:, :, 4].view(B, 1, N).expand((B, A * h_amap * w_amap, N))
        GT_class = torch.gather(box_cls, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(
            -1
        )  # M
        GT_class = GT_class[activated_anc_mask].long()

        bboxes_expand = (
            bboxes[:, :, :4].view(B, 1, N, 4).expand((B, A * h_amap * w_amap, N, 4))
        )
        bboxes = torch.gather(
            bboxes_expand,
            -2,
            max_iou_per_anc_ind.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(B, A * h_amap * w_amap, 1, 4),
        ).view(-1, 4)
        bboxes = bboxes[activated_anc_ind]
    else:
        bbox_mask = bboxes[:, :, 0] != -1  # BxN, indicate invalid boxes
        bbox_centers = (bboxes[:, :, 2:4] - bboxes[:, :, :2]) / 2.0 + bboxes[
            :, :, :2
        ]  # BxNx2

        mah_dist = torch.abs(
            grid.view(B, -1, 2).unsqueeze(2) - bbox_centers.unsqueeze(1)
        ).sum(
            dim=-1
        )  # Bx(H'xW')xN
        min_mah_dist = mah_dist.min(dim=1, keepdim=True)[0]  # Bx1xN
        grid_mask = (mah_dist == min_mah_dist).unsqueeze(1)  # Bx1x(H'xW')xN

        reshaped_iou_mat = iou_mat.view(B, A, -1, N)
        anc_with_largest_iou = reshaped_iou_mat.max(dim=1, keepdim=True)[
            0
        ]  # Bx1x(H’xW’)xN
        anc_mask = anc_with_largest_iou == reshaped_iou_mat  # BxAx(H’xW’)xN
        activated_anc_mask = (grid_mask & anc_mask).view(B, -1, N)
        activated_anc_mask &= bbox_mask.unsqueeze(1)

        # one anchor could match multiple GT boxes
        activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)
        GT_conf_scores = iou_mat.view(-1)[activated_anc_ind]
        bboxes = (
            bboxes.view(B, 1, N, 5)
            .repeat(1, A * h_amap * w_amap, 1, 1)
            .view(-1, 5)[activated_anc_ind]
        )
        GT_class = bboxes[:, 4].long()
        bboxes = bboxes[:, :4]
        activated_anc_ind = (
            activated_anc_ind / float(activated_anc_mask.shape[-1])
        ).long()

    activated_anc_coord = anchors.view(-1, 4)[activated_anc_ind]

    # GT offsets
    # bbox and anchor coordinates are x_tl, y_tl, x_br, y_br
    # offsets are t_x, t_y, t_w, t_h
    wh_offsets = torch.log(
        (bboxes[:, 2:4] - bboxes[:, :2])
        / (activated_anc_coord[:, 2:4] - activated_anc_coord[:, :2])
    )

    xy_offsets = (
        bboxes[:, :2]
        + bboxes[:, 2:4]
        - activated_anc_coord[:, :2]
        - activated_anc_coord[:, 2:4]
    ) / 2.0

    if method == "FasterRCNN":
        xy_offsets /= activated_anc_coord[:, 2:4] - activated_anc_coord[:, :2]
    else:
        assert (
            torch.max(torch.abs(xy_offsets)) <= 0.5
        ), "x and y offsets should be between -0.5 and 0.5! Got {}".format(
            torch.max(torch.abs(xy_offsets))
        )

    GT_offsets = torch.cat((xy_offsets, wh_offsets), dim=-1)

    # negative anchors
    negative_anc_mask = max_iou_per_anc < neg_thresh  # Bx(AxH’xW’)
    negative_anc_ind = torch.nonzero(negative_anc_mask.view(-1)).squeeze(-1)
    negative_anc_ind = negative_anc_ind[
        torch.randint(
            0,
            negative_anc_ind.shape[0],
            (activated_anc_ind.shape[0],),
            device=negative_anc_ind.device,
        )
    ]
    negative_anc_coord = anchors.view(-1, 4)[negative_anc_ind.view(-1)]

    # activated_anc_coord and negative_anc_coord are mainly for visualization purposes
    return (
        activated_anc_ind,
        negative_anc_ind,
        GT_conf_scores,
        GT_offsets,
        GT_class,
        activated_anc_coord,
        negative_anc_coord,
    )


def ObjectClassification(
    class_prob, GT_class, batch_size, anc_per_img, activated_anc_ind
):
    """
    Average positive-anchor cross entropy per image, then across the batch.

    Inputs:
    - class_prob: Predicted class logits
    - GT_class: GT box class label

    Outputs:
    - object_cls_loss
    """
    # average within sample and then average across batch
    # such that the class pred would not bias towards dense popular objects like `person`
    all_loss = torch.nn.functional.cross_entropy(class_prob, GT_class, reduction="none")
    object_cls_loss = 0
    for idx in range(batch_size):
        anc_ind_in_img = (activated_anc_ind >= idx * anc_per_img) & (
            activated_anc_ind < (idx + 1) * anc_per_img
        )
        object_cls_loss += all_loss[anc_ind_in_img].sum() / torch.sum(
            anc_ind_in_img
        ).clamp(min=1)
    object_cls_loss /= batch_size

    return object_cls_loss


def DetectionSolver(
    detector,
    train_loader,
    learning_rate=3e-3,
    lr_decay=1,
    num_epochs=20,
    dtype=torch.float32,
    device="cpu",
    **kwargs,
):
    """
    Run optimization to train the model.
    """

    # ship model to GPU
    detector.to(dtype=dtype, device=device)

    # optimizer setup
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()), learning_rate
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay**epoch)

    # sample minibatch data
    loss_history = []
    epoch_loss_history = []
    detector.train()

    for i in range(num_epochs):
        start_t = time.time()
        epoch_loss = 0.0

        for iter_num, data_batch in enumerate(train_loader):
            images, boxes, w_batch, h_batch, _ = data_batch
            resized_boxes = coord_trans(boxes, w_batch, h_batch, mode="p2a")
            images = images.to(dtype=dtype, device=device)
            resized_boxes = resized_boxes.to(dtype=dtype, device=device)

            loss = detector(images, resized_boxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            epoch_loss += loss.item() * images.shape[0]

            print("(Iter {} / {})".format(iter_num, len(train_loader)))

        end_t = time.time()
        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_loss_history.append(epoch_loss)
        print(
            "(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s".format(
                i + 1, num_epochs, epoch_loss, end_t - start_t
            )
        )

        lr_scheduler.step()

    # plot the training losses
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.subplot(1, 2, 2)
    plt.plot(epoch_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def DetectionEvaluater(detector, loader, dtype=torch.float32, device="cpu", **kwargs):
    """
    Report mean detector loss without gradients (this does not compute mAP).
    """
    # ship model to GPU
    detector.to(dtype=dtype, device=device)

    # sample minibatch data
    detector.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        for iter_num, data_batch in enumerate(loader):
            images, boxes, w_batch, h_batch, _ = data_batch
            resized_boxes = coord_trans(boxes, w_batch, h_batch, mode="p2a")
            images = images.to(dtype=dtype, device=device)
            resized_boxes = resized_boxes.to(dtype=dtype, device=device)

            epoch_loss += detector(images, resized_boxes) * images.shape[0]

    # divide by number of instances
    epoch_loss = epoch_loss / len(loader.dataset)
    print(f"Epoch loss: {epoch_loss:.4f}")
    return epoch_loss


@torch.no_grad()
def DetectionInference(
    detector,
    data_loader,
    dataset,
    idx_to_class,
    thresh=0.8,
    nms_thresh=0.3,
    output_dir=None,
    dtype=torch.float32,
    device="cpu",
):
    """Visualize detections or export them in Cartucho/mAP text format.

    output_dir is the evaluator's input directory; its detection-results and
    ground-truth subdirectories are refreshed on each call. Boxes from the model
    use activation coordinates and are converted to original pixels before export.
    Visualization requires the loader to visit dataset in order (shuffle=False).
    """
    detector.to(dtype=dtype, device=device)
    detector.eval()
    start_t = time.time()

    if output_dir is not None:
        det_dir = Path(output_dir) / "detection-results"
        gt_dir = Path(output_dir) / "ground-truth"
        for directory in (det_dir, gt_dir):
            directory.mkdir(parents=True, exist_ok=True)
            # Drop stale predictions from earlier evaluations on a different subset.
            for previous_result in directory.glob("*.txt"):
                previous_result.unlink()

    sample_offset = 0
    for data_batch in data_loader:
        images, boxes, w_batch, h_batch, img_ids = data_batch
        images = images.to(dtype=dtype, device=device)
        proposals, scores, classes = detector.inference(
            images, thresh=thresh, nms_thresh=nms_thresh
        )
        for idx in range(len(images)):
            valid_boxes = boxes[idx][boxes[idx, :, 0] != -1]
            predictions = torch.cat(
                (proposals[idx], classes[idx].float(), scores[idx]), dim=-1
            ).cpu()
            predictions = coord_trans(predictions, w_batch[idx], h_batch[idx])
            predictions[:, 0:4:2].clamp_(min=0, max=w_batch[idx].item())
            predictions[:, 1:4:2].clamp_(min=0, max=h_batch[idx].item())

            if output_dir is not None:
                filename = Path(img_ids[idx]).with_suffix(".txt").name
                with (
                    (det_dir / filename).open("w") as f_det,
                    (gt_dir / filename).open("w") as f_gt,
                ):
                    for box in valid_boxes:
                        f_gt.write(
                            "{} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                                idx_to_class[int(box[4])], *box[:4].tolist()
                            )
                        )
                    for box in predictions:
                        f_det.write(
                            "{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                                idx_to_class[int(box[4])],
                                box[5].item(),
                                *box[:4].tolist(),
                            )
                        )
            else:
                image, _ = dataset[sample_offset + idx]
                eecs598.vis.detection_visualizer(
                    image, idx_to_class, valid_boxes, predictions
                )
        sample_offset += len(images)

    print("Total inference time: {:.1f}s".format(time.time() - start_t))

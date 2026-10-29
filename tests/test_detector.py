"""CPU regression checks; no VOC data or pretrained-weight downloads required."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from a5_helper import (
    DetectionInference,
    FeatureExtractor,
    GenerateGrid,
    coord_trans,
    get_pascal_voc2007_data,
    voc_collate_fn,
)
from single_stage_detector import (
    FocalLoss,
    GenerateAnchor,
    GenerateProposal,
    IoU,
    PredictionNetwork,
    SingleStageDetector,
    nms,
)


class GeometryTests(unittest.TestCase):
    def test_grid_and_anchor_coordinates(self):
        grid = GenerateGrid(2, w_amap=3, h_amap=2)
        anchors = GenerateAnchor(torch.tensor([[2.0, 4.0]]), grid)
        self.assertEqual(anchors.shape, (2, 1, 2, 3, 4))
        torch.testing.assert_close(
            anchors[0, 0, 0, 0], torch.tensor([-0.5, -1.5, 1.5, 2.5])
        )

    def test_yolo_and_faster_rcnn_translations_differ(self):
        anchors = torch.tensor([0.0, 0.0, 4.0, 2.0]).reshape(1, 1, 1, 1, 4)
        offsets = torch.tensor([0.5, 0.5, 0.0, 0.0]).reshape_as(anchors)
        yolo = GenerateProposal(anchors, offsets)
        rcnn = GenerateProposal(anchors, offsets, method="FasterRCNN")
        torch.testing.assert_close(yolo.flatten(), torch.tensor([0.5, 0.5, 4.5, 2.5]))
        torch.testing.assert_close(rcnn.flatten(), torch.tensor([2.0, 1.0, 6.0, 3.0]))

    def test_iou_includes_padding_and_degenerate_boxes(self):
        proposals = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
        proposals = proposals.reshape(1, 1, 1, 2, 4)
        boxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0, 1.0], [-1.0] * 5]])
        torch.testing.assert_close(
            IoU(proposals, boxes), torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        )

    def test_coordinate_round_trip_preserves_padding_and_class(self):
        boxes = torch.tensor([[[10.0, 20.0, 50.0, 80.0, 3.0], [-1.0] * 5]])
        width, height = torch.tensor([100]), torch.tensor([200])
        activation = coord_trans(boxes, width, height, mode="p2a")
        torch.testing.assert_close(coord_trans(activation, width, height), boxes)

    def test_nms_and_empty_input(self):
        boxes = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 6.0, 6.0]]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        self.assertEqual(nms(boxes, scores).tolist(), [0, 2])
        self.assertEqual(nms(boxes, scores, topk=1).tolist(), [0])
        self.assertEqual(nms(boxes[:0], scores[:0]).shape, (0,))


class NetworkTests(unittest.TestCase):
    def test_single_image_prediction_shapes_and_backward(self):
        model = PredictionNetwork(
            8, hidden_dim=4, num_anchors=2, num_classes=3, drop_ratio=0
        )
        features = torch.randn(1, 8, 3, 3, requires_grad=True)
        confidence, offsets, classes = model(features)
        self.assertEqual(confidence.shape, (1, 2, 3, 3))
        self.assertEqual(offsets.shape, (1, 2, 4, 3, 3))
        self.assertEqual(classes.shape, (1, 3, 3, 3))
        self.assertTrue((offsets[:, :, :2].abs() <= 0.5).all())
        confidence, offsets, classes = model(
            features, torch.tensor([0, 3]), torch.tensor([5, 6])
        )
        self.assertEqual(confidence.shape, (4, 1))
        (confidence.sum() + offsets.sum() + classes.sum()).backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_focal_loss_reduces_to_cross_entropy(self):
        logits = torch.tensor([[1.0, -1.0], [-0.5, 0.5]])
        targets = torch.tensor([0, 1])
        expected = nn.functional.cross_entropy(logits, targets, reduction="none")
        torch.testing.assert_close(FocalLoss(logits, targets, 0, 1), expected)

    def test_detector_cpu_training_and_single_image_inference(self):
        # Replace only the expensive feature extractor, keeping matching, losses,
        # prediction, proposal decoding, and NMS in the integration check.
        backbone = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)), nn.Conv2d(3, 1280, 1))
        with patch("single_stage_detector.FeatureExtractor", return_value=backbone):
            model = SingleStageDetector(pretrained=False)
        images = torch.randn(2, 3, 32, 32)
        boxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0, 0.0]], [[2.0, 2.0, 4.0, 4.0, 1.0]]])
        loss = model(images, boxes)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.pred_network.net[0].weight.grad).all())
        model.eval()
        for threshold in (0.0, 1.0):
            proposals, confidence, classes = model.inference(
                images[:1], thresh=threshold
            )
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].shape[1], 4)
            self.assertEqual(confidence[0].shape, classes[0].shape)
            if threshold == 1.0:
                self.assertEqual(proposals[0].shape[0], 0)
        self.assertNotIn("anchor_list", model.state_dict())

    def test_optional_backbone_pooling_without_download(self):
        model = FeatureExtractor(pooling=True, pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(output.shape, (1, 1280))


class DataTests(unittest.TestCase):
    def test_collate_handles_single_object_dict_and_padding(self):
        def annotation(objects):
            return {"annotation": {"filename": "example.jpg", "object": objects}}

        obj = {
            "name": "cat",
            "bndbox": {"xmin": "1", "ymin": "2", "xmax": "9", "ymax": "8"},
        }
        image = Image.new("RGB", (10, 12))
        images, boxes, widths, heights, _ = voc_collate_fn(
            [(image, annotation(obj)), (image, annotation([obj, obj]))], reshape_size=16
        )
        self.assertEqual(images.shape, (2, 3, 16, 16))
        self.assertEqual(boxes.shape, (2, 2, 5))
        self.assertTrue((boxes[0, 1] == -1).all())
        torch.testing.assert_close(boxes[0, 0, :4], torch.tensor([1.0, 2.0, 9.0, 8.0]))
        self.assertEqual(widths.tolist(), [10, 10])
        self.assertEqual(heights.tolist(), [12, 12])

    def test_existing_dataset_does_not_require_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "VOCdevkit" / "VOC2007").mkdir(parents=True)
            with patch("a5_helper.datasets.VOCDetection") as dataset:
                get_pascal_voc2007_data(temp_dir, "val")
            dataset.assert_called_once_with(
                temp_dir, year="2007", image_set="val", download=False
            )

    def test_export_uses_requested_directory_and_pixel_coordinates(self):
        class FixedDetector(nn.Module):
            def inference(self, images, **kwargs):
                return (
                    [torch.tensor([[-1.0, -1.0, 8.0, 8.0]])],
                    [torch.tensor([[0.9]])],
                    [torch.tensor([[0]])],
                )

        batch = (
            torch.zeros(1, 3, 224, 224),
            torch.tensor([[[0.0, 0.0, 100.0, 50.0, 0.0]]]),
            torch.tensor([100]),
            torch.tensor([50]),
            ["example.jpg"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            DetectionInference(
                FixedDetector(), [batch], None, {0: "cat"}, output_dir=temp_dir
            )
            result = (Path(temp_dir) / "detection-results" / "example.txt").read_text()
            self.assertEqual(result, "cat 0.900000 0.00 0.00 100.00 50.00\n")
            self.assertTrue((Path(temp_dir) / "ground-truth" / "example.txt").is_file())


if __name__ == "__main__":
    unittest.main()

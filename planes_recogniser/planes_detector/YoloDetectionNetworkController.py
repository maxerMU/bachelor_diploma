import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from planes_detector.model import YOLOv3
from planes_detector import IDetectionNetworkController, BBox
import planes_detector.config as config

class YoloDetectionNetworkController(IDetectionNetworkController):
    _TRAINED_MODELS_PATH = "planes_detector/trained_models"

    def __init__(self):
        self.m_model = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES)
        self.m_optimizer = torch.optim.Adam(self.m_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    def LoadModel(self, modelPath):
        print("=> Loading checkpoint")
        checkpoint = torch.load(f"{self._TRAINED_MODELS_PATH}/{modelPath}", map_location=config.DEVICE)
        self.m_model.load_state_dict(checkpoint["state_dict"])
        self.m_optimizer.load_state_dict(checkpoint["optimizer"])
        print(sum(p.numel() for p in self.m_model.parameters() if p.requires_grad))

        # to update lr from prev checkpoint
        for param_group in self.m_optimizer.param_groups:
            param_group["lr"] = config.LEARNING_RATE
    
    def GetAllModels(self):
        models = []
        for _, _, filenames in os.walk(self._TRAINED_MODELS_PATH):
            for model in filenames:
                models.append(model)
        
        return models

    def GetBBoxes(self, imagePath):
        image = np.array(Image.open(imagePath).convert("RGB"))
        width, height = image.shape[0], image.shape[0]
        augmentations = config.test_transforms(image=image, bboxes=[])
        image = augmentations["image"]

        results = self._GetAllBBoxes(image)

        # class pred, confidence, x, y, width, height
        nmsBoxes = self.NonMaxSuppression(results[0], iou_threshold=config.NMS_IOU_THRESH, threshold=config.CONF_THRESHOLD, box_format="midpoint")

        resultBoxes = []
        for box in nmsBoxes:
            resultBoxes.append(BBox(
                int(box[2] * width),
                int(box[3] * height),
                int(box[4] * width),
                int(box[5] * height),
            ))
        
        return resultBoxes

    def _GetAllBBoxes(self, image):
        image = torch.stack([image])

        scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)

        # BATCH_SIZE x ANCHORS * S^2
        bboxes = [[]]

        self.m_model.eval()
        with torch.no_grad():
            results = self.m_model(image)

            for i in range(3):
                batch_size, _, S, _, _ = results[i].shape
                anchor = scaled_anchors[i]
                boxes_scale_i = self.Cells2ScaleBBoxes(results[i], anchor, S=S)
                bboxes[0] += boxes_scale_i[0]

            self.m_model.train()
        
        return bboxes
        
        # im2display = np.transpose(image, (1,2,0))
        # plt.imshow(im2display)
        # plt.show()

    def Cells2ScaleBBoxes(self, predictions, anchors, S):
        BATCH_SIZE = predictions.shape[0]
        num_anchors = len(anchors)
        box_predictions = predictions[..., 1:5]

        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

        cell_indices = (
            torch.arange(S)
            .repeat(predictions.shape[0], 3, S, 1)
            .unsqueeze(-1)
            .to(predictions.device)
        )
        x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
        y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
        w_h = 1 / S * box_predictions[..., 2:4]
        converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)

        return converted_bboxes.tolist()

    def NonMaxSuppression(self, bboxes, iou_threshold, threshold, box_format="corners"):
        assert type(bboxes) == list

        bboxes = [box for box in bboxes if box[1] > threshold]
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        bboxes_after_nms = []

        while bboxes:
            chosen_box = bboxes.pop(0)

            bboxes = [
                box
                for box in bboxes
                if box[0] != chosen_box[0]
                or self._IntersectionOverUnion(
                    torch.tensor(chosen_box[2:]),
                    torch.tensor(box[2:]),
                    box_format=box_format,
                )
                < iou_threshold
            ]

            bboxes_after_nms.append(chosen_box)

        return bboxes_after_nms

    def _IntersectionOverUnion(self, boxes_preds, boxes_labels, box_format="midpoint"):
        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        if box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)
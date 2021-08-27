import torch
import torch.nn as nn
from utils import iou


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.coord = 5
        self.noobj = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, 7, 7, 30)

        pred_box1 = predictions[..., 21:25]  # bbox1 => x,y,w,h
        pred_box2 = predictions[..., 26:30]  # bbox2 => x,y,w,h
        pred_confidence1 = predictions[..., 20:21]  # box1 of predicted obj
        pred_confidence2 = predictions[..., 25:26]  # box1 of predicted obj
        pred_class = predictions[..., :20]

        target_box = target[..., 21:25]
        target_obj = target[..., 20:21]
        target_class = target[..., :20]

        iou1 = iou(pred_box1, target_box)
        iou2 = iou(pred_box2, target_box)
        iou_result = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0)

        _, responsible_box = torch.max(iou_result, dim=0)

        box_mask = target[..., 20].unsqueeze(-1)

        box_predictions = box_mask * (responsible_box * pred_box2 + (1 - responsible_box) * pred_box1)
        box_targets = box_mask * target_box

        # box_predictions[..., 0:2] = box_predictions[..., 0:2] * 7
        # box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs((box_predictions[..., 2:4]) + 1e-6))
        box_predictions[..., 2:4] = torch.sqrt(torch.clamp((box_predictions[..., 2:4] + 1e-6), min=0., max=1.0))
        # box_targets[..., 0:2] = box_targets[..., 0:2] * 7
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        coord_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))
        coord_loss = coord_loss * self.coord

        pred_box = (responsible_box * pred_confidence2 + (1 - responsible_box) * pred_confidence1)

        obj_loss = self.mse(torch.flatten(box_mask * pred_box), torch.flatten(box_mask * target[..., 20:21]))

        noobj_loss = self.mse(torch.flatten((1 - box_mask) * pred_confidence1, start_dim=1), torch.flatten((1 - box_mask) * target_obj, start_dim=1))
        noobj_loss = noobj_loss + self.mse(torch.flatten((1 - box_mask) * pred_confidence2, start_dim=1), torch.flatten((1 - box_mask) * target_obj, start_dim=1))
        noobj_loss = noobj_loss * self.noobj

        class_loss = self.mse(torch.flatten(box_mask * pred_class, end_dim=-2), torch.flatten(box_mask * target_class, end_dim=-2))

        total_loss = (coord_loss + obj_loss + noobj_loss + class_loss)

        return total_loss, coord_loss, obj_loss, noobj_loss, class_loss
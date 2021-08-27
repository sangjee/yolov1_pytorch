import torch
from torch.utils.data import Dataset
import json
import os
import random
from PIL import Image
import torchvision.transforms.functional as FT

def resize(image, boxes, dims=(448, 448), return_percent_coords=True):
    new_image = FT.resize(image, dims)

    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def transform(image, boxes, labels, difficulties, split):
    assert split in {'TRAIN', 'TEST'}

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties


    new_image, new_boxes = resize(new_image, new_boxes, dims=(448, 448))

    new_image = FT.to_tensor(new_image)

    return new_image, new_boxes, new_labels, new_difficulties


class PascalVOCDataset(Dataset):
    def __init__(self, data_folder, split, keep_difficult=False):
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        file_name = self.images[i]

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        box_num = boxes.shape[0]
        target = torch.zeros(7, 7, 30)

        for box in range(box_num):
            class_label = int(labels[box])
            xmin = boxes[box][0]  # normalized coordinate
            ymin = boxes[box][1]
            xmax = boxes[box][2]
            ymax = boxes[box][3]

            absolut_x = xmin * 448  # absolut coordinate
            absolut_y = ymin * 448
            absolut_w = xmax * 448
            absolut_h = ymax * 448

            x = (xmin + xmax) * 1.0 / 2
            y = (ymin + ymax) * 1.0 / 2
            w = (xmax - xmin)
            h = (ymax - ymin)

            i = int(7 * y)
            j = int(7 * x)

            # xcell = 7 * x - j  # translate
            # ycell = 7 * y - i
            #
            xcell = 7 * x
            ycell = 7 * y

            wcell = w * 7
            hcell = h * 7

            xcenter = (absolut_x + absolut_w) * 1.0 / 2
            ycenter = (absolut_y + absolut_h) * 1.0 / 2
            width = (absolut_w - absolut_x)
            hight = (absolut_h - absolut_y)

            vocformat = [absolut_x, absolut_y, absolut_w, absolut_h, class_label]
            yoloformat = [xcenter, ycenter, width, hight, class_label]
            cellformat = [xcell, ycell, w, h]

            if target[i, j, 20] == 0:
                target[i, j, 20] = 1
                box_target = torch.tensor([x, y, w, h])
                target[i, j, 21:25] = box_target
                target[i, j, class_label] = 1

        return image, target, file_name, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = list()
        target = list()
        filename = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            target.append(b[1])
            filename.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        target = torch.stack(target, dim=0)

        return images, target, filename, difficulties
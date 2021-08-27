import torch
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image, ImageDraw


from vocdatasets import PascalVOCDataset
from model import Yolov1
from loss import YoloLoss
from utils import transform, format_yoloTovoc, get_bboxes, mean_average_precision

from torchsummary import summary as summary


DATA_FOLDER = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test001/data/out"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 2 # 64 in original paper
EPOCHS = 501
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
KEEP_DIFFICULT = True
LOAD_MODEL_FILE = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test003/data/YOLOv1_test3_4.pth.tar"
SAVE_MODEL_FILE = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test003/data/YOLOv1_test3_4.pth.tar"
checkpoint = False
epochs_since_improvement = 0
start_epoch = 0
best_loss = 100.
print_freq = 135

def train():
    global start_epoch
    BATCH_SIZE = 24
    torch.cuda.empty_cache()
    # model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model = Yolov1().to(DEVICE)
    loss_fn = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 105], gamma=0.1)
    mean_loss = []

    if checkpoint == True :
        model_load = torch.load(LOAD_MODEL_FILE)
        model.load_state_dict(model_load['state_dict'])
        optimizer.load_state_dict(model_load['optimizer'])
        epoch = model_load['epoch']
        start_epoch = epoch + 1
        print("epoch : ",epoch)
        print("load pretrained model")
        del model_load

    train_dataset = PascalVOCDataset(DATA_FOLDER, split='train', keep_difficult=KEEP_DIFFICULT)
    val_dataset = PascalVOCDataset(DATA_FOLDER, split='test', keep_difficult=KEEP_DIFFICULT)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=NUM_WORKERS,
                                             pin_memory=True)

    for epoch in range(start_epoch, EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for iter, (images, targets, filename, _) in enumerate(loop):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            out = model(images)
            loss, box, obj, noobj, c_loss = loss_fn(out,targets)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print("loss : ",loss)
            # print("box : ",box)
            # print("obj : ",obj)
            # print("noobj : ",noobj)
            # print("c_loss : ",c_loss)

            loop.set_postfix(loss=loss.item())
        del images, targets, out
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )


        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print()
        print(f"Train mAP: {mean_avg_prec}")

        print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
        # if epoch %10 == 0 :
        state = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "mean_loss": mean_loss,
            "epoch": epoch}
        torch.save(state, SAVE_MODEL_FILE)
        print("save")
        print("epoch : ", epoch)
        print("------------end of epoch----------")

def train2():
    global start_epoch
    BATCH_SIZE = 1
    torch.cuda.empty_cache()
    # model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model = Yolov1().to(DEVICE)
    loss_fn = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 105], gamma=0.1)
    mean_loss = []

    if checkpoint == True :
        model_load = torch.load(LOAD_MODEL_FILE)
        model.load_state_dict(model_load['state_dict'])
        optimizer.load_state_dict(model_load['optimizer'])
        epoch = model_load['epoch']
        start_epoch = epoch + 1
        print("epoch : ",epoch)
        print("load pretrained model")
        del model_load

    train_dataset = PascalVOCDataset(DATA_FOLDER, split='train', keep_difficult=KEEP_DIFFICULT)
    val_dataset = PascalVOCDataset(DATA_FOLDER, split='test', keep_difficult=KEEP_DIFFICULT)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=NUM_WORKERS,
                                             pin_memory=True)

    for epoch in range(start_epoch, EPOCHS):
        # loop = tqdm(train_loader, leave=True)
        for iter, (images, targets, filename, _) in enumerate(val_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            out = model(images)
            loss, box, obj, noobj, c_loss = loss_fn(out,targets)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            print("box : ",box)
            print("obj : ",obj)
            print("noobj : ",noobj)
            print("c_loss : ",c_loss)
            print("loss : ",loss)

            # loop.set_postfix(loss=loss.item())
        del images, targets, out
        pred_boxes, target_boxes = get_bboxes(
            val_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print()
        print(f"Train mAP: {mean_avg_prec}")

        print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
        if epoch %20 == 0 :
            state = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "mean_loss": mean_loss,
                "epoch": epoch}
            torch.save(state, SAVE_MODEL_FILE)
            print("save")
        print("epoch : ", epoch)
        print("------------end of epoch----------")

def test():
    checkpoint_path = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test003/data/YOLOv1_test3_3.pth.tar"
    img_path = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test002/testdata/JPEGImages/000007.jpg"
    # img_path = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test001/data/VOCdevkit/VOC2007/JPEGImages/000007.jpg"

    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')

    checkpoint = torch.load(checkpoint_path)
    model = Yolov1().to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    img = transform(original_image)
    img = img.to(DEVICE)
    img = img.unsqueeze(0)
    out = model(img)

    draw_img = torch.squeeze(img)
    draw_img = transforms.ToPILImage()(draw_img)
    draw = ImageDraw.Draw(draw_img)
    print(out.shape)
    out = out.squeeze(0)
    print(out.shape)
    pred_bbox_info = []

    for i in range(7):
        for j in range(7):
            pred_conf1 = out[i, j, 20].tolist()
            pred_conf2 = out[i, j, 25].tolist()
            pred_bbox = []
            pred_class = []
            pred_conf = []
            if pred_conf1 > pred_conf2:
                pred_bbox = out[i, j, 21:25].tolist()
                pred_class = out[i, j, :20].tolist()
                pred_conf = pred_conf1
            else:
                pred_bbox = out[i, j, 26:30].tolist()
                pred_class = out[i, j, :20].tolist()
                pred_conf = pred_conf2
            bbox_info = format_yoloTovoc(pred_bbox, pred_class, pred_conf)
            pred_bbox_info.append(bbox_info)

    for info in pred_bbox_info:
        print(info)
    #     # if info['confidence'] > 0.1:
    #     xmin = info['xmin']
    #     ymin = info['ymin']
    #     xmax = info['xmax']
    #     ymax = info['ymax']
    #     draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(0, 0, 255), width=2)
    #     # draw.text(xy=(xmin,ymin),text=info['class_name'], fill='white')
    #     # print(info['class_name'])
    # # display(draw_img)
def test2():
    checkpoint_path = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test003/data/YOLOv1_test3_4.pth.tar"
    img_path = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test002/testdata/JPEGImages/000007.jpg"
    # img_path = "C:/Users/All Users/Anaconda3/envs/tensorflow/yolov1test001/data/VOCdevkit/VOC2007/JPEGImages/000007.jpg"

    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')

    checkpoint = torch.load(checkpoint_path)
    model = Yolov1().to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    img = transform(original_image)
    img = img.to(DEVICE)
    img = img.unsqueeze(0)
    out = model(img)
    out = out.reshape((-1, 7, 7, ((5 * 2) + 20)))

    draw_img = torch.squeeze(img)
    draw_img = transforms.ToPILImage()(draw_img)
    draw = ImageDraw.Draw(draw_img)
    print(out.shape)
    out = out.squeeze(0)
    print(out.shape)
    pred_bbox_info = []

    for i in range(7):
        for j in range(7):
            pred_conf1 = out[i, j, 20].tolist()
            pred_conf2 = out[i, j, 25].tolist()
            out[i,j,23:25] = torch.sqrt(torch.clamp((out[i, j, 23:25] + 1e-6), min=0.))
            out[i,j,28:30] = torch.sqrt(torch.clamp((out[i, j, 28:30] + 1e-6), min=0.))
            # out[i, j, 21:23] = out[i, j, 21:23] * 7
            # out[i, j, 26:28] = out[i, j, 26:28] * 7
            pred_bbox = []
            pred_class = []
            pred_conf = []
            # out[i, j, 23:25] = torch.sqrt(torch.abs(out[i, j, 23:25]))
            if pred_conf1 > pred_conf2:
                pred_bbox = out[i, j, 21:25].tolist()
                pred_class = out[i, j, :20].tolist()
                pred_conf = pred_conf1
            else:
                pred_bbox = out[i, j, 26:30].tolist()
                pred_class = out[i, j, :20].tolist()
                pred_conf = pred_conf2
            bbox_info = format_yoloTovoc(pred_bbox, pred_class, pred_conf)
            pred_bbox_info.append(bbox_info)

    for info in pred_bbox_info:
        if info['confidence'] > 0.5:
            print(info)
    #     xmin = info['xmin']
    #     ymin = info['ymin']
    #     xmax = info['xmax']
    #     ymax = info['ymax']
    #     draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(0, 0, 255), width=2)
    #     # draw.text(xy=(xmin,ymin),text=info['class_name'], fill='white')
    #     # print(info['class_name'])
    # # display(draw_img)

def model_test():
    # model = Yolov1().to((DEVICE))
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    print(summary(model,input_size=(3,448,448),batch_size=1))

if __name__ == '__main__':
    checkpoint = True
    train()
# -*- coding:utf-8 -*-

from models.yolov8_pose import YOLOV8PoseInfer
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
import json
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# yolov8n-pose的权重
weights = r'./weights/yolov8n-pose.pt'
# 此处的conf=0.30, iou=0.25，则之后推理时也必须使用这个参数，即需要保证数据一致性
model = YOLOV8PoseInfer(weights, DEVICE, 0.30, 0.25)

with open(r'./class2idx.json', 'r') as f:
    class2idx = json.load(f)

classes = {class2idx[key]:key for key in class2idx.keys()}

# 需要和config.conf中记录的参数保持一致
mean = [0.6531745791435242, 0.6183152198791504, 0.6039631366729736]
std = [0.2792046070098877, 0.2701459527015686, 0.27500295639038086]

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=mean)
])

# fastvit_t8的权重
chkpoint = torch.jit.load(r'./weights/fastvit_t8.pt')

chkpoint.eval()
chkpoint.to(DEVICE)

# 测试文件夹
path = r'./test_img'
testList = os.listdir(path)
for file in testList:

    img_path = os.path.join(path, file)
    pose_img = model.infer(img_path)

    img = Image.fromarray(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)) 

    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = chkpoint(img)
    # 预测
    _, pred = torch.max(out.data, 1)
    print('Image Name:{}, predict:{}'.format(file, classes[pred.data.item()]))

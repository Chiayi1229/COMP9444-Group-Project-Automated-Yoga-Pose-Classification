# -*- coding:utf-8 -*-
import os
import cv2
from models.yolov8_pose import YOLOV8PoseInfer
from tqdm import tqdm # 进度可视化
import numpy as np
from tqdm import tqdm


import torch
from torchvision.io import read_image, ImageReadMode

def calculate_mean_std(image_paths):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = len(image_paths)

    for image_path in tqdm(image_paths):
        image = read_image(image_path,ImageReadMode.RGB).to(float) / 255.0

        current_mean = image.mean(dim=(1, 2))
        current_std = image.std(dim=(1, 2))
        if not np.isnan(current_mean.numpy()).any():
            mean += current_mean
        if not np.isnan(current_std.numpy()).any():
            std += current_std

    mean /= total_images
    std /= total_images

    return mean, std


def data_processing():
    dir_classes = os.listdir(r'./origin_img') # 遍历origin_img中文件夹名作为类别名

    # 初始化yolov8 pose模型
    weights = r'./weights/yolov8n-pose.pt'
    device = 'cuda:0'  # 使用推理设备，cuda或cpu
    # 此处的conf=0.30, iou=0.25，则之后推理时也必须使用这个参数，即需要保证数据一致性
    model = YOLOV8PoseInfer(weights, device, 0.30, 0.25)

    # 生成纯黑底pose图，存放到pose_img文件夹下
    try:
        os.mkdir(r'./pose_img')
    except Exception as e:
        print(e)

    pose_img_paths = []  # 用于记录所有pose图片位置，用于之后计算mean和std
    for dir_class in tqdm(dir_classes):
        src_path = os.path.join(r'./origin_img', dir_class)  # 原图片文件夹
        dst_path = os.path.join(r'./pose_img', dir_class)  # 将要生成的黑底骨骼图文件夹
        # 每个类别新建个文件夹
        try:
            os.mkdir(dst_path)
        except Exception as e:
            print(e)
        
        img_names = os.listdir(src_path)
        for img_name in tqdm(img_names):
            src_img_path = os.path.join(src_path, img_name) # 原图片位置
            dst_img_path = os.path.join(dst_path, img_name) # 骨骼图位置
            # 模型推理
            try:
                dst_img = model.infer(src_img_path)
                # 保存图片到骨骼图位置
                cv2.imwrite(dst_img_path, dst_img)
                pose_img_paths.append(dst_img_path)
            except Exception as e:
                print(e)

    # 计算图片均值和方差
    mean, std = calculate_mean_std(pose_img_paths)
    print(f"mean: {mean}, std: {std}")
    with open(r'./config.conf', "w") as f:
        f.write("mean: "+str(mean.tolist()))
        f.write("\n")
        f.write("std: "+str(std.tolist()))
        f.write("\n")


if __name__ == "__main__":
    data_processing()
# -*- coding:utf-8 -*-
import random
import cv2
import numpy as np
import torch
 
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
 
class YOLOV8PoseInfer:
    def __init__(self, weights, device, conf_thres, iou_thres) -> None:
        self.imgsz = 640
        self.device = device 
        self.model = AutoBackend(weights, device=torch.device(device))
        self.model.eval()
        self.names = self.model.names
        self.half = False
        self.conf = conf_thres
        self.iou = iou_thres
        self.color = {"font": (255, 255, 255)}
        self.color.update(
            {self.names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
             for i in range(len(self.names))})
 
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                    [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                 [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                 [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                 [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
        self.kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
 
    def infer(self, img_path):
        img_src = cv2.imread(img_path)

        img = self.precess_image(img_src, self.imgsz, self.half, self.device)
        preds = self.model(img)  # shape [1, 56, 6300]
        det = ops.non_max_suppression(preds, self.conf, self.iou, classes=None, agnostic=False, max_det=300, nc=len(self.names))

        # 此处由于后续要根据骨架做姿态检测，只取画面中最明显的一个det(置信度最高的)。
        # 由于nms已经排好序，因此直接取 第一个即可，因此在for循环内部return一次，这样第一个det就直接return了
        # 为了处理det中长度为0的情况，在for循环外也return一次。这里return两次不是写错了，需要注意
 
        for i, pred in enumerate(det):
            lw = max(round(sum(img_src.shape) / 2 * 0.003), 2)  # line width
            tf = max(lw - 1, 1)  # font thickness
            sf = lw / 3  # font scale
 
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img_src.shape)
            pred_bbox = pred[:, :6].cpu().detach().numpy()
 
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, img_src.shape)
            pred_kpts = pred_kpts.cpu().detach().numpy()
 
            for kpts, bbox in zip(pred_kpts, pred_bbox):
                self.draw_kpts(img_src, kpts, bbox[:4], bbox[4], self.names[bbox[5]], lw, sf, tf)
                return img_src
 
        return img_src

    def draw_kpts(self, img_src, kpts, box, score, name, lw, sf, tf, shape=(640, 640), radius=5, kpt_line=True):
        flag = False
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
 
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]]
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(img_src, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
 
        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(img_src, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
 
 
    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
 
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
 
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
 
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
 
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
 
    def precess_image(self, img_src, img_size, half, device):
        # Padded resize
        img = self.letterbox(img_src, img_size)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
 
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img
 
 
if __name__ == '__main__':
    weights = r'./yolov8n-pose.pt'
    device = 'cuda:0'
    model = YOLOV8PoseInfer(weights, device, 0.45, 0.45)
 
    img_path = r'./bus.jpg'
    img_src = model.infer(img_path)
    cv2.imshow("result", img_src)
    cv2.waitKey(0)
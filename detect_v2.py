#Object Crop Using YOLOv7
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from blur_agli import apply_blur, apply_strong_blur
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box,plot_alignment
from utils.torch_utils import select_device, load_classifier, time_synchronized

import numpy as np

import io
import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
from scipy.spatial import ConvexHull

# def detect_landmark(input_img, detected_faces):
#     # Optionally set detector and some additional detector parameters
#     face_detector = 'sfd'
#     face_detector_kwargs = {
#         "filter_threshold" : 0.8
#     }

#     # Run the 3D face alignment on a test image, without CUDA.
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cuda', flip_input=True,
#                                     face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

#     # try:
#     #     input_img = io.imread('../data/alignment/aflw-test.jpg')
#     # except FileNotFoundError:
#     #     input_img = io.imread('data/alignment/aflw-test.jpg')

#     preds = fa.get_landmarks(input_img, detected_faces)[-1]
#     points = preds[:,:2]


#     # 计算凸包
#     hull = ConvexHull(points)

#     # 绘制点和凸包
#     plt.scatter(points[:, 0], points[:, 1], label='Random Points')

#     # 绘制凸包的边
#     for simplex in hull.simplices:
#         plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

#     # 标注顶点
#     for i in hull.vertices:
#         plt.text(points[i, 0], points[i, 1], f'{i}', color='blue')

#     plt.title('Convex Hull of 68 Random Points')
#     plt.legend()
#     plt.savefig("squares.png")
#     plt.show()

#     return points


def detect_landmark(input_img, detected_faces): 
    # Optionally set detector and some additional detector parameters
    face_detector = 'sfd'
    face_detector_kwargs = {
        "filter_threshold" : 0.8
    }

    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cuda', flip_input=False,
                                    face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

    # Get the landmarks
    preds = fa.get_landmarks(input_img, detected_faces)
    
    return preds

# def draw_rounded_rectangle(mask, top_left, bottom_right, radius, color, thickness=-1):
#     """
#     在掩码上绘制一个圆角矩形
#     :param mask: 掩码图像
#     :param top_left: 矩形框的左上角
#     :param bottom_right: 矩形框的右下角
#     :param radius: 圆角半径
#     :param color: 矩形框的颜色 (255 表示白色区域)
#     :param thickness: 线条的粗细 (-1 表示填充)
#     :return: 带有圆角矩形的掩码
#     """
#     x1, y1 = top_left
#     x2, y2 = bottom_right
    
#     # 绘制中间的直线部分
#     cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
#     cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
#     # 绘制四个圆角
#     cv2.ellipse(mask, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)  # 左上角
#     cv2.ellipse(mask, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)  # 右上角
#     cv2.ellipse(mask, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)  # 左下角
#     cv2.ellipse(mask, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)  # 右下角

#     return mask



# def apply_rounded_rect_blur(image, blurratio, radius=15):
#     """
#     对图像的圆角矩形区域应用模糊
#     :param image: 输入图像
#     :param radius: 圆角半径
#     :return: 处理后的图像
#     """
#     # 获取图像的宽度和高度
#     blend_ratio = 0.7
#     height, width = image.shape[:2]
    
#     # 定义矩形框的左上角和右下角
#     top_left = (0, 0)
#     bottom_right = (width, height)
    
#     # 创建一个全黑的掩码
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
#     # 在掩码上绘制覆盖整个图像的圆角矩形
#     mask = draw_rounded_rectangle(mask, top_left, bottom_right, radius, color=255, thickness=-1)

#     # 对图像进行模糊处理
#     blurred_image = cv2.GaussianBlur(image, (blurratio, blurratio), 0)

#     # 创建结果图像，复制原图
#     # result = np.copy(image)
#     result = cv2.addWeighted(image, blend_ratio, blurred_image, 1 - blend_ratio, 0)

#     # 通过掩码将模糊应用到圆角矩形区域
#     result[mask == 255] = blurred_image[mask == 255]

#     return result



# def get_mask(mask, points):
#     points = np.array(points, dtype=np.int32)
    
#     if len(points) > 2:  # Convex hull requires at least 3 points
#         hull = cv2.convexHull(points)

#         # Fill the convex hull on the mask with white color (255)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # White area is the mask

#     return mask

# def get_mask(mask, points, randomize_points=True, random_fraction=0.00):
#     points = np.array(points, dtype=np.int32)
    
#     if len(points) > 2:  # Convex hull requires at least 3 points
#         hull = cv2.convexHull(points)

#         # Fill the convex hull on the mask with white color (255)
#         cv2.fillConvexPoly(mask, hull, color=(255, 255, 255))  # White area is the mask
        
#         if randomize_points:
#             # 找到掩码区域中的所有点
#             mask_points = np.column_stack(np.where(mask == 255))

#             # 根据 random_fraction 确定随机选取的点数
#             num_random_points = int(len(mask_points) * random_fraction)

#             # 使用 numpy.random.choice 进行随机采样
#             chosen_indices = random.choice(len(mask_points), num_random_points, replace=False)
#             random_points = mask_points[chosen_indices]

#             # 在这些随机点上做处理，比如填充不同的颜色
#             for point in random_points:
#                 cv2.circle(mask, (point[1], point[0]), 1, (127, 127, 127), -1)  # 用灰色（127）填充这些点

#     return mask

# def apply_rounded_alignment_blur(image, landmarks, blurratio):
#     blend_ratio =0.7
#     # 创建一个全黑的掩码
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
#     # 在掩码上绘制覆盖整个图像的圆角矩形
#     mask = get_mask(mask, landmarks)

#     # 对图像进行模糊处理
#     # blurred_image = cv2.blur(image, (blurratio, blurratio))
#     # 对图像进行高斯模糊处理
#     blurred_image = cv2.GaussianBlur(image, (blurratio, blurratio), 0)

#     # 创建结果图像，复制原图
#     result = np.copy(image)

#     soft_mask  = cv2.GaussianBlur(mask, (blurratio, blurratio), 0)

#     # 将掩码归一化到0到1之间，方便混合操作
#     soft_mask = soft_mask.astype(float) / 255.0

#     # 创建结果图像，复制原图
#     # result = np.copy(image)
#     blended_image = cv2.addWeighted(image, blend_ratio, blurred_image, 1 - blend_ratio, 0)

#     for c in range(0, 3):  # 遍历图像的三个通道
#         result[:, :, c] = (soft_mask * blended_image[:, :, c] + (1 - soft_mask) * image[:, :, c])


#     return result


# def apply_rounded_alignment_blur(image, landmarks, blurratio):
#     blend_ratio = 0.01  # 原图与模糊图的混合比例
#     # 创建一个全黑的掩码
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)

#     # 在掩码上绘制覆盖整个图像的圆角矩形 (假设 get_mask 函数已正确返回了圆角矩形的 mask)
#     mask = get_mask(mask, landmarks, randomize_points=False)

#     # 对图像进行高斯模糊处理
#     blurred_image = cv2.GaussianBlur(image, (blurratio, blurratio), 0)

#     # 创建结果图像，复制原图
#     result = np.copy(image)

#     # 对掩码进行高斯模糊，产生软边缘，实现渐变过渡
#     # 核大小根据需要调整，使过渡更自然
#     soft_mask = cv2.GaussianBlur(mask, (blurratio, blurratio), 0)

#     # 将掩码归一化到0到1之间，方便混合操作
#     soft_mask = soft_mask.astype(float) / 255.0

#     # 混合原图和模糊图
#     blended_image = cv2.addWeighted(image, blend_ratio, blurred_image, 1 - blend_ratio, 0)

#     # 根据软掩码进行图像融合处理
#     for c in range(3):  # 遍历图像的三个通道 (RGB)
#         result[:, :, c] = (soft_mask * blended_image[:, :, c] + (1 - soft_mask) * image[:, :, c])

#     return result


# def apply_rounded_alignment_blur(image, landmarks, blurratio):
#     blend_ratio = 0.3  # 调整混合比例，使模糊图占更多权重
#     # 创建一个全黑的掩码
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)

#     # 在掩码上绘制覆盖整个图像的圆角矩形
#     mask = get_mask(mask, landmarks)

#     # 对图像进行高斯模糊处理
#     blurred_image = cv2.GaussianBlur(image, (blurratio, blurratio), 0)

#     # 增加模糊处理的次数，进一步增强模糊效果
#     blurred_image = cv2.GaussianBlur(blurred_image, (blurratio, blurratio), 0)

#     # 创建结果图像，复制原图
#     result = np.copy(image)

#     # 对掩码进行高斯模糊，产生软边缘，实现渐变过渡
#     soft_mask = cv2.GaussianBlur(mask, (blurratio, blurratio), 0)

#     # 将掩码归一化到0到1之间，方便混合操作
#     soft_mask = soft_mask.astype(float) / 255.0

#     # 计算边缘区域的混合
#     blended_image = cv2.addWeighted(image, 1 - blend_ratio, blurred_image, blend_ratio, 0)

#     # 根据软掩码进行图像融合处理
#     for c in range(3):  # 遍历图像的三个通道 (RGB)
#         result[:, :, c] = (soft_mask * blended_image[:, :, c] + (1 - soft_mask) * image[:, :, c])

#     # 增强渐变效果：对结果图像进行再一次模糊处理
#     result = cv2.GaussianBlur(result, (blurratio, blurratio), 0)

#     return result


def crop_with_margin(image, xyxy: np.ndarray, margin: float = 0.2):
    """
    裁剪图像，边框缩小指定比例。

    :param image: 输入图像
    :param xyxy: 裁剪区域的坐标 [x1, y1, x2, y2]
    :param margin: 缩小边框的比例，默认为 0.2 (20%)
    :return: 裁剪后的图像
    """
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
    
    # print("x1, y1, x2, y2",x1, y1, x2, y2)
    
    # 计算宽度和高度
    width = x2 - x1
    height = y2 - y1

    # 缩小边框
    new_width = width * (1 - margin)
    new_height = height * (1 - margin)

    # 计算新的坐标（保持中心不变）
    x1_new = x1 + (width - new_width) / 2
    y1_new = y1 + (height - new_height) / 2
    x2_new = x2 - (width - new_width) / 2
    y2_new = y2 - (height - new_height) / 2

    # 裁剪图像
    crop_obj = image[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
    new_coords = np.array([int(x1_new), int(y1_new), int(x2_new), int(y2_new)])
    return crop_obj, new_coords

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, blurratio,hidedetarea = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.blurratio,opt.hidedetarea
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # detected_faces = []
                # for *xyxy, conf, cls in reversed(det):
                #     detected_faces.append(np.array([int(xyxy[0]), int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]))
                    
                # alignment_preds = detect_landmark(im0, detected_faces)

                alignment_preds= []
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    #Add Object Blurring Code
                    #..................................................................
                    crop_obj = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                    
                    detected_face = [np.array([int(xyxy[0]), int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])]
                    landmarks = detect_landmark(im0, detected_face)[-1][:,:2]
                    alignment_preds.append(landmarks)
                    
                    cropped_landmarks = landmarks - np.array([int(xyxy[0]), int(xyxy[1])])
                    
                    # crop_obj, new_xyxy = crop_with_margin(im0, xyxy)
                    # blur = cv2.blur(crop_obj,(blurratio,blurratio))
                    
                    
                    # blur = apply_circular_blur(crop_obj, blurratio)
                    # blur = random_closed_loop(crop_obj, blurratio)
                    # blur = random_near_rect_loop(crop_obj, blurratio)
                    # blur = apply_elliptical_blur(crop_obj, blurratio)
                    # blur = apply_rounded_rect_blur(crop_obj, blurratio)
                    # blur = apply_rounded_alignment_blur(crop_obj, cropped_landmarks, blurratio)
                    blur = apply_blur(crop_obj, cropped_landmarks, blurratio,im0[landmarks[3]])
                    im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
                    # im0[int(new_xyxy[1]):int(new_xyxy[3]),int(new_xyxy[0]):int(new_xyxy[2])] = blur
                    #..................................................................
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if not hidedetarea:
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            plot_alignment(im0, alignment_preds)
                            
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--blurratio',type=int,default=20, required=True, help='blur opacity')
    parser.add_argument('--hidedetarea',action='store_true', help='Hide Detected Area')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
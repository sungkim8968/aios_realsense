import aios
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import argparse
import time
import socket
import pygame
from datetime import datetime

x = 640
y = 480
w = 0
h = 0
center_x = 320
center_y = 240


class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(
            self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(
                    self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh),
                                 interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height *
                                 hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh),
                                 interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(
                srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(
            cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # if isinstance(outputs, tuple):
        #     print("1")
        #     outputs = list(outputs)
        # if float(cv2.__version__[:3])>=4.7:
        #     print("2")
        #     outputs = [outputs[2], outputs[0], outputs[1]] ###opencv4.7需要这一步，opencv4.5不需要
        # Perform inference on the image
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(
            outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / \
                  (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            # x1,y1,score1, ..., x5,y5,score5
            kpts = pred[..., -15:].reshape((-1, 15))

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(
                self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride]
                                                    [:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride]
                                                    [:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  # 合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]),
                            5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  # xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  # max_class_confidence

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  # 合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold)
        # print(indices)
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def draw_detections(self, image, boxes, scores, kpts):
        global x, y, w, h, center_x, center_y
        dist_to_center_tmp = 1000
        center_x_tmp1 = 320
        center_y_tmp1 = 240
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            center_x_tmp2 = x + int(w / 2)
            center_y_tmp2 = y + int(h / 2)
            dist_to_center = depth_frame.get_distance(center_x, center_y)
            print("******************距离为： ", dist_to_center)
            if dist_to_center_tmp > dist_to_center > 0 and score > 0.79:
                dist_to_center_tmp = dist_to_center
                center_x_tmp1 = center_x_tmp2
                center_y_tmp1 = center_y_tmp2
            # print("x", x)
            # print("y", y)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h),
                          (0, 0, 255), thickness=3)
            cv2.putText(image, "face:" + str(round(score, 2)), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            for i in range(5):
                cv2.circle(
                    image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
                # cv2.putText(image, str(i), (int(kp[i * 3]), int(kp[i * 3 + 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                # (255, 0, 0), thickness=1)
        center_x = center_x_tmp1
        center_y = center_y_tmp1

        return image


Server_IP_list = []


#     调用五次多项式对p1, p2 进行规划
def five_poly_programming(p1, v1, a1, p2, p3, t1, t2, dt):
    v2 = (p3 - p1) / (t1 + t2)
    a2 = (v2 - v1) / t1
    model_fpp = QuinticPolynomial(p1, v1, a1, p2, v2, a2, t1)
    points = []
    for T in np.arange(t1, t2, dt):
        point = model_fpp.calc_point(T)
        points.append(point)
    return v2, a2, points


def jog():
    qs_right = np.loadtxt('waving/right_arm.txt')
    qs_left = np.loadtxt('waving/left_arm.txt')
    pygame.init()
    pygame.joystick.init()
    done = False

    # 左手
    pos_left_1, vel_left_1, cur_left_1 = aios.getCVP('10.10.20.10', 1)
    pos_left_2, vel_left_2, cur_left_2 = aios.getCVP('10.10.20.11', 1)
    pos_left_3, vel_left_3, cur_left_3 = aios.getCVP('10.10.20.12', 1)
    pos_left_4, vel_left_4, cur_left_4 = aios.getCVP('10.10.20.13', 1)
    pos_left_5, vel_left_5, cur_left_5 = aios.getCVP('10.10.20.14', 1)
    pos_left_6, vel_left_6, cur_left_6 = aios.getCVP('10.10.20.15', 1)
    pos_left_7, vel_left_7, cur_left_7 = aios.getCVP('10.10.20.16', 1)
    # 右手
    pos_right_1, vel_right_1, cur_right_1 = aios.getCVP('10.10.20.30', 1)
    pos_right_2, vel_right_2, cur_right_2 = aios.getCVP('10.10.20.31', 1)
    pos_right_3, vel_right_3, cur_right_3 = aios.getCVP('10.10.20.32', 1)
    pos_right_4, vel_right_4, cur_right_4 = aios.getCVP('10.10.20.33', 1)
    pos_right_5, vel_right_5, cur_right_5 = aios.getCVP('10.10.20.34', 1)
    pos_right_6, vel_right_6, cur_right_6 = aios.getCVP('10.10.20.35', 1)
    pos_right_7, vel_right_7, cur_right_7 = aios.getCVP('10.10.20.36', 1)
    while 1:
        # for event in pygame.event.get():  # User did something
        #     if event.type == pygame.QUIT:  # If user clicked close
        #         done = True  # Flag that we are done so we exit this loop
        # joystick_count = pygame.joystick.get_count()
        # for i in range(joystick_count):
        #     joystick = pygame.joystick.Joystick(i)
        #     joystick.init()
        #     putton0 = joystick.get_button(0)
        #     if putton0 == 1:
        for q in range(len(qs_right)):
            aios.setPosition(qs_right[q][0] / 360 * 51, 0, 0, False, '10.10.20.30', 1)
            aios.setPosition(qs_right[q][1] / 360 * 51, 0, 0, False, '10.10.20.31', 1)
            aios.setPosition(qs_right[q][2] / 360 * 31, 0, 0, False, '10.10.20.32', 1)
            aios.setPosition(qs_right[q][3] / 360 * 51, 0, 0, False, '10.10.20.33', 1)
            aios.setPosition(qs_right[q][4] / 360 * 31, 0, 0, False, '10.10.20.34', 1)
            aios.setPosition(qs_right[q][5] / 360 * 51, 0, 0, False, '10.10.20.35', 1)
            aios.setPosition(qs_right[q][6] / 360 * 31, 0, 0, False, '10.10.20.36', 1)

            aios.setPosition(qs_left[q][0] / 360 * 51, 0, 0, False, '10.10.20.10', 1)
            aios.setPosition(qs_left[q][1] / 360 * 51, 0, 0, False, '10.10.20.11', 1)
            aios.setPosition(qs_left[q][2] / 360 * 51, 0, 0, False, '10.10.20.12', 1)
            aios.setPosition(qs_left[q][3] / 360 * 51, 0, 0, False, '10.10.20.13', 1)
            aios.setPosition(qs_left[q][4] / 360 * 31, 0, 0, False, '10.10.20.14', 1)
            aios.setPosition(qs_left[q][5] / 360 * 51, 0, 0, False, '10.10.20.15', 1)
            aios.setPosition(qs_left[q][6] / 360 * 31, 0, 0, False, '10.10.20.16', 1)

            time.sleep(0.003)
        time.sleep(1)
        #     print("*******")qs_right
        # print('数目： ', putton0)


def robot():
    # 头
    all_x, vel_x, cur = aios.getCVP('10.10.20.93', 1)  # 左右摆头
    all_y, vel_y, cur = aios.getCVP('10.10.20.95', 1)  # 上下点头
    #
    p1 = all_x
    v1 = 0
    a1 = 3.1415926 / 2
    # pos_waist, vel_waist, cur_waist = aios.getCVP('10.10.20.90', 1)
    all_y = -all_y
    num = 1
    p2 = 0.0
    p3 = 0.0
    list_points = []
    while 1:
        delta_cam_x = center_x - 320
        delta_cam_y = center_y - 240
        # 最大速度 (delta_cam_x * 69)/(640 * 360)， 实际上应该再乘以一个缩小比例因子
        delta_act_x = (delta_cam_x * 69) / (640 * 360)
        delta_act_y = (delta_cam_y * 42) / (480 * 360)
        if num == 2:
            p2 = all_x + delta_act_x
        if num >= 3:
            p3 = all_x + delta_act_x
            tmp_v2, tmp_a2, list_points = five_poly_programming(p1, v1, a1, p2, p3, 0.05, 0.05, 0.0025)
            for point in list_points:
                aios.setPosition(point, 0, 0, False, '10.10.20.93', 1)
                time.sleep(0.0025)
            v1 = tmp_v2
            a1 = tmp_a2
            p1 = p2
            p2 = p3
            if (all_y / 31) * 360 < -8:
                print("上下最大角度是-8")
                # 给的是腰的角度
                aios.setPosition((8 * 31 / 360), 0, 0, False, '10.10.20.95', 1)
            elif (all_y / 31) * 360 > 8:
                print("上下最大角度是8")
                # 给的是腰的角度
                aios.setPosition((-8 * 31 / 360), 0, 0, False, '10.10.20.95', 1)
            else:
                # aios.setPosition(pos_wast, 0, 0, False, '10.10.20.90', 1)
                aios.setPosition(-all_y, 0, 0, False, '10.10.20.95', 1)
                print("正常")

            if (all_x / 31) * 360 < -40:
                print("左右最大角度是-40")
                # 给的是腰的角度
                aios.setPosition((-40 * 31 / 360), 0, 0, False, '10.10.20.93', 1)
                # aios.setPosition(0, 0, 0, False, '10.10.20.90', 1)
            elif (all_x / 31) * 360 > 40:
                print("左右最大角度是40")
                # 给的是腰的角度
                aios.setPosition((40 * 31 / 360), 0, 0, False, '10.10.20.93', 1)
                # aios.setPosition(0, 0, 0, False, '10.10.20.90', 1)
            else:
                # aios.setPosition(pos_wast, 0, 0, False, '10.10.20.90', 1)
                aios.setPosition(all_x, 0, 0, False, '10.10.20.93', 1)
                print("正常")

        # time.sleep(0.0025)

        # 在此处做判断
        # if delta_x < -0.005:
        #     all_x = all_x + (-0.005)
        # elif delta_x > 0.005:
        #     all_x = all_x + 0.005
        # else:
        #     all_x = all_x + delta_x
        # all_y = all_y + delta_y

        num = + 1
        # 头转
        print("delta_act_x: ", delta_act_x)
        print("all_x: ", all_x)


if __name__ == '__main__':
    Server_IP_list = aios.broadcast_func()

    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str,
                        default='images/2.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.onnx',
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.45,
                        type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5,
                        type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize YOLOv8_face object detector
    YOLOv8_face_detector = YOLOv8_face(
        args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_device_from_file("666.bag")#这是打开相机录制的视频
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    YOLOv8_face_detector = YOLOv8_face(
        args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    # Start streaming
    pipeline.start(config)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    PORT = 8000
    server_address = ("127.0.0.1", PORT)  # 接收方 服务器的ip地址和端口号
    if Server_IP_list:

        encoderIsReady = True
        for i in range(len(Server_IP_list)):
            if not aios.encoderIsReady(Server_IP_list[i], 1):
                encoderIsReady = False
        print('\n')
        if encoderIsReady:
            for i in range(len(Server_IP_list)):
                aios.getRoot(Server_IP_list[i])
            print('\n')
            # aios.setMotorConfig
            for i in range(len(Server_IP_list)):
                enableSuccess = aios.enable(Server_IP_list[i], 1)
            print(enableSuccess)
            if enableSuccess:
                new_thread = threading.Thread(target=robot, name="T1")
                # new_thread_read = threading.Thread(target=jog, name="T2")
                new_thread.setDaemon(True)
                # new_thread_read.setDaemon(True)
                new_thread.start()
                # new_thread_read.start()
                while 1:

                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())

                    color_image = np.asanyarray(color_frame.get_data())

                    # Detect Objects
                    boxes, scores, classids, kpts = YOLOv8_face_detector.detect(
                        color_image)
                    if center_x < 640:
                        dist_to_center = depth_frame.get_distance(
                            center_x, center_y)
                    dstimg = YOLOv8_face_detector.draw_detections(
                        color_image, boxes, scores, kpts)

                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', color_image)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q') or key == 27:
                        cv2.destroyAllWindows()
                        break

import time 
import cv2
import numpy as np
import torch
import robomaster
from robomaster import robot
from robomaster import camera
from robomaster import gimbal

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        初始化YOLO模型
        """
        self.conf_threshold = conf_threshold
        # 加载YOLO模型
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None

    def detect(self, image):
        """
        使用YOLO进行目标检测
        """
        if self.model is None:
            return []
        
        # 进行推理
        results = self.model(image)
        
        # 提取检测结果
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > self.conf_threshold:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': results.names[int(cls)]
                })
        
        return detections

class HoughCircleDetector:
    def __init__(self, dp=1, min_dist=30, param1=50, param2=30, min_radius=10, max_radius=100):
        """
        初始化霍夫圆环检测器
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, image, roi=None):
        """
        在图像中检测圆环
        roi: 感兴趣区域 [x1, y1, x2, y2]
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 如果指定了ROI，只在ROI区域内检测
        if roi is not None:
            x1, y1, x2, y2 = roi
            gray_roi = gray[y1:y2, x1:x2]
        else:
            gray_roi = gray
        
        # 高斯模糊去噪
        gray_blur = cv2.GaussianBlur(gray_roi, (9, 9), 2)
        
        # 霍夫圆环检测
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 如果使用了ROI，需要调整坐标
                if roi is not None:
                    x += x1
                    y += y1
                detected_circles.append((x, y, r))
        
        return detected_circles

class RobotTracker:
    def __init__(self, ep_robot, yolo_model_path):
        """
        初始化机器人跟踪器
        """
        self.ep_robot = ep_robot
        self.ep_gimbal = ep_robot.gimbal
        self.ep_camera = ep_robot.camera
        
        # 初始化检测器
        self.yolo_detector = YOLODetector(yolo_model_path)
        self.circle_detector = HoughCircleDetector()
        
        # 跟踪参数
        self.target_class = 'person'  # 要跟踪的目标类别
        self.frame_center = (640, 360)  # 720P图像中心点
        self.follow_threshold = 50  # 中心点偏移阈值
        
    def calculate_gimbal_move(self, target_center):
        """
        计算云台需要移动的角度
        """
        dx = target_center[0] - self.frame_center[0]
        dy = target_center[1] - self.frame_center[1]
        
        # 将像素偏移转换为角度（需要根据实际相机参数调整）
        yaw_angle = -dx * 0.1  # 水平移动角度
        pitch_angle = -dy * 0.05  # 垂直移动角度
        
        # 限制角度范围
        yaw_angle = max(-30, min(30, yaw_angle))
        pitch_angle = max(-20, min(20, pitch_angle))
        
        return yaw_angle, pitch_angle
    
    def process_frame(self, frame):
        """
        处理单帧图像
        """
        # YOLO目标检测
        detections = self.yolo_detector.detect(frame)
        
        target_detection = None
        target_circles = []
        
        # 查找指定类别的目标
        for detection in detections:
            if detection['class_name'] == self.target_class:
                target_detection = detection
                break
        
        # 如果在目标周围检测到圆环
        if target_detection is not None:
            x1, y1, x2, y2 = target_detection['bbox']
            
            # 扩展ROI区域以在目标周围检测圆环
            roi_expansion = 20
            roi_x1 = max(0, x1 - roi_expansion)
            roi_y1 = max(0, y1 - roi_expansion)
            roi_x2 = min(frame.shape[1], x2 + roi_expansion)
            roi_y2 = min(frame.shape[0], y2 + roi_expansion)
            
            # 在目标周围检测圆环
            target_circles = self.circle_detector.detect(frame, [roi_x1, roi_y1, roi_x2, roi_y2])
        
        return target_detection, target_circles
    
    def draw_detections(self, frame, target_detection, target_circles):
        """
        在图像上绘制检测结果
        """
        # 绘制YOLO检测框
        if target_detection is not None:
            x1, y1, x2, y2 = target_detection['bbox']
            confidence = target_detection['confidence']
            class_name = target_detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 绘制霍夫圆环
        for (x, y, r) in target_circles:
            cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
        
        # 绘制图像中心点
        cv2.circle(frame, self.frame_center, 5, (255, 255, 255), -1)
        cv2.line(frame, (self.frame_center[0]-10, self.frame_center[1]), 
                (self.frame_center[0]+10, self.frame_center[1]), (255, 255, 255), 1)
        cv2.line(frame, (self.frame_center[0], self.frame_center[1]-10), 
                (self.frame_center[0], self.frame_center[1]+10), (255, 255, 255), 1)
        
        return frame
    
    def follow_target(self, duration=30):
        """
        执行目标跟踪
        """
        print("开始目标跟踪...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 获取图像帧
                frame = self.ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
                if frame is None:
                    continue
                
                # 处理图像
                target_detection, target_circles = self.process_frame(frame)
                
                # 绘制检测结果
                display_frame = self.draw_detections(frame.copy(), target_detection, target_circles)
                
                # 如果检测到目标且有圆环，进行跟踪
                if target_detection is not None and len(target_circles) > 0:
                    # 计算目标中心点
                    x1, y1, x2, y2 = target_detection['bbox']
                    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # 计算云台移动角度
                    yaw_angle, pitch_angle = self.calculate_gimbal_move(target_center)
                    
                    # 移动云台
                    self.ep_gimbal.move(yaw=yaw_angle, pitch=pitch_angle).wait_for_completed()
                    
                    print(f"跟踪目标: 偏移({target_center[0]-self.frame_center[0]}, "
                          f"{target_center[1]-self.frame_center[1]}), "
                          f"云台移动(yaw:{yaw_angle:.1f}, pitch:{pitch_angle:.1f})")
                
                # 显示结果
                cv2.imshow('RoboMaster Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)  # 控制处理频率
                
        except Exception as e:
            print(f"跟踪过程中出错: {e}")
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # 初始化机器人
    ep_robot = robot.Robot()
    
    try:
        # 连接机器人
        ep_robot.initialize(conn_type="sta")
        print("机器人连接成功")
        
        # 设置YOLO模型路径（请修改为你的模型路径）
        yolo_model_path = "best.pt"  # 你的YOLO模型文件路径
        
        # 创建跟踪器
        tracker = RobotTracker(ep_robot, yolo_model_path)
        
        # 启动相机
        tracker.ep_camera.start_video_stream(display=False)
        print("相机启动成功")
        
        # 等待相机稳定
        time.sleep(2)
        
        # 开始跟踪（持续30秒）
        tracker.follow_target(duration=30)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
    
    finally:
        # 清理资源
        try:
            tracker.ep_camera.stop_video_stream()
            ep_robot.close()
            print("资源清理完成")
        except:
            pass
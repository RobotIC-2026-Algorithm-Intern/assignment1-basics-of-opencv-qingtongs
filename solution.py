import cv2
import numpy as np
import json
import os


class ColorRange:
    def __init__(self, config):
        self.crange = {}
        color_config = config.get("color_ranges", {})
        for color_key, range_info in color_config.items():
            lower = range_info.get("lower", [0, 0, 0])
            upper = range_info.get("upper", [255, 255, 255])
            self.crange[color_key] = (lower, upper)

    def add_crange(self, key, value):
        self.crange[key] = value

    def color_mask(self, color, hsv_img):
        if color not in self.crange:
            print(f"未找到颜色: {color}")
            return None

        color_l = np.array(self.crange[color][0])
        color_u = np.array(self.crange[color][1])

        mask = cv2.inRange(hsv_img, color_l, color_u)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆形结构元素
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算：去除小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算：填充小孔洞
        return mask


class BallDetector:
    def __init__(self, color_range, config):
        self.color_range = color_range
        roi_config = config.get("roi", {})
        self.roi = (
            roi_config.get("x1", 200),
            roi_config.get("y1", 150),
            roi_config.get("x2", 400),
            roi_config.get("y2", 350)
        )

        det_config = config.get("detection_params", {})
        self.min_pixels_ratio = det_config.get("min_pixels_ratio", 0.01)
        self.min_pixels_fixed = det_config.get("min_pixels_fixed", 50)
        self.min_area_ratio = det_config.get("min_area_ratio", 0.005)
        # 从配置加载字体参数（使用OpenCV内置字体，无需外部文件）
        font_config = config.get("font", {})
        self.font_scale = font_config.get("size", 0.8)  # 字体缩放比例
        self.font_thickness = font_config.get("thickness", 2)  # 字体粗细
        self.opencv_font = cv2.FONT_HERSHEY_SIMPLEX  # OpenCV内置西文字体

    def set_roi(self, x1, y1, x2, y2):
        self.roi = (x1, y1, x2, y2)
        print(f"R2区域已更新为: ({x1}, {y1}) 到 ({x2}, {y2})")

    def draw_status_text(self, img, text, position, color=(0, 0, 255)):
        """在图像上绘制文本（使用OpenCV内置字体，无需外部文件）"""
        cv2.putText(
            img,
            text,  # 文本内容
            position,  # 绘制位置（左上角坐标）
            self.opencv_font,  # 字体类型
            self.font_scale,  # 缩放比例
            color,  # 颜色（BGR格式）
            self.font_thickness  # 线条粗细
        )
        return img

    def process_frame(self, frame):
        if not self.roi:
            frame = self.draw_status_text(frame, "R2 Status: ROI Not Set", (20, 30))
            return frame, "ROI Not Set"

        x1, y1, x2, y2 = self.roi
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        roi = frame[y1:y2, x1:x2]
        roi_area = (x2 - x1) * (y2 - y1)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask1 = self.color_range.color_mask('red1', hsv_roi)
        red_mask2 = self.color_range.color_mask('red2', hsv_roi)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2) if (red_mask1 is not None and red_mask2 is not None) else None
        red_count = cv2.countNonZero(red_mask) if red_mask is not None else 0

        blue_mask = self.color_range.color_mask('blue', hsv_roi)
        blue_count = cv2.countNonZero(blue_mask) if blue_mask is not None else 0

        purple_mask = self.color_range.color_mask('purple', hsv_roi)
        purple_count = cv2.countNonZero(purple_mask) if purple_mask is not None else 0

        color_counts = {
            'Red': red_count,
            'Blue': blue_count,
            'Purple': purple_count
        }
        dominant_color = max(color_counts, key=color_counts.get)
        max_count = max(color_counts.values())

        min_pixels = max(self.min_pixels_fixed, int(roi_area * self.min_pixels_ratio))
        # 若最大像素数低于阈值，或占比过低，则判定为无球
        if max_count < min_pixels or (max_count / roi_area) < self.min_area_ratio:
            status = "No Ball"
        else:
            status = f"{dominant_color} Ball"

        # 绘制ROI边框和状态文本
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边框标记ROI
        frame = self.draw_status_text(frame, f"R2 Status: {status}", (20, 30))  # 显示状态
        return frame, status

    def process_videos(self, config):
        video_config = config.get("video_paths", {})
        video_paths = [path for path in video_config.values() if os.path.exists(path)]

        if not video_paths:
            print("错误：配置文件中未找到有效的视频路径")
            return

        for path in video_paths:
            video_name = os.path.basename(path)
            print(f"开始处理视频: {video_name}")

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"错误：无法打开视频 - {video_name}")
                continue

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, status = self.process_frame(frame)

                cv2.imshow(f"Video Processing - {video_name}", processed_frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                frame_count += 1

            cap.release()
            cv2.destroyWindow(f"Video Processing - {video_name}")
            print(f"视频处理完成: {video_name} | 总帧数: {frame_count}\n")

        cv2.destroyAllWindows()


def load_config(config_path):
    """从JSON文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    config_path = os.path.join(os.getcwd(), "config.json")

    try:
        # 加载配置文件
        config = load_config(config_path)
        print("配置文件加载成功")

        # 初始化颜色范围处理器
        color_range = ColorRange(config)

        # 初始化球检测器
        detector = BallDetector(color_range, config)
        print(f"从配置加载ROI: ({detector.roi[0]}, {detector.roi[1]}) 到 ({detector.roi[2]}, {detector.roi[3]})")

        # 开始处理视频
        detector.process_videos(config)

    except Exception as e:
        print(f"程序错误: {str(e)}")


if __name__ == "__main__":
    main()
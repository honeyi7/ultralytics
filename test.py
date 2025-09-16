import cv2
import numpy as np
from ultralytics import YOLO
import random

# --- 配置参数 ---
VIDEO_PATH = './clip_jinshuiRoad_1.mp4'  # 你的视频文件路径
OUTPUT_PATH = './tracking_output.mp4'    # <--- 新增: 定义输出视频文件的路径
MODEL_PATH = './yolov8s.pt'              # YOLOv8模型文件，'n'是最小的模型，速度最快
MAX_TARGETS = 4                          # 右侧最多显示几个跟踪窗口
WINDOW_SIZE = (384, 384)                 # 右侧每个跟踪窗口的大小 (宽, 高)
CONF_THRESHOLD = 0.5                     # 只跟踪置信度高于此值的对象
CLASSES_TO_TRACK = [0, 2]                # 要跟踪的类别ID (0: person, 2: car in COCO dataset)


# --- 主函数 ---
def main():
    """
    主处理函数
    """
    # 1. 加载YOLOv8模型
    model = YOLO(MODEL_PATH)

    # 2. 打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {VIDEO_PATH}")
        return
    print()
    print(" 视频打开成功，开始处理...")
    print("111")
    print("2222")
    # 获取视频的宽度、高度和FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # <--- 新增: 获取视频的帧率

    # --- 视频写入器初始化 ---
    # <--- 新增: 计算最终输出画面的尺寸
    # 最终画面的高度由右侧窗口决定
    final_height = WINDOW_SIZE[1] * MAX_TARGETS
    # 根据新的高度计算主视频帧缩放后的宽度
    scale = final_height / frame_height
    resized_main_width = int(frame_width * scale)
    # 最终画面的宽度是缩放后的主视频 + 右侧窗口的宽度
    final_width = resized_main_width + WINDOW_SIZE[0]

    # <--- 新增: 定义视频编码器并创建VideoWriter对象
    # 对于.mp4文件，'mp4v'是一个很好的选择
    # 对于.avi文件，可以使用'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (final_width, final_height))
    print(f"视频处理结果将保存到: {OUTPUT_PATH}")
    # -------------------------

    # 3. 初始化目标跟踪相关变量
    tracked_targets = {}  # 存储当前正在跟踪的目标 {track_id: {"bbox": (x,y,w,h), "color": (B,G,R)}}

    # 为不同ID生成随机颜色
    def generate_colors(n):
        return [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for _ in range(n)]

    colors = generate_colors(100)  # 预先生成100个颜色备用

    while True:
        # 4. 读取一帧视频
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束或读取错误。")
            break

        # 5. 使用YOLOv8进行跟踪
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, classes=CLASSES_TO_TRACK, verbose=False,
                              tracker='bytetrack.yaml')

        # 6. 处理跟踪结果
        annotated_frame = frame.copy()

        current_tracked_ids = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                current_tracked_ids.append(track_id)
                x, y, w, h = box

                if track_id not in tracked_targets and len(tracked_targets) < MAX_TARGETS:
                    tracked_targets[track_id] = {
                        "bbox": (x, y, w, h),
                        "color": colors[track_id % len(colors)]
                    }
                elif track_id in tracked_targets:
                    tracked_targets[track_id]["bbox"] = (x, y, w, h)

                if track_id in tracked_targets:
                    color = tracked_targets[track_id]["color"]
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        disappeared_ids = set(tracked_targets.keys()) - set(current_tracked_ids)
        for track_id in disappeared_ids:
            del tracked_targets[track_id]

        # 7. 创建并填充右侧的跟踪窗口
        window_canvas = np.zeros((WINDOW_SIZE[1] * MAX_TARGETS, WINDOW_SIZE[0], 3), dtype=np.uint8)

        target_list = list(tracked_targets.items())
        for i in range(MAX_TARGETS):
            if i < len(target_list):
                track_id, target_info = target_list[i]
                x, y, w, h = target_info["bbox"]
                color = target_info["color"]

                crop_w, crop_h = WINDOW_SIZE[0], WINDOW_SIZE[1]
                x_center, y_center = int(x), int(y)
                crop_x1 = max(0, x_center - crop_w // 2)
                crop_y1 = max(0, y_center - crop_h // 2)
                crop_x2 = min(frame_width, x_center + crop_w // 2)
                crop_y2 = min(frame_height, y_center + crop_h // 2)

                target_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                h_crop, w_crop, _ = target_crop.shape
                padded_crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                padded_crop[0:h_crop, 0:w_crop] = target_crop

                cv2.rectangle(padded_crop, (0, 0), (crop_w - 1, crop_h - 1), color, 3)
                cv2.putText(padded_crop, f"Tracking ID: {track_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                start_y = i * WINDOW_SIZE[1]
                end_y = start_y + WINDOW_SIZE[1]
                window_canvas[start_y:end_y, :] = padded_crop
            else:
                start_y = i * WINDOW_SIZE[1]
                cv2.putText(window_canvas, "Waiting for target...",
                            (20, start_y + WINDOW_SIZE[1] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # 8. 拼接并显示最终结果
        # <--- 修改: 使用预先计算好的尺寸来缩放，而不是在循环中重复计算
        resized_main_frame = cv2.resize(annotated_frame, (resized_main_width, final_height))
        final_display = cv2.hconcat([resized_main_frame, window_canvas])

        # <--- 新增: 将最终画面写入视频文件
        out.write(final_display)

        # 实时显示（可以保留，也可以注释掉以加快处理速度）
        # cv2.imshow("YOLOv8 Multi-Target Tracking Demo", final_display)

        # 9. 按 'q' 键退出循环 (在没有实时显示时，这个逻辑可以移除或保留用于中断处理)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 10. 释放资源
    print("处理完成，正在释放资源...")
    cap.release()
    out.release() # <--- 新增: 释放VideoWriter对象
    cv2.destroyAllWindows()
    print(f"视频已成功保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
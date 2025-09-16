import cv2
import numpy as np
from ultralytics import YOLO
import random
import collections

# --- 配置参数 ---
VIDEO_PATH = './TestVideo.mp4'  # 你的视频文件路径
OUTPUT_PATH = './tracking_output_periodic.mp4'  # 输出视频文件的路径
MODEL_PATH = './yolov8s.pt'  # YOLOv8模型文件
MAX_TARGETS = 8  # 右侧最多显示几个跟踪窗口
WINDOW_SIZE = (384, 384)  # 右侧每个跟踪窗口的大小 (宽, 高)
CONF_THRESHOLD = 0.1  # 【关键修改】降低置信度阈值，以便检测到更多可能的目标
CLASSES_TO_TRACK = [0, 2, 3, 5, 7]  # 要跟踪的类别ID (0: person, 2: car, 3: motorcycle, 5: bus, 7: truck)
UPDATE_INTERVAL_SECONDS = 3  # 全局检测和目标分配的间隔时间（秒）
MAX_DISPLAY_HEIGHT = 960  # 实时预览窗口的最大高度，防止画面超出屏幕


def generate_colors(n):
    """为不同ID生成随机但鲜明的颜色"""
    return [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for _ in range(n)]


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

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算全局更新所需的帧数间隔
    update_interval_frames = int(fps * UPDATE_INTERVAL_SECONDS)
    print(f"视频帧率: {fps:.2f} FPS. 每 {update_interval_frames} 帧进行一次全局目标更新。")

    # --- 视频写入器初始化 ---
    final_height = WINDOW_SIZE[1] * MAX_TARGETS
    scale = final_height / frame_height
    resized_main_width = int(frame_width * scale)
    final_width = resized_main_width + WINDOW_SIZE[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (final_width, final_height))
    print(f"视频处理结果将保存到: {OUTPUT_PATH}")

    # --- 跟踪逻辑核心变量 ---
    frame_counter = 0
    # window_assignments 字典用于锁定每个窗口跟踪的目标ID
    # 格式: {window_index: track_id}, 例如 {0: 15, 1: 22}
    window_assignments = {}
    # all_targets_info 字典存储所有当前帧中有效目标的信息
    # 格式: {track_id: {"bbox": (x,y,w,h), "color": (B,G,R)}}
    all_targets_info = {}

    colors = generate_colors(100)  # 预先生成100个颜色备用

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束或读取错误。")
            break

        frame_counter += 1

        # 5. 使用YOLOv8进行跟踪 (这一步在每一帧都会执行，以获取所有目标的最新位置)
        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, classes=CLASSES_TO_TRACK, verbose=False,
                              tracker='bytetrack.yaml')

        # 从结果中提取当前帧所有被跟踪到的目标信息
        current_frame_boxes = results[0].boxes
        current_tracked_ids = []

        # 清空上一帧的目标信息，准备用当前帧的数据填充
        all_targets_info.clear()

        if current_frame_boxes.id is not None:
            boxes_xywh = current_frame_boxes.xywh.cpu().numpy()
            track_ids = current_frame_boxes.id.int().cpu().tolist()

            current_tracked_ids = track_ids  # 记录当前帧所有可见目标的ID

            for box, track_id in zip(boxes_xywh, track_ids):
                # 无论是否被窗口锁定，都更新其最新信息
                all_targets_info[track_id] = {
                    "bbox": box,
                    "color": colors[track_id % len(colors)]
                }

        # 6. 【核心逻辑】判断是否到达全局更新的时刻
        if frame_counter % update_interval_frames == 1:
            print(f"--- 第 {frame_counter} 帧: 执行全局目标检测与分配 ---")

            # 获取当前所有未被分配的目标ID
            assigned_ids = set(window_assignments.values())
            available_ids = set(current_tracked_ids) - assigned_ids

            # 为空闲的窗口分配新的目标
            for i in range(MAX_TARGETS):
                if i not in window_assignments and available_ids:
                    new_id_to_assign = sorted(list(available_ids))[0]  # 简单地分配第一个可用的ID
                    window_assignments[i] = new_id_to_assign
                    available_ids.remove(new_id_to_assign)
                    print(f"窗口 {i} 分配到新目标 ID: {new_id_to_assign}")

        # 7. 检查已分配的目标是否已丢失
        lost_windows = []
        for window_idx, track_id in window_assignments.items():
            if track_id not in current_tracked_ids:
                lost_windows.append(window_idx)

        # 如果目标丢失，则释放对应的窗口
        for window_idx in lost_windows:
            lost_id = window_assignments.pop(window_idx)
            print(f"目标 ID {lost_id} 已丢失，释放窗口 {window_idx}")

        # 8. 绘制主画面和右侧跟踪窗口
        annotated_frame = frame.copy()
        window_canvas = np.zeros((WINDOW_SIZE[1] * MAX_TARGETS, WINDOW_SIZE[0], 3), dtype=np.uint8)

        # 绘制主画面中的所有检测框
        for track_id, info in all_targets_info.items():
            x, y, w, h = info["bbox"]
            color = info["color"]
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 填充右侧的跟踪窗口
        for i in range(MAX_TARGETS):
            start_y = i * WINDOW_SIZE[1]
            end_y = start_y + WINDOW_SIZE[1]

            if i in window_assignments:
                track_id = window_assignments[i]
                # 确认目标信息仍然存在 (理论上一定存在，除非刚丢失)
                if track_id in all_targets_info:
                    target_info = all_targets_info[track_id]
                    x, y, w, h = target_info["bbox"]
                    color = target_info["color"]

                    # 从原图中裁剪目标区域
                    crop_w, crop_h = WINDOW_SIZE
                    x_center, y_center = int(x), int(y)
                    crop_x1 = max(0, x_center - crop_w // 2)
                    crop_y1 = max(0, y_center - crop_h // 2)
                    crop_x2 = min(frame_width, x_center + crop_w // 2)
                    crop_y2 = min(frame_height, y_center + crop_h // 2)

                    target_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    # 粘贴到对应窗口画布上
                    h_crop, w_crop, _ = target_crop.shape
                    padded_crop = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                    padded_crop[0:h_crop, 0:w_crop] = target_crop

                    # 在窗口上绘制边框和ID
                    cv2.rectangle(padded_crop, (0, 0), (crop_w - 1, crop_h - 1), color, 3)
                    cv2.putText(padded_crop, f"Locking ID: {track_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,
                                2)

                    window_canvas[start_y:end_y, :] = padded_crop
                else:  # 这种情况发生在目标刚刚丢失的那一帧
                    cv2.putText(window_canvas, "Target Lost", (50, start_y + WINDOW_SIZE[1] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            else:
                cv2.putText(window_canvas, "Waiting for target...", (20, start_y + WINDOW_SIZE[1] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # 9. 拼接并写入最终结果
        resized_main_frame = cv2.resize(annotated_frame, (resized_main_width, final_height))
        final_display = cv2.hconcat([resized_main_frame, window_canvas])
        out.write(final_display)

        # 实时显示处理画面 (修改部分)
        # 为了实时预览，将最终画面缩放到适合屏幕的大小
        display_scale = MAX_DISPLAY_HEIGHT / final_display.shape[0]
        if display_scale < 1:
            display_width = int(final_display.shape[1] * display_scale)
            display_frame_for_show = cv2.resize(final_display, (display_width, MAX_DISPLAY_HEIGHT))
        else:
            display_frame_for_show = final_display

        cv2.imshow("YOLOv8 Periodic Tracking Demo", display_frame_for_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户按下 'q' 键，提前退出。")
            break

    # 10. 释放资源
    print("处理完成，正在释放资源...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已成功保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


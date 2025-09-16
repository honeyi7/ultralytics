import os
import sys
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def main():
    # -------------------------------------------------------------------
    # 1. 路径和环境配置 (Paths and Environment Configuration)
    # -------------------------------------------------------------------

    # --- 指定数据集存放的根目录 (AutoDL数据盘) ---
    dataset_root_dir = '/root/autodl-tmp/yolov8/ultralytics/datasets'

    # 更新ultralytics的全局设置，让所有数据集都下载到这里
    SETTINGS.update({'datasets_dir': dataset_root_dir})

    # 确保目录存在
    os.makedirs(dataset_root_dir, exist_ok=True)

    print(f"✅ 数据集将统一存放在: {SETTINGS.get('datasets_dir')}")

    # --- 指定本地模型权重的路径 ---
    # 脚本会查找与自己同目录下的 'yolov8n.pt' 文件
    local_weights_path = 'yolov8s.pt'

    # 检查权重文件是否存在
    if not os.path.exists(local_weights_path):
        print(f"错误：权重文件 '{local_weights_path}' 不存在！")
        print("请先下载权重文件，并将其与此脚本放在同一个目录下。")
        print("下载命令: wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt")
        sys.exit(1)  # 退出脚本

    print(f"✅ 准备从本地加载权重: {local_weights_path}")

    # -------------------------------------------------------------------
    # 2. 训练参数配置 (Training Parameter Configuration)
    # -------------------------------------------------------------------

    # 指定要使用的GPU
    gpu_devices = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices

    # 要使用的数据集配置文件 (YOLOv8会自动下载到上面指定的`dataset_root_dir`)
    dataset_yaml = 'coco.yaml'  # 使用完整的COCO数据集

    # 训练参数
    epochs = 10  # 训练10个epoch
    batch_size = 128  # 总的batch size
    img_size = 640  # 图像尺寸

    print("\n============================================================")
    print(f"开始YOLOv8训练任务...")
    print(f"使用本地权重: {local_weights_path}")
    print(f"使用的数据集: {dataset_yaml}")
    print(f"计划使用的GPU: {gpu_devices}")
    print("============================================================")

    # -------------------------------------------------------------------
    # 3. 加载本地模型并开始训练 (Load Local Model and Start Training)
    # -------------------------------------------------------------------
    try:
        # --- 关键步骤：从本地 .pt 文件初始化模型 ---
        model = YOLO(local_weights_path)
        print("✅ 本地模型加载成功！")

        print("\n开始训练... YOLOv8将自动下载并准备数据集（如果尚未存在）。")

        # 开始训练
        model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=[int(g) for g in gpu_devices.split(',')]
        )

        print("\n============================================================")
        print("🎉 训练任务已成功完成！")
        print(f"请检查 '{dataset_root_dir}' 目录下是否已生成 'coco' 数据集文件夹。")
        print("同时，训练结果（包括权重）会保存在 'runs/detect/train' 目录下。")
        print("============================================================")

    except Exception as e:
        print(f"\n❌ 错误：训练过程中发生异常！")
        print(f"详细错误: {e}")


if __name__ == '__main__':
    main()
import os
import torch
import torch.nn as nn
import torchvision.models as models
from time import sleep, time


def run_stress_test():
    # -------------------------------------------------------------------
    # 1. 配置和检查环境 (Setup & Sanity Checks)
    # -------------------------------------------------------------------

    # 假设使用0号和1号GPU
    gpu_ids_str = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    try:
        device_ids = [int(id) for id in gpu_ids_str.split(',')]
    except ValueError:
        print(f"错误: GPU ID '{gpu_ids_str}' 格式不正确。")
        return

    if not torch.cuda.is_available():
        print("错误: PyTorch 未检测到可用的CUDA环境。")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus < len(device_ids):
        print(f"错误: 检测到的GPU数量 ({num_gpus}) 少于您指定的数量 ({len(device_ids)})。")
        return

    print("===================== 4090 多卡压力测试 =====================")
    print(f"✅ PyTorch CUDA 可用, 检测到 {num_gpus} 个GPU。")
    print(f"✅ 本次测试将使用 GPU: {device_ids}")
    for i in device_ids:
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print("==============================================================")

    # -------------------------------------------------------------------
    # 2. 创建一个更复杂的模型 (Instantiate a Complex Model)
    # -------------------------------------------------------------------

    # 使用经典的ResNet-50模型，它比之前的简单CNN复杂得多
    # weights=None 确保不会触发任何网络下载
    print("\n正在创建 ResNet-50 模型...")
    model = models.resnet50(weights=None)

    # 将模型主体先放到主GPU上 (例如: device 0)
    primary_device = f'cuda:{device_ids[0]}'
    model.to(primary_device)

    # --- 核心步骤: 使用DataParallel将模型封装以支持多卡 ---
    model = nn.DataParallel(model, device_ids=device_ids)
    print("✅ 模型已成功使用 torch.nn.DataParallel 进行封装。")

    # -------------------------------------------------------------------
    # 3. 模拟高负载训练 (Simulate High-Load Training)
    # -------------------------------------------------------------------

    # 针对24G显存，我们可以设置非常大的Batch Size来增加负载
    # 例如，每个GPU处理256个样本
    batch_size_per_gpu = 256
    total_batch_size = batch_size_per_gpu * len(device_ids)
    img_size = 224  # ResNet标准输入尺寸
    num_test_steps = 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("\n🚀 开始模拟高负载训练...")
    print(f"   - 模型: ResNet-50")
    print(f"   - 图像尺寸: {img_size}x{img_size}")
    print(f"   - 每个GPU的批量: {batch_size_per_gpu}")
    print(f"   - 总批量大小: {total_batch_size}")
    print("\n🔥 请立即新开一个终端，并运行 `watch -n 1 nvidia-smi` 来持续监控GPU状态。")
    print("   您应该会看到两张4090的功耗(Power)、显存(Memory)和利用率(GPU-Util)都显著上升。")
    sleep(5)

    # 预热一次CUDA，让计时更准确
    print("\n正在进行CUDA预热...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    print("预热完成，测试开始！")

    start_time = time()
    for i in range(num_test_steps):
        iter_start_time = time()

        # 动态创建足够大的假的图像和标签，并放到主GPU上
        fake_images = torch.randn(total_batch_size, 3, img_size, img_size, device=primary_device)
        fake_labels = torch.randint(0, 1000, (total_batch_size,), device=primary_device)

        # 前向传播 (DataParallel自动切分数据到所有GPU)
        outputs = model(fake_images)

        # 计算损失
        loss = criterion(outputs, fake_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 等待所有GPU核心完成当前批次计算
        torch.cuda.synchronize()
        iter_end_time = time()

        print(f"  [测试步骤 {i + 1:02d}/{num_test_steps}] 损失值: {loss.item():.4f} | "
              f"耗时: {(iter_end_time - iter_start_time) * 1000:.2f} ms")

    end_time = time()
    total_time = end_time - start_time
    avg_time_per_step = total_time / num_test_steps

    print("\n==============================================================")
    print("🎉 压力测试完成！")
    print(f"总计 {num_test_steps} 个模拟步骤耗时 {total_time:.2f} 秒。")
    print(f"平均每步耗时: {avg_time_per_step * 1000:.2f} ms。")
    print("\n[最终确认]:")
    print("  - 如果在测试中没有报错。")
    print("  - 并且您在 `nvidia-smi` 中观察到两张4090的功耗、显存和利用率都达到了较高水平。")
    print("  ==> 那么您的双4090多卡环境配置非常成功，且性能正常！")
    print("==============================================================")


if __name__ == '__main__':
    # 需要torchvision包，如果没安装，提示用户安装
    try:
        import torchvision
    except ImportError:
        print("错误: 未找到 torchvision 包。请先安装: pip install torchvision")
    else:
        run_stress_test()
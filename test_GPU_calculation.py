import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("GPU 数量:", torch.cuda.device_count())
print("当前 GPU:", torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    # 创建两个大矩阵
    a = torch.randn(10000, 10000).cuda()  # 移到 GPU
    b = torch.randn(10000, 10000).cuda()
    
    print("开始矩阵乘法（将在 GPU 上运行）...")
    c = torch.mm(a, b)  # 矩阵乘法
    print("计算完成！结果形状:", c.shape)
else:
    print("❌ CUDA 不可用！")
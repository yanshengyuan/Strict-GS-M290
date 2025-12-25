import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from matplotlib.font_manager import FontProperties
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from M290_MachineSimu_GPU.M290 import M290
from M290_MachineSimu_GPU.complex_field_tools_GPU.complex_field_tools import *

# --------------------- 用户手动设置 ---------------------
beamshape = 'ring'
plane = 'foc'
sample_id = 1   # 设置为 1 或 2
vis_dir = 'M290_GS_algorithm_results/'
os.makedirs(vis_dir, exist_ok=True)

# --------------------- 中文字体 ---------------------
font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc", size=14)

# --------------------- 设备 ---------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------- 载入单个样本 ---------------------
phase_path = f'exmp_data/{beamshape}_phase_{plane}_{sample_id}.npy'
intensity_path = f'exmp_data/{beamshape}_intensity_{plane}_{sample_id}.npy'

phase_data = np.load(phase_path)
intensity_data = np.load(intensity_path)

# 保存初始数据
plt.imsave(vis_dir+f'phase_start_{sample_id}.png', phase_data, cmap='gray')
plt.imsave(vis_dir+f'beamshape_{sample_id}.png', intensity_data, cmap='gray')

# 转为 torch tensor，batchsize=1
phase = torch.from_numpy(phase_data).unsqueeze(0).to(device)
beamshape_tensor = torch.from_numpy(intensity_data).unsqueeze(0).to(device)
batchsize = 1

# --------------------- 初始化机器 ---------------------
machine = M290('lightsource.npy', device, plane)
lightsource = machine.lightsource
init_field = torch.stack([machine.initField]*batchsize, dim=0)

# 近场初始化
near_field = SubIntensity(init_field, torch.stack([lightsource]*batchsize, dim=0))
near_field = SubPhase(near_field, phase)

# --------------------- GS 循环 ---------------------
iterations = 150
field = init_field.clone()
start = time.perf_counter()

for i in range(iterations):

    # 1. 源平面幅度约束
    field = SubIntensity(field, torch.stack([lightsource]*batchsize, dim=0))

    # 2. 正向传播
    field, size, curvature = machine.M290_forward(field)

    # 缓存正向传播前的像平面强度
    img_intensity_before_replace = Intensity(field, flag=2).cpu().numpy()

    # 3. 图像平面幅度约束
    field = SubIntensity(field, beamshape_tensor)

    # 4. 逆向传播
    field = machine.M290_inverse(field, size, curvature)

    # 缓存逆向传播前的源平面强度
    src_intensity_before_replace = Intensity(field, flag=2).cpu().numpy()

    # --------------------- 2x3 中文可视化 ---------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 第一行
    axes[0, 0].imshow(Phase(field)[0].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("恢复源平面相位（当前迭代）", fontproperties=font)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(src_intensity_before_replace[0], cmap='gray')
    axes[0, 1].set_title("源平面强度（逆向传播）", fontproperties=font)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_intensity_before_replace[0], cmap='gray')
    axes[0, 2].set_title("像平面强度（正向传播）", fontproperties=font)
    axes[0, 2].axis("off")

    # 第二行
    axes[1, 0].imshow(phase[0].cpu().numpy(), cmap='gray')
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, -0.12, "真实源平面相位", transform=axes[1, 0].transAxes,
                    ha='center', va='top', fontproperties=font)

    axes[1, 1].imshow(lightsource.cpu().numpy(), cmap='gray')
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, -0.12, "真实源平面强度", transform=axes[1, 1].transAxes,
                    ha='center', va='top', fontproperties=font)

    axes[1, 2].imshow(beamshape_tensor[0].cpu().numpy(), cmap='gray')
    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, -0.12, "真实像平面强度", transform=axes[1, 2].transAxes,
                    ha='center', va='top', fontproperties=font)

    plt.tight_layout()
    plt.show()
    plt.close()

    # 保存最后一轮结果
    if i == iterations-1:
        plt.imsave(vis_dir+f"GS_phase_{sample_id}.png", Phase(field)[0].cpu().numpy(), cmap='gray')
        plt.imsave(vis_dir+f"GS_intensity_{sample_id}.png", Intensity(field, flag=2)[0].cpu().numpy(), cmap='gray')

    print(f"迭代: {i+1}/{iterations}")

end = time.perf_counter()
print(f"Runtime: {end - start:.3f} s")

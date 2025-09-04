import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(file_path):
    # 初始化存储数据的字典
    data = {
        'epoch': [],
        'best_l2_error': [],
        'x_disp_loss': [],
        'y_disp_loss': [],
        'hole_disp_loss': [],
        'free_bc_loss': [],
        'x_pde_loss': [],
        'y_pde_loss': []
    }

    epoch_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # 匹配Best L2相对误差
            if line.startswith('Best L2 relative error:'):
                value = float(line.split(':')[-1].strip())
                data['best_l2_error'].append(value)
                data['epoch'].append(epoch_count)
                epoch_count += 1

            # 匹配各种损失函数
            elif line.startswith('x-direction prescribed displacement loss'):
                value = line.split(':')[-1].strip() if ':' in line else line.split('loss')[-1].strip()
                if value.lower() == 'inf':
                    data['x_disp_loss'].append(np.inf)
                else:
                    try:
                        data['x_disp_loss'].append(float(value))
                    except ValueError:
                        data['x_disp_loss'].append(np.nan)

            elif line.startswith('y-direction prescribed displacement loss:'):
                value = line.split(':')[-1].strip()
                if value.lower() == 'inf':
                    data['y_disp_loss'].append(np.inf)
                else:
                    try:
                        data['y_disp_loss'].append(float(value))
                    except ValueError:
                        data['y_disp_loss'].append(np.nan)

            elif line.startswith('hole prescribed displacement loss'):
                value = line.split(':')[-1].strip() if ':' in line else line.split('loss')[-1].strip()
                if value.lower() == 'inf':
                    data['hole_disp_loss'].append(np.inf)
                else:
                    try:
                        data['hole_disp_loss'].append(float(value))
                    except ValueError:
                        data['hole_disp_loss'].append(np.nan)

            elif line.startswith('free boundary condtion loss:'):
                value = line.split(':')[-1].strip()
                if value.lower() == 'inf':
                    data['free_bc_loss'].append(np.inf)
                else:
                    try:
                        data['free_bc_loss'].append(float(value))
                    except ValueError:
                        data['free_bc_loss'].append(np.nan)

            elif line.startswith('x-direction PDE residual loss:'):
                value = line.split(':')[-1].strip()
                if value.lower() == 'inf':
                    data['x_pde_loss'].append(np.inf)
                else:
                    try:
                        data['x_pde_loss'].append(float(value))
                    except ValueError:
                        data['x_pde_loss'].append(np.nan)

            elif line.startswith('y-direction PDE residual loss:'):
                value = line.split(':')[-1].strip()
                if value.lower() == 'inf':
                    data['y_pde_loss'].append(np.inf)
                else:
                    try:
                        data['y_pde_loss'].append(float(value))
                    except ValueError:
                        data['y_pde_loss'].append(np.nan)

    return data


def plot_losses(data):
    plt.figure(figsize=(15, 10))

    # L2相对误差
    plt.subplot(2, 2, 1)
    plt.plot(data['epoch'], data['best_l2_error'], 'b-o')
    plt.title('Best L2 Relative Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)

    # 位移相关损失
    plt.subplot(2, 2, 2)
    plt.plot(data['epoch'], data['x_disp_loss'], 'r-', label='X Displacement')
    plt.plot(data['epoch'], data['y_disp_loss'], 'g-', label='Y Displacement')
    plt.plot(data['epoch'], data['hole_disp_loss'], 'b-', label='Hole Displacement')
    plt.title('Prescribed Displacement Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # 边界条件损失
    plt.subplot(2, 2, 3)
    plt.plot(data['epoch'], data['free_bc_loss'], 'm-o')
    plt.title('Free Boundary Condition Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)

    # PDE残差损失
    plt.subplot(2, 2, 4)
    plt.plot(data['epoch'], data['x_pde_loss'], 'r-', label='X PDE Residual')
    plt.plot(data['epoch'], data['y_pde_loss'], 'b-', label='Y PDE Residual')
    plt.title('PDE Residual Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 使用示例
file_path = 'output-2118342.log'  # 替换为您的文件路径
data = parse_log_file(file_path)
plot_losses(data)
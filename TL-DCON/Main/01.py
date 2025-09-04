import re
import matplotlib.pyplot as plt
import numpy as np


def parse_losses_from_log(file_path):
    """
    从日志文件中解析损失值
    """
    # 定义正则表达式模式来匹配不同类型的损失
    patterns = {
        'x_disp_loss': r'x-direction prescribed displacement loss ([\d\.eE+-]+)',
        'y_disp_loss': r'y-direction prescribed displacement loss: ([\d\.eE+-]+)',
        'hole_disp_loss': r'hole prescribed displacement loss ([\d\.eE+-]+)',
        'free_bc_loss': r'free boundary condtion loss: ([\d\.eE+-]+)',
        'x_pde_loss': r'x-direction PDE residual loss: ([\d\.eE+-]+)',
        'y_pde_loss': r'y-direction PDE residual loss: ([\d\.eE+-]+)',
        'best_error': r'Best L2 relative error: ([\d\.eE+-]+)'
    }

    # 初始化存储数据的字典
    losses = {key: [] for key in patterns.keys()}
    epochs = []

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # 找到所有匹配的损失值
        for key, pattern in patterns.items():
            matches = re.findall(pattern, content)
            losses[key] = [float(match) for match in matches]

        # 确定epoch数量（以最长的损失列表为准）
        max_length = max(len(losses[key]) for key in losses)
        epochs = list(range(1, max_length + 1))

    return epochs, losses


def plot_losses(epochs, losses, save_path=None):
    """
    绘制损失函数曲线
    """
    plt.figure(figsize=(15, 12))

    # 1. 绘制所有损失函数
    plt.subplot(2, 2, 1)
    for key, values in losses.items():
        if key != 'best_error' and len(values) > 0:
            plt.plot(epochs[:len(values)], values, label=key, marker='o', markersize=3)

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Losses')
    plt.yscale('log')  # 使用对数坐标，因为损失值变化范围很大
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # 2. 绘制L2相对误差
    plt.subplot(2, 2, 2)
    if losses['best_error']:
        plt.plot(epochs[:len(losses['best_error'])], losses['best_error'],
                 label='Best L2 Relative Error', color='red', marker='s', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('L2 Relative Error')
        plt.title('L2 Relative Error Progression')
        plt.legend()
        plt.grid(True)

    # 3. 绘制位移相关损失
    plt.subplot(2, 2, 3)
    disp_losses = ['x_disp_loss', 'y_disp_loss', 'hole_disp_loss']
    colors = ['blue', 'green', 'orange']
    for loss_key, color in zip(disp_losses, colors):
        if losses[loss_key]:
            plt.plot(epochs[:len(losses[loss_key])], losses[loss_key],
                     label=loss_key, color=color, marker='^', markersize=3)

    plt.xlabel('Epoch')
    plt.ylabel('Displacement Loss')
    plt.title('Displacement-related Losses')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # 4. 绘制PDE和边界条件损失
    plt.subplot(2, 2, 4)
    pde_bc_losses = ['free_bc_loss', 'x_pde_loss', 'y_pde_loss']
    colors = ['purple', 'brown', 'pink']
    for loss_key, color in zip(pde_bc_losses, colors):
        if losses[loss_key]:
            plt.plot(epochs[:len(losses[loss_key])], losses[loss_key],
                     label=loss_key, color=color, marker='d', markersize=3)

    plt.xlabel('Epoch')
    plt.ylabel('PDE/BC Loss')
    plt.title('PDE and Boundary Condition Losses')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # 打印统计信息
    print("损失统计信息:")
    for key, values in losses.items():
        if values:
            print(f"{key}: 最小值={min(values):.2e}, 最大值={max(values):.2e}, 最后值={values[-1]:.2e}")


def main():
    # 文件路径 - 请根据实际情况修改
    log_file_path = "output-2118342.log"  # 替换为你的日志文件路径
    output_image_path = "training_losses.png"  # 输出图像路径

    try:
        # 解析日志文件
        epochs, losses = parse_losses_from_log(log_file_path)

        # 绘制损失曲线
        plot_losses(epochs, losses, output_image_path)

        print(f"可视化完成！图像已保存至: {output_image_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")


if __name__ == "__main__":
    main()
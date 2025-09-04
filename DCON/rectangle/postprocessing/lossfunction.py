import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_log_file(log_file_path):
    """
    解析日志文件，提取损失函数数据
    """
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 初始化数据存储
    epochs = []
    relative_errors = []
    loss_data = {
        'x_direction_prescribed': [],
        'y_direction_prescribed': [],
        'hole_prescribed': [],
        'free_boundary': [],
        'x_direction_pde': [],
        'y_direction_pde': []
    }

    # 解析相对误差数据
    error_pattern = r'Best L2 relative error: ([\d.]+)'
    errors = re.findall(error_pattern, content)
    relative_errors = [float(error) for error in errors]
    epochs = list(range(len(relative_errors)))

    # 解析各个损失函数数据
    loss_patterns = {
        'x_direction_prescribed': r'x-direction prescribed displacement loss ([\deE.+-]+)',
        'y_direction_prescribed': r'y-direction prescribed displacement loss: ([\deE.+-]+)',
        'hole_prescribed': r'hole prescribed displacement loss ([\deE.+-]+)',
        'free_boundary': r'free boundary condtion loss: ([\deE.+-]+)',
        'x_direction_pde': r'x-direction PDE residual loss: ([\deE.+-]+)',
        'y_direction_pde': r'y-direction PDE residual loss: ([\deE.+-]+)'
    }

    for loss_name, pattern in loss_patterns.items():
        matches = re.findall(pattern, content)
        # 处理科学计数法表示的数字
        values = []
        for match in matches:
            if 'inf' in match.lower():
                values.append(np.nan)  # 将inf替换为NaN
            else:
                try:
                    values.append(float(match))
                except ValueError:
                    # 处理科学计数法
                    values.append(float(match.replace('e', 'E')))
        loss_data[loss_name] = values

    return epochs, relative_errors, loss_data


def plot_losses(log_file_path, save_path=None):
    """
    绘制损失函数可视化图表
    """
    # 解析数据
    epochs, relative_errors, loss_data = parse_log_file(log_file_path)

    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Loss Visualization\n(Dataset: Dataset_Rectangle, Model: DCON)', fontsize=16,
                 fontweight='bold')

    # 绘制相对误差
    ax = axes[0, 0]
    ax.plot(epochs, relative_errors, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Relative L2 Error')
    ax.set_title('Relative L2 Error')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 绘制各个损失函数
    loss_names = [
        ('x_direction_prescribed', 'X-direction Prescribed', axes[0, 1]),
        ('y_direction_prescribed', 'Y-direction Prescribed', axes[0, 2]),
        ('hole_prescribed', 'Hole Prescribed', axes[1, 0]),
        ('free_boundary', 'Free Boundary', axes[1, 1]),
        ('x_direction_pde', 'X-direction PDE', axes[1, 2]),
        ('y_direction_pde', 'Y-direction PDE', None)
    ]

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    # 绘制前5个损失函数
    for i, (loss_key, title, ax) in enumerate(loss_names[:5]):
        if ax is not None:
            values = loss_data[loss_key]
            # 过滤掉NaN值（inf）
            valid_epochs = [epoch for epoch, val in zip(epochs, values) if not np.isnan(val)]
            valid_values = [val for val in values if not np.isnan(val)]

            ax.plot(valid_epochs, valid_values, color=colors[i], linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss Value')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

    # 在最后一个位置绘制所有损失函数的对比
    ax_all = axes[1, 2]
    for i, (loss_key, title, _) in enumerate(loss_names[2:]):  # 从第三个开始，避免重叠
        values = loss_data[loss_key]
        valid_epochs = [epoch for epoch, val in zip(epochs, values) if not np.isnan(val)]
        valid_values = [val for val in values if not np.isnan(val)]

        if len(valid_values) > 0:
            ax_all.plot(valid_epochs, valid_values, color=colors[i + 2], linewidth=2, label=title)

    ax_all.set_xlabel('Epochs')
    ax_all.set_ylabel('Loss Value')
    ax_all.set_title('Multiple Losses Comparison')
    ax_all.grid(True, alpha=0.3)
    ax_all.set_yscale('log')
    ax_all.legend()

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.show()

    return fig


def create_summary_table(log_file_path):
    """
    创建损失函数的统计摘要表格
    """
    epochs, relative_errors, loss_data = parse_log_file(log_file_path)

    summary_data = []
    for loss_name, values in loss_data.items():
        # 过滤掉NaN值
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            summary_data.append({
                'Loss Function': loss_name.replace('_', ' ').title(),
                'Min Value': f"{min(valid_values):.2e}",
                'Max Value': f"{max(valid_values):.2e}",
                'Final Value': f"{valid_values[-1]:.2e}",
                'Data Points': len(valid_values)
            })

    # 添加相对误差
    summary_data.append({
        'Loss Function': 'Relative L2 Error',
        'Min Value': f"{min(relative_errors):.4f}",
        'Max Value': f"{max(relative_errors):.4f}",
        'Final Value': f"{relative_errors[-1]:.4f}",
        'Data Points': len(relative_errors)
    })

    # 创建DataFrame
    df = pd.DataFrame(summary_data)
    print("损失函数统计摘要:")
    print(df.to_string(index=False))

    return df


# 使用示例
if __name__ == "__main__":
    log_file_path = "output-2238452.log"  # 替换为你的.log文件路径
    save_image_path = "training_loss_visualization.png"  # 保存图像路径

    try:
        # 绘制损失函数图表
        fig = plot_losses(log_file_path, save_image_path)

        # 创建统计摘要
        summary_df = create_summary_table(log_file_path)

        # 打印训练信息
        print("\n训练信息:")
        print(f"总训练周期: {len(re.findall(r'Best L2 relative error', open(log_file_path).read()))}")
        print(
            f"最终相对误差: {summary_df[summary_df['Loss Function'] == 'Relative L2 Error']['Final Value'].values[0]}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
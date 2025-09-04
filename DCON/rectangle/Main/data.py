import scipy.io as sio
import numpy as np
import torch
from scipy.interpolate import griddata      # 从模块中导入库，用于在非结构化网格上进行数据插值


def generate_plate_dataloader(config):
    # load the new data file
    mat_contents = sio.loadmat(r'../data/Dataset_Rectangle.mat')

    # extract data
    u = mat_contents['ux']  # (B, M) x方向位移
    v = mat_contents['uy']  # (B, M) y方向位移
    xx = mat_contents['xx']  # 坐标点x
    yy = mat_contents['yy']  # 坐标点y
    # 组合成 (M, 2) 的坐标矩阵
    coor = np.hstack((xx, yy))

    flag_BCxy = mat_contents['flag_BCxy']  # (M, 1) 左边界固定
    flag_BCy = mat_contents['flag_BCy']  # (M, 1) 上下边界
    flag_BC_load = mat_contents['flag_BC_load']  # (M, 1) 右边界节点索引（已存在）
    f_bc = mat_contents['f_bc']  # (B, 101) 边界条件离散点

    # 打印数据形状进行调试
    print(f"ux shape: {u.shape}")
    print(f"uy shape: {v.shape}")
    print(f"coor shape: {coor.shape}")
    print(f"flag_BCxy shape: {flag_BCxy.shape}")
    print(f"flag_BCy shape: {flag_BCy.shape}")
    print(f"flag_BC_load shape: {flag_BC_load.shape}")
    print(f"f_bc shape: {f_bc.shape}")

    # 材料参数
    youngs = 300e5 * 1e-4  # 应用相同的缩放因子
    nu = 0.3

    # 处理边界条件：将101个离散点插值到实际的右边界节点上
    num_samples = u.shape[0]

    # 获取右边界节点的坐标
    right_nodes_indices = np.where(flag_BC_load.flatten() == 1)[0]
    right_nodes_coords = coor[right_nodes_indices]
    right_nodes_y = right_nodes_coords[:, 1]
    num_right_nodes = len(right_nodes_indices)

    # 创建插值后的边界条件参数
    params_list = []
    for i in range(num_samples):
        # 原始101个离散点的坐标（假设在[0,1]区间均匀分布）
        bc_y_coords = np.linspace(0, 1, 101)
        bc_values = f_bc[i]  # 第i个样本的边界条件值

        # 插值到实际的右边界节点
        interp_values = griddata(bc_y_coords, bc_values, right_nodes_y,
                                 method='linear', fill_value=0.0)

        # 构建参数矩阵: [x, y, u] for each boundary node
        sample_params = np.column_stack([
            right_nodes_coords[:, 0],  # x坐标
            right_nodes_coords[:, 1],  # y坐标
            interp_values  # 插值后的边界条件值
        ])
        params_list.append(sample_params)

    # 转换为numpy数组
    params = np.array(params_list)  # (B, num_right_nodes, 3)
    num_bc_nodes = params.shape[1]

    # 转换为torch tensor
    params = torch.tensor(params, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    coors = torch.tensor(coor, dtype=torch.float32)
    flag_BCxy = torch.tensor(flag_BCxy, dtype=torch.float32)
    flag_BCy = torch.tensor(flag_BCy, dtype=torch.float32)
    flag_BC_load = torch.tensor(flag_BC_load, dtype=torch.float32)

    # 划分数据集
    datasize = u.shape[0]
    bar1 = [0, int(0.7 * datasize)]
    bar2 = [int(0.7 * datasize), int(0.8 * datasize)]
    bar3 = [int(0.8 * datasize), datasize]

    train_dataset = torch.utils.data.TensorDataset(
        params[bar1[0]:bar1[1]],
        u[bar1[0]:bar1[1]],
        v[bar1[0]:bar1[1]]
    )
    val_dataset = torch.utils.data.TensorDataset(
        params[bar2[0]:bar2[1]],
        u[bar2[0]:bar2[1]],
        v[bar2[0]:bar2[1]]
    )
    test_dataset = torch.utils.data.TensorDataset(
        params[bar3[0]:bar3[1]],
        u[bar3[0]:bar3[1]],
        v[bar3[0]:bar3[1]]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['train']['batchsize'],
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['train']['batchsize'],
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=2,
                                              shuffle=False)

    return coors, train_loader, val_loader, test_loader, youngs, nu, num_bc_nodes, flag_BCxy, flag_BCy, flag_BC_load


def create_predict_data(config, coors, flag_BC_load, load_value=5.0):
    """
    创建预测用的边界条件数据
    """
    # 获取右边界节点的坐标
    right_nodes_indices = np.where(flag_BC_load.numpy().flatten() == 1)[0]
    right_nodes_coords = coors.numpy()[right_nodes_indices]
    num_right_nodes = len(right_nodes_indices)

    print(f"Number of right boundary nodes for prediction: {num_right_nodes}")

    # 创建新的边界条件参数：所有右边界节点荷载为指定值
    new_params = np.column_stack([
        right_nodes_coords[:, 0],  # x坐标
        right_nodes_coords[:, 1],  # y坐标
        np.full(num_right_nodes, load_value)  # 所有节点荷载
    ])

    # 扩展为批量形式 (1, num_right_nodes, 3)
    new_params = np.expand_dims(new_params, axis=0)
    new_params = torch.tensor(new_params, dtype=torch.float32)

    # 创建对应的真实位移（用零填充，因为我们是预测）
    dummy_u = torch.zeros(1, coors.shape[0], dtype=torch.float32)
    dummy_v = torch.zeros(1, coors.shape[0], dtype=torch.float32)

    # 创建数据加载器
    predict_dataset = torch.utils.data.TensorDataset(new_params, dummy_u, dummy_v)
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1, shuffle=False)

    return predict_loader, new_params.shape[1]  # 返回loader和边界节点数量
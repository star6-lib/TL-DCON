import scipy.io as sio
import numpy as np
import torch
import h5py

# Define the function for loading the darcy problem dataset
def generate_darcy_dataloader(config):
    '''
    Input:
        config: providing the training configuration
    Output:
        coors: a set of fixed coordinates for PDE predictions    (M, 2)
        data loaders for training, validation and testing
            - each batch consists of (boundary condition values, PDE solution values)
            - boundary condition values shape: (B, M', 3)
            - PDE solution values shape: (B, M)
    '''

    # load the data
    mat_contents = sio.loadmat(r'../data/Darcy_star.mat')
    f_bc = mat_contents['BC_input_var']  # (K, N, 3)
    u = mat_contents['u_field']  # (K, M)
    coor = mat_contents['coor']  # (M, 2)
    BC_flags = mat_contents['IC_flag'].T  # (M, 1)
    num_bc_nodes = f_bc.shape[1]
    print('raw data shape check:', f_bc.shape, u.shape, coor.shape, BC_flags.shape)
    print('number of nodes on boundary:', num_bc_nodes)

    # define dataset
    fbc = torch.tensor(f_bc)  # (K, N, 3)
    sol = torch.tensor(u)  # (K, M)
    coors = torch.tensor(coor)  # (M,2)
    datasize = fbc.shape[0]

    # define data loaders
    bar1 = [0, int(0.7 * datasize)]
    bar2 = [int(0.7 * datasize), int(0.8 * datasize)]
    bar3 = [int(0.8 * datasize), int(datasize)]
    train_dataset = torch.utils.data.TensorDataset(fbc[bar1[0]:bar1[1], :, :], sol[bar1[0]:bar1[1], :])
    val_dataset = torch.utils.data.TensorDataset(fbc[bar2[0]:bar2[1], :, :], sol[bar2[0]:bar2[1], :])
    test_dataset = torch.utils.data.TensorDataset(fbc[bar3[0]:bar3[1], :, :], sol[bar3[0]:bar3[1], :])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

    return coors, BC_flags, num_bc_nodes, train_loader, val_loader, test_loader


def load_mat_v73(file_path):
    """专门用于加载MATLAB v7.3格式的.mat文件"""
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            # 获取数据
            dataset = f[key]
            # 转换数据类型
            if isinstance(dataset, h5py.Dataset):
                # 对于数值数据，直接转换为numpy数组并转置（MATLAB是列优先）
                data_array = np.array(dataset)
                if len(data_array.shape) > 1:
                    data_array = data_array.T  # 转置以匹配MATLAB的存储顺序
                data[key] = data_array
            else:
                print(f"跳过非数据集对象: {key}")

    return data


def get_data(mat_contents, key, expected_shape=None):
    """从mat_contents中获取数据并处理形状"""
    if key in mat_contents:
        data = mat_contents[key]
        print(f"{key} 形状: {data.shape}")

        # 处理转置问题：MATLAB数据在h5py中需要转置
        if len(data.shape) == 2 and data.shape[0] == 1:
            # 如果是行向量，转换为列向量
            data = data.reshape(-1, 1)
        elif len(data.shape) == 2 and data.shape[1] == 1:
            # 如果是列向量，保持原状
            pass
        elif len(data.shape) == 2:
            # 对于2D矩阵，检查是否需要转置
            if expected_shape and data.shape != expected_shape:
                # 如果形状不匹配，尝试转置
                data = data.T
                print(f"转置 {key} 为形状: {data.shape}")

        return data
    else:
        print(f"警告: 找不到变量 {key}")
        return None


# Define the dataset for plate problem with Dataset_1Circle.mat
def generate_plate_dataloader(config):
    # load the data from Dataset_1Circle.mat using h5py for v7.3 format
    try:
        mat_contents = load_mat_v73(r'../data/Dataset_1Circle.mat')
        print("成功使用h5py加载v7.3格式的mat文件")
    except Exception as e:
        print(f"无法使用h5py加载mat文件: {e}")
        return None

    # 打印所有可用的变量名以便调试
    print("可用变量:", list(mat_contents.keys()))

    # 获取数据并正确处理形状
    ux_train = get_data(mat_contents, 'ux_train')  # 应该是 (1900, n_nodes)
    uy_train = get_data(mat_contents, 'uy_train')  # 应该是 (1900, n_nodes)
    ux_test = get_data(mat_contents, 'ux_test')  # 应该是 (100, n_nodes)
    uy_test = get_data(mat_contents, 'uy_test')  # 应该是 (100, n_nodes)
    f_bc_train = get_data(mat_contents, 'f_bc_train')  # 应该是 (1900, 101)
    f_bc_test = get_data(mat_contents, 'f_bc_test')  # 应该是 (100, 101)
    coor_x = get_data(mat_contents, 'xx')  # 应该是 (n_nodes, 1)
    coor_y = get_data(mat_contents, 'yy')  # 应该是 (n_nodes, 1)

    # 提取边界标志
    flag_BCxy = get_data(mat_contents, 'flag_BCxy')  # 应该是 (n_nodes, 1)
    flag_BCy = get_data(mat_contents, 'flag_BCy')  # 应该是 (n_nodes, 1)
    flag_load = get_data(mat_contents, 'flag_BC_load')  # 应该是 (n_nodes, 1)

    # 检查并修正数据形状
    if ux_train is not None and ux_train.shape[0] == 1048 and ux_train.shape[1] == 1900:
        print("检测到需要转置ux数据")
        ux_train = ux_train.T
        uy_train = uy_train.T
        ux_test = ux_test.T
        uy_test = uy_test.T
        print(f"转置后: ux_train={ux_train.shape}, uy_train={uy_train.shape}")

    if coor_x is not None and coor_x.shape[0] == 1 and coor_x.shape[1] == 2096:
        print("检测到需要重塑坐标数据")
        coor_x = coor_x.reshape(-1, 1)
        coor_y = coor_y.reshape(-1, 1)
        print(f"重塑后: coor_x={coor_x.shape, coor_y.shape}")

    # 检查数据是否成功加载
    if any(v is None for v in [ux_train, uy_train, coor_x, coor_y]):
        print("错误: 缺少必要的数据变量")
        return None

    # 确定节点数量
    n_nodes = coor_x.shape[0]
    n_train_samples = ux_train.shape[0]
    n_test_samples = ux_test.shape[0]

    # Combine coordinates
    coor = np.concatenate((coor_x, coor_y), axis=1)  # (n_nodes, 2)

    # Load material properties
    youngs = 300.0e5 * 1e-4
    nu = 0.3

    # 获取加载边界节点的索引
    if flag_load is not None:
        # 确保flag_load是1D数组
        if len(flag_load.shape) > 1:
            flag_load = flag_load.flatten()
        load_node_indices = np.where(flag_load == 1)[0]
        num_bc_nodes = len(load_node_indices)
        print(f"找到 {num_bc_nodes} 个加载边界节点")
    else:
        print("警告: 没有找到flag_load，将使用默认边界节点")
        # 如果没有边界标志，假设使用前101个节点作为边界
        num_bc_nodes = min(101, n_nodes)
        load_node_indices = np.arange(num_bc_nodes)

    print(f"数据集信息:")
    print(f"  训练样本数: {n_train_samples}")
    print(f"  测试样本数: {n_test_samples}")
    print(f"  总节点数: {n_nodes}")
    print(f"  边界节点数: {num_bc_nodes}")
    print(f"  坐标形状: {coor.shape}")
    print(f"  ux_train形状: {ux_train.shape}")
    print(f"  f_bc_train形状: {f_bc_train.shape if f_bc_train is not None else 'None'}")

    # 构建参数输入
    params_train = []
    for i in range(n_train_samples):
        # 获取加载边界节点的坐标
        bc_coords = coor[load_node_indices]

        # 获取对应的f_bc值
        if f_bc_train is not None and i < f_bc_train.shape[0]:
            bc_values = f_bc_train[i]
            # 确保bc_values的长度与边界节点数匹配
            if len(bc_values) != len(load_node_indices):
                print(f"警告: f_bc长度{len(bc_values)}与边界节点数{len(load_node_indices)}不匹配")
                # 使用插值或截断
                min_len = min(len(bc_values), len(load_node_indices))
                bc_values = bc_values[:min_len]
                bc_coords = bc_coords[:min_len]

            param_sample = np.concatenate([bc_coords, bc_values.reshape(-1, 1)], axis=1)
            params_train.append(param_sample)
        else:
            # 如果没有f_bc数据，使用零值
            bc_values = np.zeros(len(load_node_indices))
            param_sample = np.concatenate([bc_coords, bc_values.reshape(-1, 1)], axis=1)
            params_train.append(param_sample)

    params_train = np.array(params_train)

    # 构建测试集参数
    params_test = []
    for i in range(n_test_samples):
        bc_coords = coor[load_node_indices]
        if f_bc_test is not None and i < f_bc_test.shape[0]:
            bc_values = f_bc_test[i]
            if len(bc_values) != len(load_node_indices):
                min_len = min(len(bc_values), len(load_node_indices))
                bc_values = bc_values[:min_len]
                bc_coords = bc_coords[:min_len]
            param_sample = np.concatenate([bc_coords, bc_values.reshape(-1, 1)], axis=1)
            params_test.append(param_sample)
        else:
            bc_values = np.zeros(len(load_node_indices))
            param_sample = np.concatenate([bc_coords, bc_values.reshape(-1, 1)], axis=1)
            params_test.append(param_sample)

    params_test = np.array(params_test)

    print(f"参数形状: train={params_train.shape}, test={params_test.shape}")

    # 定义数据集为torch.tensor
    params_train = torch.tensor(params_train, dtype=torch.float32)
    params_test = torch.tensor(params_test, dtype=torch.float32)
    u_train = torch.tensor(ux_train, dtype=torch.float32)
    v_train = torch.tensor(uy_train, dtype=torch.float32)
    u_test = torch.tensor(ux_test, dtype=torch.float32)
    v_test = torch.tensor(uy_test, dtype=torch.float32)
    coors = torch.tensor(coor, dtype=torch.float32)

    # 转换边界标志
    if flag_BCxy is not None:
        flag_BCxy = flag_BCxy.flatten()
    else:
        flag_BCxy = np.zeros(n_nodes)
        print("警告: 使用默认flag_BCxy")

    if flag_BCy is not None:
        flag_BCy = flag_BCy.flatten()
    else:
        flag_BCy = np.zeros(n_nodes)
        print("警告: 使用默认flag_BCy")

    if flag_load is not None:
        flag_load = flag_load.flatten()
    else:
        flag_load = np.zeros(n_nodes)
        print("警告: 使用默认flag_load")

    # 定义数据加载器
    train_dataset = torch.utils.data.TensorDataset(params_train, u_train, v_train)

    # 分割测试集为验证集和测试集
    val_size = min(50, n_test_samples // 2)
    test_size = n_test_samples - val_size

    val_dataset = torch.utils.data.TensorDataset(
        params_test[:val_size], u_test[:val_size], v_test[:val_size]
    )
    test_dataset = torch.utils.data.TensorDataset(
        params_test[val_size:val_size + test_size], u_test[val_size:val_size + test_size],
        v_test[val_size:val_size + test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['batchsize'],
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['train']['batchsize'],
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False
    )

    return coors, train_loader, val_loader, test_loader, youngs, nu, num_bc_nodes, flag_BCxy, flag_BCy, flag_load


def generate_generalization_test_data(config):
    """Generate test data with uniform tension load on right boundary (100MPa)
    - Left boundary: fixed (u=0, v=0)
    - Right boundary: uniform tension 100MPa
    - Other boundaries (top/bottom/hole): free
    """
    print("正在生成均匀拉伸荷载测试数据 (100MPa)")

    # 加载原始数据获取网格信息
    try:
        mat_contents = load_mat_v73(r'../data/Dataset_1Circle.mat')
        print("成功使用h5py加载v7.3格式的mat文件")
    except Exception as e:
        print(f"无法使用h5py加载mat文件: {e}")
        return None

    # 获取坐标数据
    coor_x = get_data(mat_contents, 'xx')
    coor_y = get_data(mat_contents, 'yy')
    if coor_x is None or coor_y is None:
        print("错误: 无法获取坐标数据")
        return None

    # 确保坐标形状正确
    if coor_x.shape[0] == 1 and coor_x.shape[1] > 1:
        coor_x = coor_x.reshape(-1, 1)
        coor_y = coor_y.reshape(-1, 1)

    coor = np.concatenate((coor_x, coor_y), axis=1)
    n_nodes = coor.shape[0]

    # 获取边界标志
    flag_BCxy = get_data(mat_contents, 'flag_BCxy')
    flag_BCy = get_data(mat_contents, 'flag_BCy')
    flag_load = get_data(mat_contents, 'flag_BC_load')

    # 确保边界标志形状正确
    if flag_BCxy is not None and len(flag_BCxy.shape) > 1:
        flag_BCxy = flag_BCxy.flatten()
    if flag_BCy is not None and len(flag_BCy.shape) > 1:
        flag_BCy = flag_BCy.flatten()
    if flag_load is not None and len(flag_load.shape) > 1:
        flag_load = flag_load.flatten()

    # 如果没有边界标志，基于坐标创建
    if flag_BCxy is None:
        print("警告: 使用坐标创建左边界标志")
        left_boundary = coor[:, 0] == coor[:, 0].min()
        flag_BCxy = left_boundary.astype(float)

    if flag_load is None:
        print("警告: 使用坐标创建右边界标志")
        right_boundary = coor[:, 0] == coor[:, 0].max()
        flag_load = right_boundary.astype(float)

    if flag_BCy is None:
        print("警告: 使用坐标创建其他边界标志")
        top_boundary = coor[:, 1] == coor[:, 1].max()
        bottom_boundary = coor[:, 1] == coor[:, 1].min()

        # 孔洞边界: 距离中心点一定距离的节点
        center = np.array([0.5, 0.5])
        distances = np.linalg.norm(coor - center, axis=1)
        hole_boundary = (distances > 0.2) & (distances < 0.3)  # 假设孔洞半径约0.25

        flag_BCy = (top_boundary | bottom_boundary | hole_boundary).astype(float)

    # 获取加载边界节点的索引
    load_node_indices = np.where(flag_load == 1)[0]
    num_bc_nodes = len(load_node_indices)
    print(f"找到 {num_bc_nodes} 个加载边界节点")

    # 创建均匀拉伸荷载 (100MPa = 100 N/mm²)
    uniform_load = 100.0  # MPa
    num_samples = 2  # 测试样本数量

    # 创建参数输入: 对于均匀荷载，所有边界节点的f_bc值相同
    bc_coords = coor[load_node_indices]
    params = []

    for i in range(num_samples):
        # 创建均匀荷载分布
        bc_values = np.full(len(load_node_indices), uniform_load)
        param_sample = np.concatenate([bc_coords, bc_values.reshape(-1, 1)], axis=1)
        params.append(param_sample)

    params = np.array(params)  # (num_samples, num_bc_nodes, 3)

    # 创建零位移场作为占位符 (实际预测时会由模型生成)
    u = np.zeros((num_samples, n_nodes))
    v = np.zeros((num_samples, n_nodes))

    # 材料属性
    youngs = 300.0e5 * 1e-4  # 与训练数据一致
    nu = 0.3

    # 转换为torch tensor
    params = torch.tensor(params, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    coors = torch.tensor(coor, dtype=torch.float32)

    # 创建测试数据集
    test_dataset = torch.utils.data.TensorDataset(params, u, v)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"均匀拉伸荷载测试数据创建完成:")
    print(f"  荷载大小: {uniform_load} MPa")
    print(f"  测试样本数: {num_samples}")
    print(f"  边界节点数: {num_bc_nodes}")
    print(f"  总节点数: {n_nodes}")

    return coors, test_loader, youngs, nu, num_bc_nodes, flag_BCxy, flag_BCy, flag_load
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
    coor_x = get_data('xx')
    coor_y = get_data('yy')
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
    flag_BCxy = get_data('flag_BCxy')
    flag_BCy = get_data('flag_BCy')
    flag_load = get_data('flag_BC_load')

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
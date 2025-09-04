elif args.phase == 'predict':
# 预测阶段 - 泛化检验
print("Starting prediction phase...")

# 加载训练好的模型
model_path = r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded trained model from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file {model_path} not found. Please train the model first.")
    exit(1)

model.eval()
model.to(device)

# 创建新的边界条件：右端受到大小为5的边界荷载
print("Creating new boundary conditions with load magnitude 5...")

# 获取右边界节点的坐标（使用 flag_BC_load）
right_nodes_indices = np.where(flag_BC_load.numpy().flatten() == 1)[0]
right_nodes_coords = coors.numpy()[right_nodes_indices]
num_right_nodes = len(right_nodes_indices)

print(f"Number of right boundary nodes: {num_right_nodes}")

# 创建新的边界条件参数：所有右边界节点荷载为5
new_bc_value = 5.0  # 大小为5的边界荷载
new_params = np.column_stack([
    right_nodes_coords[:, 0],  # x坐标
    right_nodes_coords[:, 1],  # y坐标
    np.full(num_right_nodes, new_bc_value)  # 所有节点荷载为5
])

# 扩展为批量形式 (1, num_right_nodes, 3)
new_params = np.expand_dims(new_params, axis=0)
new_params = torch.tensor(new_params, dtype=torch.float32).to(device)

print(f"New parameters shape: {new_params.shape}")

# 创建对应的真实位移（这里用零填充，因为我们是预测）
dummy_u = torch.zeros(1, coors.shape[0], dtype=torch.float32).to(device)
dummy_v = torch.zeros(1, coors.shape[0], dtype=torch.float32).to(device)

# 创建数据加载器
predict_dataset = torch.utils.data.TensorDataset(new_params, dummy_u, dummy_v)
predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1, shuffle=False)

# 进行预测
print("Running prediction...")
with torch.no_grad():
    for batch_idx, (par, u, v) in enumerate(predict_loader):
        print(f"Processing batch {batch_idx + 1}")

        # 准备坐标数据
        x_coor = coors[:, 0].unsqueeze(0).float().to(device)
        y_coor = coors[:, 1].unsqueeze(0).float().to(device)

        print(f"x_coor shape: {x_coor.shape}")
        print(f"y_coor shape: {y_coor.shape}")
        print(f"par shape: {par.shape}")

        # 模型前向传播
        u_pred, v_pred = model(x_coor, y_coor, par)

        print(f"u_pred shape: {u_pred.shape}")
        print(f"v_pred shape: {v_pred.shape}")

# 将预测结果转换为numpy
u_pred_np = u_pred.cpu().numpy()[0]
v_pred_np = v_pred.cpu().numpy()[0]
coors_np = coors.numpy()

print(f"Prediction completed. u_pred range: [{np.min(u_pred_np):.6f}, {np.max(u_pred_np):.6f}]")
print(f"Prediction completed. v_pred range: [{np.min(v_pred_np):.6f}, {np.max(v_pred_np):.6f}]")

# 确保输出目录存在
import os

os.makedirs('../res/plots', exist_ok=True)
os.makedirs('../res/predictions', exist_ok=True)

# 绘制x方向位移图
plt.figure(figsize=(12, 5))
scatter = plt.scatter(coors_np[:, 0], coors_np[:, 1], c=u_pred_np, cmap='jet', s=10)
plt.colorbar(scatter, label='X-Displacement')
plt.title('X-Direction Displacement (Load = 5)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal')
plt.savefig('../res/plots/x_displacement_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("X-displacement plot saved.")

# 绘制y方向位移图
plt.figure(figsize=(12, 5))
scatter = plt.scatter(coors_np[:, 0], coors_np[:, 1], c=v_pred_np, cmap='jet', s=10)
plt.colorbar(scatter, label='Y-Displacement')
plt.title('Y-Direction Displacement (Load = 5)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal')
plt.savefig('../res/plots/y_displacement_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("Y-displacement plot saved.")

# 绘制合位移图
total_displacement = np.sqrt(u_pred_np ** 2 + v_pred_np ** 2)
plt.figure(figsize=(12, 5))
scatter = plt.scatter(coors_np[:, 0], coors_np[:, 1], c=total_displacement, cmap='jet', s=10)
plt.colorbar(scatter, label='Total Displacement')
plt.title('Total Displacement (Load = 5)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal')
plt.savefig('../res/plots/total_displacement_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("Total displacement plot saved.")

# 输出统计信息
print("\nPrediction Results Summary:")
print(f"Max X-displacement: {np.max(u_pred_np):.6f}")
print(f"Min X-displacement: {np.min(u_pred_np):.6f}")
print(f"Max Y-displacement: {np.max(v_pred_np):.6f}")
print(f"Min Y-displacement: {np.min(v_pred_np):.6f}")
print(f"Max total displacement: {np.max(total_displacement):.6f}")

# 保存预测结果
np.savez('../res/predictions/prediction_results.npz',
         x_displacement=u_pred_np,
         y_displacement=v_pred_np,
         coordinates=coors_np,
         load_value=new_bc_value)

print("Prediction data saved to ../res/predictions/prediction_results.npz")
print("Prediction completed successfully!")
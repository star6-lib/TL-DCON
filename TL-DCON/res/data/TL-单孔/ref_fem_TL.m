% 带孔平板有限元分析 - 使用新数据格式
clear all; close all; clc;

% 加载数据文件
load('Dataset_1Circle.mat'); % 替换为您的.mat文件名

% 从加载的数据中获取节点坐标
node_coords = [xx, yy]; % 组合x,y坐标 [x, y]
num_nodes = length(xx);

% 确定边界节点
x_coords = node_coords(:,1);
y_coords = node_coords(:,2);

% 边界识别容差
tol = 1e-6 * (max(x_coords) - min(x_coords));

% 1. 左边界节点 (固定)
left_nodes = find(abs(x_coords - min(x_coords)) < tol);

% 2. 右边界节点 (施加荷载)
right_nodes = find(abs(x_coords - max(x_coords)) < tol);

% 3. 孔边界节点 (自由，不处理)
% 通过几何特征识别孔边界 (假设孔位于中心)
x_center = (min(x_coords) + max(x_coords))/2;
y_center = (min(y_coords) + max(y_coords))/2;
hole_radius = 0.2 * min(max(x_coords)-min(x_coords), max(y_coords)-min(y_coords));
hole_nodes = find(sqrt((x_coords-x_center).^2 + (y_coords-y_center).^2) <= hole_radius);

% 材料参数 (单位: N, mm)
E = 30000000;        % 杨氏模量(MPa)
nu = 0.25;         % 泊松比
thickness = 10;    % 板厚(mm)

% 荷载参数
total_force = 2000; % 2kN = 2000N
right_force_per_node = total_force / length(right_nodes); % 节点力(N)

% 使用Delaunay三角剖分生成三角形网格
tri = delaunayTriangulation(node_coords);
elements = tri.ConnectivityList;
num_elements = size(elements, 1);

% 定义材料属性矩阵 (平面应力)
D = (E/(1-nu^2)) * [1, nu, 0;
                   nu, 1, 0;
                   0, 0, (1-nu)/2];

% 组装全局刚度矩阵
ndof = 2*num_nodes;  % 总自由度
K = sparse(ndof, ndof);

for e = 1:num_elements
    % 获取单元节点坐标
    elem_nodes = elements(e, :);
    elem_coords = node_coords(elem_nodes, :);
    
    % 计算三角形单元刚度矩阵
    Ke = tri_element_stiffness(elem_coords, D, thickness);
    
    % 组装到全局矩阵
    dof_indices = [2*elem_nodes-1; 2*elem_nodes];
    dof_indices = dof_indices(:)';
    
    for i = 1:length(dof_indices)
        for j = 1:length(dof_indices)
            K(dof_indices(i), dof_indices(j)) = K(dof_indices(i), dof_indices(j)) + Ke(i, j);
        end
    end
end

% 初始化载荷向量
F = zeros(ndof, 1);

% 在右边界节点施加x方向力
right_dofs_x = 2*right_nodes - 1; % x方向自由度
F(right_dofs_x) = right_force_per_node;

% 处理边界条件
fixed_dofs = [2*left_nodes-1; 2*left_nodes]; % 固定左边界x和y方向
free_dofs = setdiff(1:ndof, fixed_dofs);

% 求解自由度的位移
K_ff = K(free_dofs, free_dofs);
F_f = F(free_dofs) - K(free_dofs, fixed_dofs) * zeros(length(fixed_dofs),1); % 固定端位移为0

U_f = K_ff \ F_f;

% 合并所有自由度位移
U = zeros(ndof, 1);
U(free_dofs) = U_f;

% 提取位移分量 (单位: mm)
ux = U(1:2:end-1);
uy = U(2:2:end);

% 计算应力 (可选)
stress = calculate_stress(node_coords, elements, U, D);

% 可视化结果
plot_results(node_coords, ux, uy, hole_nodes);

% 保存结果
save_results(node_coords, ux, uy);

% --- 辅助函数 ---
function Ke = tri_element_stiffness(coords, D, thickness)
    % 计算三角形单元刚度矩阵 (考虑厚度)
    x = coords(:,1);
    y = coords(:,2);
    
    % 计算三角形面积
    A = 0.5 * abs((x(2)-x(1))*(y(3)-y(1)) - (x(3)-x(1))*(y(2)-y(1)));
    
    % 形函数导数
    b = [y(2)-y(3); y(3)-y(1); y(1)-y(2)] / (2*A);
    c = [x(3)-x(2); x(1)-x(3); x(2)-x(1)] / (2*A);
    
    % B矩阵
    B = zeros(3, 6);
    for n = 1:3
        B(1, 2*n-1) = b(n);
        B(2, 2*n) = c(n);
        B(3, 2*n-1) = c(n);
        B(3, 2*n) = b(n);
    end
    
    % 单元刚度矩阵
    Ke = B' * D * B * A * thickness;
end

function stress = calculate_stress(node_coords, elements, U, D)
    % 计算单元应力 (可选)
    num_elements = size(elements, 1);
    stress = zeros(num_elements, 3); % [σxx, σyy, σxy]
    
    for e = 1:num_elements
        elem_nodes = elements(e, :);
        elem_coords = node_coords(elem_nodes, :);
        
        x = elem_coords(:,1);
        y = elem_coords(:,2);
        A = 0.5 * abs((x(2)-x(1))*(y(3)-y(1)) - (x(3)-x(1))*(y(2)-y(1)));
        
        % B矩阵
        b = [y(2)-y(3); y(3)-y(1); y(1)-y(2)] / (2*A);
        c = [x(3)-x(2); x(1)-x(3); x(2)-x(1)] / (2*A);
        
        B = zeros(3, 6);
        for n = 1:3
            B(1, 2*n-1) = b(n);
            B(2, 2*n) = c(n);
            B(3, 2*n-1) = c(n);
            B(3, 2*n) = b(n);
        end
        
        % 单元位移
        Ue = [U(2*elem_nodes(1)-1); U(2*elem_nodes(1));
              U(2*elem_nodes(2)-1); U(2*elem_nodes(2));
              U(2*elem_nodes(3)-1); U(2*elem_nodes(3))];
        
        % 单元应力
        stress(e,:) = (D * B * Ue)';
    end
end

function plot_results(node_coords, ux, uy, hole_nodes)
    % 绘制位移云图
    figure('Position', [100, 100, 1200, 500]);
    
    % X方向位移
    subplot(1,2,1);
    scatter(node_coords(:,1), node_coords(:,2), 30, ux, 'filled');
    hold on;
    plot(node_coords(hole_nodes,1), node_coords(hole_nodes,2), 'k.', 'MarkerSize', 10);
    colorbar;
    title('X方向位移 (mm)');
    axis equal tight;
    
    % Y方向位移
    subplot(1,2,2);
    scatter(node_coords(:,1), node_coords(:,2), 30, uy, 'filled');
    hold on;
    plot(node_coords(hole_nodes,1), node_coords(hole_nodes,2), 'k.', 'MarkerSize', 10);
    colorbar;
    title('Y方向位移 (mm)');
    axis equal tight;
end

function save_results(node_coords, ux, uy)
    % 保存位移结果到Excel
    disp_data = [node_coords, ux, uy];
    filename = 'displacement_results.xlsx';
    writematrix(disp_data, filename, 'Sheet', 1, 'Range', 'A1');
    
    % 添加表头
    header = {'X_coord', 'Y_coord', 'Ux', 'Uy'};
    writecell(header, filename, 'Sheet', 1, 'Range', 'A1');
    writematrix(disp_data, filename, 'Sheet', 1, 'Range', 'A2');
    
    disp(['位移结果已保存到: ' filename]);
end
% 带孔平板有限元分析
clear all; close all; clc;

% 几何参数(mm)
L = 20;           % 平板边长
hole_radius = 3;  % 圆孔半径

% 材料参数
E = 200000;        % 杨氏模量(MPa)
nu = 0.25;         % 泊松比

% 网格参数
num_nodes_per_edge = 82;  % 每条边节点数
num_elements = (num_nodes_per_edge-1)^2;  % 单元数
num_nodes = num_nodes_per_edge^2;         % 节点总数

% 生成节点坐标
x = linspace(-L/2, L/2, num_nodes_per_edge);
y = linspace(-L/2, L/2, num_nodes_per_edge);
[X, Y] = meshgrid(x, y);
node_coords = [X(:), Y(:)];

% 移除圆孔内的节点
dist_from_center = sqrt(sum(node_coords.^2, 2));
valid_nodes = dist_from_center >= hole_radius;
node_coords = node_coords(valid_nodes, :);
num_nodes = size(node_coords, 1);

% 重新编号节点
node_map = zeros(size(valid_nodes));
node_map(valid_nodes) = 1:num_nodes;

% 生成单元
elements = zeros(num_elements, 4);  % 四节点四边形单元
element_count = 0;

for i = 1:num_nodes_per_edge-1
    for j = 1:num_nodes_per_edge-1
        % 获取四个节点的原始编号
        node1 = j + (i-1)*num_nodes_per_edge;
        node2 = (j+1) + (i-1)*num_nodes_per_edge;
        node3 = (j+1) + i*num_nodes_per_edge;
        node4 = j + i*num_nodes_per_edge;
        
        % 检查这些节点是否都在有效节点列表中
        if valid_nodes(node1) && valid_nodes(node2) && valid_nodes(node3) && valid_nodes(node4)
            element_count = element_count + 1;
            elements(element_count, :) = [node_map(node1), node_map(node2), node_map(node3), node_map(node4)];
        end
    end
end

% 移除未使用的单元行
elements = elements(1:element_count, :);
num_elements = element_count;

% 定义材料属性矩阵
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
    
    % 计算单元刚度矩阵
    Ke = element_stiffness(elem_coords, D);
    
    % 组装全局刚度矩阵
    dof = [2*elem_nodes-1; 2*elem_nodes];
    for i = 1:8
        for j = 1:8
            K(dof(i), dof(j)) = K(dof(i), dof(j)) + Ke(i, j);
        end
    end
end

% 应用边界条件
% 1. 孔边界固定
hole_nodes = find(sqrt(sum(node_coords.^2, 2)) <= hole_radius*1.01);
fixed_dofs = [2*hole_nodes-1; 2*hole_nodes];

% 2. 左右边界位移
left_nodes = find(node_coords(:,1) <= -L/2*0.99);
right_nodes = find(node_coords(:,1) >= L/2*0.99);
left_dofs = 2*left_nodes-1;
right_dofs = 2*right_nodes-1;

% 位移值(mm)
left_disp = -2;
right_disp = 2;

% 组装载荷向量
F = zeros(ndof, 1);

% 应用位移边界条件
free_dofs = setdiff(1:ndof, [fixed_dofs; left_dofs; right_dofs]);

% 修改刚度矩阵和载荷向量以应用位移边界条件
for i = 1:length(left_dofs)
    dof = left_dofs(i);
    K(dof, :) = 0;
    K(dof, dof) = 1;
    F(dof) = left_disp;
end

for i = 1:length(right_dofs)
    dof = right_dofs(i);
    K(dof, :) = 0;
    K(dof, dof) = 1;
    F(dof) = right_disp;
end

% 求解线性方程组
U = K \ F;

% 提取位移分量
ux = U(1:2:ndof-1);
uy = U(2:2:ndof);

% 可视化位移结果 - X方向位移
figure('Position', [100, 100, 800, 600]);
scatter(node_coords(:,1), node_coords(:,2), 30, ux, 'filled');
colorbar;
title('X方向位移');
xlabel('X (mm)');
ylabel('Y (mm)');
axis equal tight;
% 保存X方向位移图
print('x_disp.png', '-dpng', '-r300');

% 可视化位移结果 - Y方向位移
figure('Position', [100, 100, 800, 600]);
scatter(node_coords(:,1), node_coords(:,2), 30, uy, 'filled');
colorbar;
title('Y方向位移');
xlabel('X (mm)');
ylabel('Y (mm)');
axis equal tight;
% 保存Y方向位移图
print('y_disp.png', '-dpng', '-r300');

% 定义需要查询的点坐标
query_points = [9.5, 10; 
                9, 10; 
                8.5, 10; 
                8, 10; 
                7.5, 10; 
                7, 10];

% 查找最近节点并获取位移数据
disp_data = zeros(size(query_points, 1), 4); % [x, y, ux, uy]

for i = 1:size(query_points, 1)
    % 计算所有节点到查询点的距离
    dist = sqrt((node_coords(:,1)-query_points(i,1)).^2 + (node_coords(:,2)-query_points(i,2)).^2);
    [~, idx] = min(dist);
    
    % 存储数据
    disp_data(i, :) = [node_coords(idx,1), node_coords(idx,2), ux(idx), uy(idx)];
end

% 输出到Excel文件
filename = 'displacement_data.xlsx';
header = {'X坐标(mm)', 'Y坐标(mm)', 'X方向位移(mm)', 'Y方向位移(mm)'};
data = num2cell(disp_data);
output = [header; data];
writecell(output, filename);

disp(['位移数据已保存到: ' filename]);

% 单元刚度矩阵计算函数
function Ke = element_stiffness(coords, D)
    % coords: 4x2矩阵，包含四个节点的坐标
    % D: 材料属性矩阵
    
    % Gauss积分点
    gauss_points = [-0.57735, 0.57735];
    gauss_weights = [1, 1];
    
    Ke = zeros(8, 8);
    
    % 循环每个Gauss点
    for i = 1:2
        xi = gauss_points(i);
        for j = 1:2
            eta = gauss_points(j);
            weight = gauss_weights(i) * gauss_weights(j);
            
            % 形函数导数
            dN = [
                -0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta);
                -0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi)
            ];
            
            % Jacobian矩阵
            J = dN * coords;
            detJ = det(J);
            invJ = inv(J);
            
            % B矩阵
            B = zeros(3, 8);
            dN_xy = invJ * dN;
            
            for n = 1:4
                B(1, 2*n-1) = dN_xy(1, n);
                B(2, 2*n) = dN_xy(2, n);
                B(3, 2*n-1) = dN_xy(2, n);
                B(3, 2*n) = dN_xy(1, n);
            end
            
            % 累加单元刚度矩阵
            Ke = Ke + B' * D * B * detJ * weight;
        end
    end
end
% 带孔平板有限元分析 - 使用外部数据文件
clear all; close all; clc;

% 加载数据文件
load('plate_dis_high.mat'); % 假设文件包含coors和flag_BCxy变量

% 从加载的数据中获取节点坐标和边界条件信息
node_coords = coors; % 节点坐标 [x, y]
hole_boundary_flags = flag_BCxy; % 孔边界标志 (1表示是孔边界)

% 确定孔边界节点
hole_nodes = find(hole_boundary_flags == 1);

% 材料参数
E = 200000;        % 杨氏模量(MPa)
nu = 0.25;         % 泊松比

% 确定节点和单元信息
num_nodes = size(node_coords, 1);

% 使用Delaunay三角剖分生成三角形网格
tri = delaunayTriangulation(node_coords);
elements = tri.ConnectivityList;
num_elements = size(elements, 1);

% 检查单元类型
if size(elements, 2) == 3
    disp('检测到三角形单元，将使用三角形单元刚度矩阵');
    element_type = 'triangle';
elseif size(elements, 2) == 4
    disp('检测到四边形单元');
    element_type = 'quadrilateral';
else
    error('未知的单元类型');
end

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
    
    % 根据单元类型计算单元刚度矩阵
    if strcmp(element_type, 'quadrilateral')
        Ke = quad_element_stiffness(elem_coords, D);
        dof_indices = [2*elem_nodes-1; 2*elem_nodes];
        dof_indices = dof_indices(:)'; % 转换为行向量
    else
        Ke = tri_element_stiffness(elem_coords, D);
        dof_indices = [2*elem_nodes-1; 2*elem_nodes];
        dof_indices = dof_indices(:)'; % 转换为行向量
    end
    
    % 组装全局刚度矩阵
    for i = 1:length(dof_indices)
        for j = 1:length(dof_indices)
            K(dof_indices(i), dof_indices(j)) = K(dof_indices(i), dof_indices(j)) + Ke(i, j);
        end
    end
end

% 应用边界条件
% 1. 孔边界固定
fixed_dofs = [2*hole_nodes-1; 2*hole_nodes];
fixed_dofs = fixed_dofs(:); % 转换为列向量

% 2. 左右边界位移
% 确定左右边界节点
x_coords = node_coords(:,1);
x_min = min(x_coords);
x_max = max(x_coords);

tol = 1e-6 * (x_max - x_min); % 设置容差
left_nodes = find(abs(x_coords - x_min) < tol);
right_nodes = find(abs(x_coords - x_max) < tol);

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

% 准备所有节点的位移数据
disp_data = [node_coords, ux, uy]; % [x, y, ux, uy]

% 输出到Excel文件
filename = 'displacement_data.xlsx';
header = {'X坐标(mm)', 'Y坐标(mm)', 'X方向位移(mm)', 'Y方向位移(mm)'};
data = num2cell(disp_data);
output = [header; data];
writecell(output, filename);

disp(['所有节点的位移数据已保存到: ' filename]);

% 四边形单元刚度矩阵计算函数
function Ke = quad_element_stiffness(coords, D)
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
            
            % 形函数导数 (对于四边形单元)
            dN_dxi = 0.25 * [eta-1, 1-eta, 1+eta, -1-eta];
            dN_deta = 0.25 * [xi-1, -1-xi, 1+xi, 1-xi];
            
            % Jacobian矩阵
            J = [dN_dxi; dN_deta] * coords;
            detJ = det(J);
            invJ = inv(J);
            
            % 导数转换
            dN = [dN_dxi; dN_deta];
            dN_dxy = invJ * dN;
            dN_dx = dN_dxy(1,:);
            dN_dy = dN_dxy(2,:);
            
            % B矩阵
            B = zeros(3, 8);
            for n = 1:4
                B(1, 2*n-1) = dN_dx(n);
                B(2, 2*n) = dN_dy(n);
                B(3, 2*n-1) = dN_dy(n);
                B(3, 2*n) = dN_dx(n);
            end
            
            % 累加单元刚度矩阵
            Ke = Ke + B' * D * B * detJ * weight;
        end
    end
end

% 三角形单元刚度矩阵计算函数
function Ke = tri_element_stiffness(coords, D)
    % coords: 3x2矩阵，包含三个节点的坐标
    % D: 材料属性矩阵
    
    % 三角形单元使用面积坐标和单点积分
    
    % 节点坐标
    x = coords(:,1);
    y = coords(:,2);
    
    % 计算三角形面积
    A = 0.5 * abs((x(2)-x(1))*(y(3)-y(1)) - (x(3)-x(1))*(y(2)-y(1)));
    
    % 形函数导数 (常数应变三角形)
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
    Ke = B' * D * B * A;
end
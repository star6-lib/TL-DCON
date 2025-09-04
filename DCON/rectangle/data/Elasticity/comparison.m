close all
clear
load bc_source
global ubc

num = 2000; % Number of samples

for i = 1:num
    
    ubc = f_bc(i, :);
    
    model = createpde('structural','static-planestrain');
    
    % 创建长方形几何体 [x1, y1, x2, y2, x3, y3, x4, y4]
    % 长2宽1的长方形：从(0,0)到(2,1)
    rect = [3; 4; 0; 2; 2; 0; 0; 0; 1; 1];
    g = decsg(rect);
    
    geometryFromEdges(model,g);
    
    structuralProperties(model,'YoungsModulus',300.0E5,'PoissonsRatio',0.3);

    if i == 1
        figure
        pdegplot(model,'VertexLabels','on');
        title 'Geometry with Vertex Labels';
    end

    if i == 1
        figure
        pdegplot(model,'EdgeLabel','on');
        title 'Geometry with Edge Labels';
    end
    % 左边界固定 (Edge 4)
    structuralBC(model,'Edge',4,'XDisplacement',0, 'YDisplacement',0);
    
    % 在右边界施加水平荷载 (Edge 2)
    structuralBoundaryLoad(model,'Edge',2,'SurfaceTraction',@myload, 'Vectorized', 'on');
    
    generateMesh(model,'Hmax',0.05); % 调整网格大小
    R = solve(model);
    
    X = R.Mesh.Nodes;
    X = X';
    xx = X(:, 1);
    yy = X(:, 2);
    intrpDisp = interpolateDisplacement(R,xx,yy);
    
    ux(i,:) = reshape(intrpDisp.ux,size(xx));
    uy(i,:) = reshape(intrpDisp.uy,size(yy));
    
    stress_x(i,:) = R.Stress.sxx;
    stress_y(i,:) = R.Stress.syy;
    
    % 只在第一次迭代时创建边界标记变量和可视化
    if i == 1
        % 获取网格节点坐标
        nodes = R.Mesh.Nodes';
        
        % 创建边界标记变量（所有角点都划分到左右边界）
        % 左边界 (x = 0) - 包含左下和左上角点
        left_boundary = (abs(nodes(:,1) - 0) < 1e-6);
        
        % 右边界 (x = 2) - 包含右下和右上角点
        right_boundary = (abs(nodes(:,1) - 2) < 1e-6);
        
        % 上下边界 (y = 0 或 y = 1)，排除左右边界的点
        top_bottom_boundary = ((abs(nodes(:,2) - 0) < 1e-6) | (abs(nodes(:,2) - 1) < 1e-6)) & ...
                             ~(left_boundary | right_boundary);
        
        % 创建标记向量
        flag_BCxy = double(left_boundary);
        flag_BCy = double(top_bottom_boundary);
        flag_BC_load = double(right_boundary);
        
        % 可视化边界标记
        figure;
        hold on;
        
        % 绘制所有网格节点
        scatter(nodes(:,1), nodes(:,2), 20, 'k', 'filled', 'MarkerEdgeAlpha', 0.3, 'MarkerFaceAlpha', 0.3);
        
        % 标记左边界节点（红色）- 包含角点
        left_nodes = nodes(left_boundary, :);
        scatter(left_nodes(:,1), left_nodes(:,2), 50, 'r', 'filled', 'DisplayName', '左边界 (flag\_BCxy)');
        
        % 标记上下边界节点（绿色）- 不包含角点
        tb_nodes = nodes(top_bottom_boundary, :);
        scatter(tb_nodes(:,1), tb_nodes(:,2), 50, 'g', 'filled', 'DisplayName', '上下边界 (flag\_BCy)');
        
        % 标记右边界节点（蓝色）- 包含角点
        right_nodes = nodes(right_boundary, :);
        scatter(right_nodes(:,1), right_nodes(:,2), 50, 'b', 'filled', 'DisplayName', '右边界 (flag\_BC\_load)');
        
        title('边界节点标记可视化（角点划分到左右边界）');
        xlabel('X坐标');
        ylabel('Y坐标');
        legend('show');
        axis equal;
        grid on;
        hold off;
        
        % 显示统计信息
        fprintf('边界节点统计:\n');
        fprintf('左边界节点数: %d\n', sum(left_boundary));
        fprintf('右边界节点数: %d\n', sum(right_boundary));
        fprintf('上下边界节点数: %d\n', sum(top_bottom_boundary));
        fprintf('总节点数: %d\n', size(nodes, 1));
        
        % 验证角点确实在左右边界中
        corner_nodes = nodes((abs(nodes(:,1) - 0) < 1e-6 & (abs(nodes(:,2) - 0) < 1e-6)) | ...
                            (abs(nodes(:,1) - 0) < 1e-6 & (abs(nodes(:,2) - 1) < 1e-6)) | ...
                            (abs(nodes(:,1) - 2) < 1e-6 & (abs(nodes(:,2) - 0) < 1e-6)) | ...
                            (abs(nodes(:,1) - 2) < 1e-6 & (abs(nodes(:,2) - 1) < 1e-6)), :);
        fprintf('角点数量: %d\n', size(corner_nodes, 1));
        fprintf('角点在左边界中的数量: %d\n', sum(ismember(corner_nodes, left_nodes, 'rows')));
        fprintf('角点在右边界中的数量: %d\n', sum(ismember(corner_nodes, right_nodes, 'rows')));
        fprintf('角点在上下边界中的数量: %d\n', sum(ismember(corner_nodes, tb_nodes, 'rows')));
    end
    
    % 可选：显示某个样本的位移云图
    if i == 1 % 只显示第一个样本的结果作为示例
        figure
        % 计算总位移大小
        total_disp = sqrt(R.Displacement.ux.^2 + R.Displacement.uy.^2);
        pdeplot(model,'XYData',total_disp,'ColorMap','jet')
        title('Total Displacement Magnitude (Sample 1)');
    end
end

ux = ux(1:2000,:);
uy = uy(1:2000,:);
stressX = stress_x(1:2000,:);
stressY = stress_y(1:2000,:);
f_bc = f_bc(1:2000,:);

% 保存所有变量，包括边界标记
save('Dataset_Rectangle', 'f_bc', 'xx', 'yy', 'ux', ...
    'uy', 'stressX', 'stressY', 'flag_BCxy', 'flag_BCy', 'flag_BC_load');
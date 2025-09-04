function addBoundaryFlagsToDataset(matFilePath)
% 在已有的Dataset_1Circle.mat文件中添加边界标记变量
% 输入参数：matFilePath - .mat文件的完整路径

% 如果未提供文件路径，使用默认文件名
if nargin < 1
    matFilePath = 'Dataset_1Circle.mat';
end

% 加载现有数据
load(matFilePath);

% 获取网格节点坐标
X_coords = xx;
Y_coords = yy;

% 定义容差（用于判断节点是否在边界上）
tol = 1e-5;

% ========== 首先识别所有边界 ==========
% 左边界：x坐标接近0的节点
left_boundary = abs(X_coords - 0) < tol;

% 右边界：x坐标接近1的节点
right_boundary = abs(X_coords - 1) < tol;

% 上边界：y坐标接近1的节点
top_boundary = abs(Y_coords - 1) < tol;

% 下边界：y坐标接近0的节点
bottom_boundary = abs(Y_coords - 0) < tol;

% 孔洞边界：距离圆心(0.5, 0.5)半径为0.25的圆
center_x = 0.5;
center_y = 0.5;
radius = 0.25;
distance_from_center = sqrt((X_coords - center_x).^2 + (Y_coords - center_y).^2);
hole_boundary = abs(distance_from_center - radius) < tol;

% ========== 处理边界重合点 ==========
% 找出左右边界与上下边界的重合点
left_top_corner = left_boundary & top_boundary;      % 左上角
left_bottom_corner = left_boundary & bottom_boundary; % 左下角
right_top_corner = right_boundary & top_boundary;    % 右上角
right_bottom_corner = right_boundary & bottom_boundary; % 右下角

% 从上下边界中排除与左右边界重合的点
top_boundary_clean = top_boundary & ~(left_top_corner | right_top_corner);
bottom_boundary_clean = bottom_boundary & ~(left_bottom_corner | right_bottom_corner);

% ========== 创建边界标记变量 ==========
% flag_BCxy：左边界标记（包含重合点）
flag_BCxy = double(left_boundary);

% flag_BCy：上下边界及孔洞边界标记（不包含与左右边界重合的点）
y_constrained_boundaries = top_boundary_clean | bottom_boundary_clean | hole_boundary;
flag_BCy = double(y_constrained_boundaries);

% flag_BC_load：右边界标记（包含重合点）
flag_BC_load = double(right_boundary);

% ========== 验证边界标记 ==========
fprintf('边界标记统计:\n');
fprintf('左边界节点数 (flag_BCxy): %d (包含%d个角点)\n', sum(flag_BCxy), sum(left_top_corner | left_bottom_corner));
fprintf('Y约束边界节点数 (flag_BCy): %d (已排除角点)\n', sum(flag_BCy));
fprintf('右边界节点数 (flag_BC_load): %d (包含%d个角点)\n', sum(flag_BC_load), sum(right_top_corner | right_bottom_corner));
fprintf('左上角点: %d, 左下角点: %d\n', sum(left_top_corner), sum(left_bottom_corner));
fprintf('右上角点: %d, 右下角点: %d\n', sum(right_top_corner), sum(right_bottom_corner));
fprintf('总节点数: %d\n', length(X_coords));

% ========== 可视化验证（可选） ==========
visualizeBoundaryFlags(X_coords, Y_coords, flag_BCxy, flag_BCy, flag_BC_load, ...
                      left_top_corner, left_bottom_corner, right_top_corner, right_bottom_corner);

% ========== 保存到原文件 ==========
save(matFilePath, 'f_bc_train', 'f_bc_test', 'xx', 'yy', ...
     'ux_train', 'ux_test', 'uy_train', 'uy_test', ...
     'stressX_train', 'stressX_test', 'stressY_train', 'stressY_test', ...
     'flag_BCxy', 'flag_BCy', 'flag_BC_load', '-v7.3');

fprintf('边界标记变量已成功添加到文件: %s\n', matFilePath);

end

function visualizeBoundaryFlags(X, Y, flag_BCxy, flag_BCy, flag_BC_load, ...
                               left_top, left_bottom, right_top, right_bottom)
% 可视化边界标记
    
    figure('Position', [100, 100, 1200, 400]);
    
    % 子图1: 左边界（包含角点）
    subplot(1,3,1);
    % 绘制所有点
    scatter(X, Y, 10, [0.8 0.8 0.8], 'filled', 'MarkerEdgeAlpha', 0.3);
    hold on;
    % 绘制左边界（红色）
    scatter(X(flag_BCxy == 1), Y(flag_BCxy == 1), 30, 'r', 'filled');
    % 标记角点（黄色）
    scatter(X(left_top), Y(left_top), 50, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    scatter(X(left_bottom), Y(left_bottom), 50, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    hold off;
    title('左边界标记 (flag\_BCxy)');
    axis equal;
    xlabel('X坐标');
    ylabel('Y坐标');
    legend('其他', '左边界', '角点', 'Location', 'best');
    
    % 子图2: Y约束边界（不包含角点）
    subplot(1,3,2);
    % 绘制所有点
    scatter(X, Y, 10, [0.8 0.8 0.8], 'filled', 'MarkerEdgeAlpha', 0.3);
    hold on;
    % 绘制Y约束边界（蓝色）
    scatter(X(flag_BCy == 1), Y(flag_BCy == 1), 30, 'b', 'filled');
    hold off;
    title('Y约束边界标记 (flag\_BCy)');
    axis equal;
    xlabel('X坐标');
    ylabel('Y坐标');
    legend('其他', 'Y约束边界', 'Location', 'best');
    
    % 子图3: 右边界（包含角点）
    subplot(1,3,3);
    % 绘制所有点
    scatter(X, Y, 10, [0.8 0.8 0.8], 'filled', 'MarkerEdgeAlpha', 0.3);
    hold on;
    % 绘制右边界（绿色）
    scatter(X(flag_BC_load == 1), Y(flag_BC_load == 1), 30, 'g', 'filled');
    % 标记角点（黄色）
    scatter(X(right_top), Y(right_top), 50, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    scatter(X(right_bottom), Y(right_bottom), 50, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    hold off;
    title('右边界标记 (flag\_BC\_load)');
    axis equal;
    xlabel('X坐标');
    ylabel('Y坐标');
    legend('其他', '右边界', '角点', 'Location', 'best');
    
    sgtitle('边界标记可视化（角点统一标记到左右边界）');
end
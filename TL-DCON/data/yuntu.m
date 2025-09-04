% 加载.mat文件
load('plate_dis_high.mat'); % 替换为你的实际文件名

% 提取坐标数据
x_coords = coors(:, 1);
y_coords = coors(:, 2);

% 设置要绘制的份数
num_plots = min(5, size(final_u, 1)); % 最多绘制5份

% 为每份数据创建位移散点图
for i = 1:num_plots
    % 提取当前份的位移数据
    u_data = final_u(i, :)';
    v_data = final_v(i, :)';
    
    % 创建新图形
    figure;
    
    % 绘制x方向位移散点图
    subplot(2, 1, 1);
    scatter(x_coords, y_coords, 20, u_data, 'filled');
    colorbar;
    title(sprintf('X方向位移散点图 (第%d份)', i));
    xlabel('X坐标');
    ylabel('Y坐标');
    axis equal;
    
    % 绘制y方向位移散点图
    subplot(2, 1, 2);
    scatter(x_coords, y_coords, 20, v_data, 'filled');
    colorbar;
    title(sprintf('Y方向位移散点图 (第%d份)', i));
    xlabel('X坐标');
    ylabel('Y坐标');
    axis equal;
    
    % 调整图形布局
    set(gcf, 'Position', [100, 100, 800, 800]);
end
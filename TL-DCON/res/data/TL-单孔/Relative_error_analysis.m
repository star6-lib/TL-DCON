% 读取 predict.xlsx 文件数据
predictData = readtable('predict.xlsx');

% 读取 ref.xlsx 文件数据
refData = readtable('ref.xlsx');

% 提取坐标数据
x = predictData.x;
y = predictData.y;

% 提取 u 和 v 方向的位移数据
u_predict = predictData.u;
v_predict = predictData.v;
u_ref = refData.u;
v_ref = refData.v;

% 定义最小分母阈值（防止除以0）
min_denominator = 1e-10;

% 严格按照图示公式计算u方向相对误差（百分比形式）
rel_error_u = abs(u_predict - u_ref) ./ max(abs(u_ref), min_denominator) * 100;

% 严格按照图示公式计算v方向相对误差（百分比形式）
rel_error_v = abs(v_predict - v_ref) ./ max(abs(v_ref), min_denominator) * 100;

% 设置图片清晰度
set(groot, 'defaultFigureRenderer', 'painters');
set(groot, 'defaultFigureRendererMode','manual');

% 绘制 u 方向位移相对误差散点图
figure('Position', [100 100 800 600]);
s1 = scatter(x, y, 30, rel_error_u, 'filled', 'MarkerFaceAlpha', 0.8);
hot_map = hot;
hot_map = flipud(hot_map);
colormap(hot_map); 
c1 = colorbar('Location', 'eastoutside');
c1.Label.String = 'u 方向相对误差 (%)';  % 添加百分比单位
title('u 方向位移相对误差 (%)');        % 添加百分比单位
xlabel('x 坐标');
ylabel('y 坐标');
print('-dpng', '-r300', 'u_direction_error.png');

% 绘制 v 方向位移相对误差散点图
figure('Position', [100 100 800 600]);
s2 = scatter(x, y, 30, rel_error_v, 'filled', 'MarkerFaceAlpha', 0.8);
colormap('cool'); 
c2 = colorbar('Location', 'eastoutside');
c2.Label.String = 'v 方向相对误差 (%)';  % 添加百分比单位
title('v 方向位移相对误差 (%)');        % 添加百分比单位
xlabel('x 坐标');
ylabel('y 坐标');
print('-dpng', '-r300', 'v_direction_error.png');
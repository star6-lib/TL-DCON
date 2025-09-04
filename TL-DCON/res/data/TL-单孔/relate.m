% 读取 predict.xlsx 文件数据
predictData = readtable('predict.xlsx');

% 读取 ref.xlsx 文件数据
refData = readtable('ref.xlsx');

% 提取坐标数据
x = predictData{:, 1};
y = predictData{:, 2};

% 提取 u 和 v 方向的位移数据
u_predict = predictData{:, 3};
v_predict = predictData{:, 4};
u_ref = refData{:, 3};
v_ref = refData{:, 4};

% 计算 u 方向的相对误差
relative_error_u = abs((u_ref - u_predict)./ u_ref);

% 计算 v 方向的相对误差
relative_error_v = abs((v_ref - v_predict)./ v_ref);

% 设置图片清晰度
set(groot, 'defaultFigureRenderer', 'painters');
set(groot, 'defaultFigureRendererMode','manual');

% 绘制 u 方向相对误差散点图
figure('Position', [100 100 800 600]);
s1 = scatter(x, y, 30, relative_error_u, 'filled', 'MarkerFaceAlpha', 0.8);
colormap('hot'); 
c1 = colorbar('Location', 'eastoutside');
c1.Label.String = 'u 方向相对误差';
title('原几何模型上 u 方向位移相对误差散点图');
xlabel('x 坐标');
ylabel('y 坐标');

% 绘制 v 方向相对误差散点图
figure('Position', [100 100 800 600]);
s2 = scatter(x, y, 30, relative_error_v, 'filled', 'MarkerFaceAlpha', 0.8);
colormap('cool'); 
c2 = colorbar('Location', 'eastoutside');
c2.Label.String = 'v 方向相对误差';
title('原几何模型上 v 方向位移相对误差散点图');
xlabel('x 坐标');
ylabel('y 坐标');
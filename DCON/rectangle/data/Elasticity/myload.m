function bcMatrix = myload(location, state)

bcMatrix = zeros(2,length(location.x));
global ubc
N = 101;
X = linspace(0, 1, N); % 修改为从0到1，对应y坐标从0到1

% 在右边界上，根据y坐标插值载荷
load_x = interp1(X, ubc, location.y);

bcMatrix(1,:) = 1e3*load_x; % x方向荷载
bcMatrix(2,:) = 0;          % y方向荷载为0
end
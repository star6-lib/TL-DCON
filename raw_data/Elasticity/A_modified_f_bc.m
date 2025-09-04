% 加载bc_source.mat文件
load('bc_source.mat');
f_bc(1,:) = 5;
save('bc_source.mat', 'f_bc');
disp('操作完成，第一行数据已被修改')

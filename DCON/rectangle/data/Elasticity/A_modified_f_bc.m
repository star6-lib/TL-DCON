% 加载 bc_source.mat 文件
load('bc_source.mat');

% 将 f_bc 变量中第一行的所有数据设置为1
f_bc(1, :) = 5;

% 保存修改后的变量回原文件（覆盖原文件）
save('bc_source.mat', 'f_bc');

% 或者保存到新文件（保留原文件）
% save('bc_source_modified.mat', 'f_bc');

disp('操作完成！第一行数据已全部被修改。');
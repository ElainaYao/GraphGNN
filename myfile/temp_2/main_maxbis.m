%% read files
N = 100;
p = 0.5;
generative_model = 'ErdosRenyi';
num_test = 1000;
path_output = '/Users/Rebecca_yao/Documents/RESEARCH/Graph/GraphGNN/myfile/temp_2/output/';
file_name = sprintf('testdata_%s_N%d_p%0.1f_num%d.mat', generative_model, N, p, num_test);
path_plus_name = fullfile(path_output, file_name);
load(path_plus_name)
%% make W
L2 = double(L);
[num_test, ~, N] = size(L2);
W = zeros(N+1, N+1);
num_rounding = 1000;
res = zeros(num_test,1);
for test_i = 1:num_test
    if mod(test_i, 50) == 0 
        test_i
    end
    W(1:N, 1:N) = L2(test_i, :, :);
    Y = zeros(N+1, N+1);
    Y(1:(N+1), N+1) = ones(N+1,1);   
    % SDP
    res(test_i) = SDP_maxbis(W,Y,num_rounding);
end
%% save files
file_name = sprintf('res_%s_N%d_p%0.1f_num%d.mat', generative_model, N, p, num_test);
path_plus_name = fullfile(path_output, file_name);
save(path_plus_name, 'res');
function [res] = SDP_maxbis(W,Y,num_rounding)
[N, ~] = size(W);
N = N-1;
%% SDP
%-- A is the adjacency matrix
%-- W is a matrix of size (n+1) x (n+1), 
%----- upper left nxn block = A, 
%----- lower left 1xn block = zeros(1,n)
%----- upper right nx1 block = zeros(n,1)
%----- lower right 1x1 block = zeros(1,1)
%-- n is the total number of the nodes
%-- Y is a matrix of size (n+1) x (n+1), 
%----- last column of Y is 1's
cvx_begin sdp quiet 
    variable X(N+1,N+1) symmetric
    maximize trace(W*X)
    X == semidefinite(N+1);
    diag(X) == ones(N+1,1);
    trace(X*Y) == 1;
cvx_end
%% Rounding
res = zeros(1,num_rounding);
for j = 1:num_rounding
    x_res = ones(N,1);
    while sum(x_res) ~= 0
        U = chol(X); % X of size (n+1)x(n+1)
        x = sign(U*randn(N+1,1)); % We only care about the first n entries
        x_res = x(1:N);
    end
    X_res = x_res*transpose(x_res);
    % check sum(x(1:N)) == 0
    res(j) = trace(W(1:N, 1:N)*X_res);
end
res = mean(res);
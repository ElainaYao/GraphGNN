%% SDP
%-- A is the adjacency matrix
%-- W is a matrix of size (n+1) x (n+1), 
%----- upper left nxn block = A, 
%----- lower left 1xn block = zeros(1,n)
%----- upper right nx1 block = zeros(n,1)
%----- lower right 1x1 block = zeros(1,1)
%-- n is the total number of the nodes
cvx_begin sdp quiet 
    variable X(n+1,n+1) symmetric
    minimize trace(W*X)
    X == semidefinite(n+1);
    diag(X) == ones(n+1,1);
    trace(X*Y) == 1;
cvx_end
%% Rounding
U = chol(X); % X of size (n+1)x(n+1)
x = sign(U*randn(n+1,1)); % We only care about the first n entries
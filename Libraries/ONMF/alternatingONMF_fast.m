function [U, V] = alternatingONMF_fast(M, U, maxiter)

[m, n] = size(M); 
[m, r] = size(U);
% Mn: normalized version of M
norm2v = sqrt(sum(M.^2, 1)); 
Mn = M .* repmat(1 ./ (norm2v + 1e-16), m, 1); 

% Normalize columns of U
norm2v = sqrt(sum(U.^2, 1)); 
U = U .* repmat(1 ./ (norm2v + 1e-16), m, 1); 

for iter = 1:maxiter
    % V = argmin_V ||M-UV||_F, V >= 0, rows V orthogonal
    % Compute the angles between U and Mn
    A = Mn' * U;  % n by r matrix
    [~, b] = max(A');  % Find the index of the maximum value for each column
    
    % Precompute norms of columns of U (we need them for the denominator)
    norm_U = sqrt(sum(U.^2, 1));
    
    % Vectorized update of V using the indices b
    % This avoids the loop over columns of M
    V = zeros(r, n);  % Initialize V as a zero matrix
    V(sub2ind(size(V), b, 1:n)) = sum(M .* U(:, b), 1) ./ (norm_U(b).^2);  % Update V

    % U = argmin_U ||M-UV||_F, U >= 0
    U = (nnlsm_blockpivot(V', M'))';  % Update U using nnls
    
    % Normalize columns of U
    norm2v = sqrt(sum(U.^2, 1)); 
    U = U .* repmat(1 ./ (norm2v + 1e-16), m, 1);
end

% Final V calculation (outside the loop)
A = Mn' * U;  % n by r matrix
[~, b] = max(A');  % Find the index of the maximum value for each column

norm_U = sqrt(sum(U.^2, 1));  % Compute the norms of U columns
V = zeros(r, n);  % Initialize V as a zero matrix
V(sub2ind(size(V), b, 1:n)) = sum(M .* U(:, b), 1) ./ (norm_U(b).^2);  % Update V

% Scale (U,V) for V to have unit norm rows
for i = 1:r
    nvi = norm(V(i, :)); 
    V(i, :) = V(i, :) / nvi;
    U(:, i) = U(:, i) * nvi;
end

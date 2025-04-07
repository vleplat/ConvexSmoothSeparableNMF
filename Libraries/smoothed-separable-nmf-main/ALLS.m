% Algorithm for Learning a Latent Simplex
%
% function [W,K] = ALLS(X,r,p)
%
% Heuristic to solve the following problem:
% Given a matrix X, find a matrix W such that X~=WH for some H>=0,
% under the assumption that each column of W has p columns of X close
% to it (called the p proximal latent points).
%
% INPUTS
%
% X: data set of size m*n
% r: number of columns of W
% p: number of proximal latent points
%
% OUTPUTS
%
% W: the matrix such that X~=WH
% K: indices of the selected data points (one column per iteration)
%
%
% The Algorithm for Learning a latent simplex (LLS)
% was proposed in the paper
% Bhattacharyya, C., Kannan, R.: Finding a latent k-simplex in
% o*(k nnz (data)) time via subset smoothing.
% In: Proc. of the 14 Annual ACM-SIAM Symposium on Discrete Algorithms,
% pp. 122-140, 2020.
%
% This implementation is a supplementary material to the paper
% Smoothed Separable Nonnegative Matrix Factorization
% by N. Nadisic, N. Gillis, and C. Kervazo
% https://arxiv.org/abs/2110.05528


function [W,K] = ALLS(X,r,p)

[Y,S,Z] = svds(X,r); % Y contains the first r singular vectors of X
                     % Faster algorithms could be used; see the paper above
m = size(X,1);
% (I-VV^T) is the projection onto the orthogonal complement of the columns
% of W extracted so far.
V = [];
for k = 1 : r
    % Random direction in the subspace spanned by Y
    % and orthogonal to the previously extracted vertices,
    % then multiplied by the data matrix
    if k == 1
        u = (randn(1,r)*Y')*X;
        % Note: X could be replaced by Y*Z to reduce the computational load,
        % this is done in
        % Bakshi, A., Bhattacharyya, C., Kannan, R., Woodruff, D.P.,
        % Zhou, S.: Learning a latent simplex in input sparsity time.
        % In: ICLR, 2021.
    else
        diru = Y*rand(r,1);
        diru = diru - V*(V'*diru);
        u = diru'*X;
    end
    % Select largest entries in absolute value
    [a,b] = sort(-abs(u)); % This does not work well
    % Select indices
    K(k,:) = b(1:p);
    % Compute vertex
    if p == 1
        W(:,k) = X(:,K(k,:));
    else
        W(:,k) = mean( X(:,K(k,:))' )';
    end
    % Update the projector
    V = updateorthbasis(V,W(:,k));
end
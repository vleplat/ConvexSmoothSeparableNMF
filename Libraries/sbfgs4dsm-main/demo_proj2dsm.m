% Demo for the optimal doubly stochastic matrix approximaton
%  min_X 0.5*\|X - A\|_F^2, 
%  s.t. X>=0, Xe = e, X'e = e.

clc, clear;

dimVEC = [100; 1000];
for iter = 1:length(dimVEC)
    dim = dimVEC(iter);
    fprintf('Iter: %d, dim: %d\n', iter, dim);
    
    A = randn(dim);   
    e = ones(dim, 1);
    y0 = zeros(2*dim,1);
    
    %% Our Structured BFGS Method
    disp(' Our s-BFGS Algorithm ...');
    [X_BFGS, out_BFGS] = proj_dsm_BFGS(A, y0, dim);
end

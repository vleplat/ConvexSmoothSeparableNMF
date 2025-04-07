% Alternating orthogonal NMF
%
% Optimize alternatively U>=0 and V>=0, min ||M-UV|| rows of V orthogonal.

function [U,V] = alternatingONMF(M,U,maxiter)

[m,n] = size(M); 
[m,r] = size(U);
% Mn: normalized version of M
norm2v = sqrt(sum(M.^2,1)); 
Mn=M.*repmat(1./(norm2v+1e-16),m,1); 
% Normalize columns of U
norm2v = sqrt(sum(U.^2,1)); 
U=U.*repmat(1./(norm2v+1e-16),m,1);   
for i = 1 : maxiter
    % V = argmin_V ||M-UV||_F, V >= 0, rows V orthogonal
    % Compute the angles between U and Mn
    A = Mn'*U; %n by r matrix
    [a,b] = max(A'); 
    V = zeros(r,n);
    % to be improved without loop
    for i = 1 : n
        V(b(i),i) =  M(:,i)'*U(:,b(i))/norm(U(b(i),:))^2;
    end
    
    % U = argmin_U ||M-UV||_F, U >= 0
    U = (nnlsm_blockpivot(V',M'))';
    % Normalize columns of U
    norm2v = sqrt(sum(U.^2,1)); 
    U=U.*repmat(1./(norm2v+1e-16),m,1); 
    % if zero column; reinitialize to random
end
% V = argmin_V ||M-UV||_F, V >= 0, rows V orthogonal
% Compute the angles between U and Mn
A = Mn'*U; %n by r matrix
[a,b] = max(A');
V = zeros(r,n);
% to be improved without loop
for i = 1 : n
    V(b(i),i) =  M(:,i)'*U(:,b(i))/norm(U(b(i),:))^2;
end

% Scale (U,V) for V to have unit norm rows
for i = 1 : r
    nvi = norm(V(i,:)); 
    V(i,:) = V(i,:) / nvi;
    U(:,i) = U(:,i) * nvi;
end
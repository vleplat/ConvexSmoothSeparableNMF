% COMPARE_CLUSTERING computes the error between the optimal binary
% solution Hopt and the output of the ODsymNMF algorithm by finding the
% best match between columns of the 2 matrices. For that purpose, an
% assignement problem using Munkres algorithm is used.

% -----------------
% Input
% -----------------
%   Hout           : (n x r) output matrix of ComputeODsymNMF
%   Hopt           : (n x r) optimal matrix of the ODsymNMF problem
%   bin            : 0: Hout is unchanged 
%                    1: Hout is binarised using round 
%                    2: Hout is binarised using the max of each row
%   overlap        : 0: no overlap between clusters - 1: overlap. This
%                       options is used in the binarizing process
%
% -----------------
% Output
% -----------------
%   err            : error of the assignement problem

function [err,nCorClus,perm] = Compare_clustering(Hout,Hopt,bin,overlap)
    
    % Get the sizes of the matrices
    [n,r] = size(Hopt);
    
    % If the binarising flag is ON we binarise Hout.
    if bin==1
        Hout = round(Hout);
    elseif bin==2
        Hout = Binarizing(Hout,overlap);
    end
    
    % Initialize the cost matrix. costmat(i,j) represents the cost of
    % assigning the ith column of Hout to the jth column of Hopt.
    costmat = zeros(r,r);
    
    % Iterate over each element
    for i=1:r
        for j=1:r
            % The cost of assigning columns is equal to the L2-norm of the
            % difference between the 2 columns. The more different they
            % are, the more the norm will be and the cost will therefore be
            % bigger.
            costmat(i,j) = norm(Hout(:,i)-Hopt(:,j),'fro')^2;
            
            % L1 norm
            % costmat(i,j) = sum(abs(Hout(:,i)-Hopt(i)));
        end
    end
    
    % Get the cost of the best assignement = error between Hopt and Hout.
    [perm,err] = munkres(costmat);
    perm = sum(perm.*(1:r)');
    
    nCorClus = Correct_clusters(Hout(:,perm),Hopt);
    
%     err2 = norm(Hout(:,perm)-Hopt,'fro')/norm(Hopt,'fro');
    
    % Rescale the error with respect to the dimensions
    % L2-norm
    err = sqrt(err/r/n);
    % L1-norm
    % err = err/n;
end

% CORRECT_CLUSTERS return the number of clusters in Hout which are the
% sames as in Hopt. The columns of Hout and Hopt must already match (an
% assignement algorithm is supposed to have been used).

% -----------------
% Input
% -----------------
%   Hout           : (n x r) output matrix of ComputeODsymNMF
%   Hopt           : (n x r) optimal matrix of the ODsymNMF problem
%
% -----------------
% Output
% -----------------
%   nCorClus       : number of correct clusters in [0,r]

function nCorClus = Correct_clusters(Hout,Hopt)
    [~,r] = size(Hout);
    nCorClus = 0;
    
    for i=1:r
        if Hout(:,i)==Hopt(:,i)
            nCorClus = nCorClus + 1;
        end
    end
end
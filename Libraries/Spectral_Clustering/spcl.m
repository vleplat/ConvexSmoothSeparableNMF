function [C, L, U] = spcl(W, k, Type)
%SPECTRALCLUSTERING Executes spectral clustering algorithm
%   Executes the spectral clustering algorithm defined by
%   Type on the adjacency matrix W and returns the k cluster
%   indicator vectors as columns in C.
%   If L and U are also called, the (normalized) Laplacian and
%   eigenvectors will also be returned.
%
%   'W' - Adjacency matrix, needs to be square
%   'k' - Number of clusters to look for
%   'Type' - Defines the type of spectral clustering algorithm
%            that should be used. Choices are:
%      1 - Unnormalized
%      2 - Normalized according to Shi and Malik (2000)
%      3 - Normalized according to Jordan and Weiss (2002)
%
%   References:
%   - Ulrike von Luxburg, "A Tutorial on Spectral Clustering", 
%     Statistics and Computing 17 (4), 2007
%
%   Author: Ingo Buerk
%   Year  : 2011/2012
%   Bachelor Thesis
% calculate degree matrix
 

degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);
% compute unnormalized Laplacian
L = D - W;
% compute normalized Laplacian if needed
switch Type
    case 2
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate inverse of D
        D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L;
    case 3
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate D^(-1/2)
        D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L * D;
end



% compute the eigenvectors corresponding to the k smallest
% eigenvalues

%[U, ~] = eigs(L, k,'sr')
  diff =eps;
  [U, ~] = eigs(L, k, diff) ;
  
  
  a=find(isnan(U)==1);p=1;la=length(a);
  while la>0  % make sure U doesn't include NAN
 [U, ~] = eigs(L, k, diff) ;  
  a=find(isnan(U)==1);
  p=p+1;la=length(a);
  end
 
 
% in case of the Jordan-Weiss algorithm, we need to normalize
% the eigenvectors row-wise
if Type == 3
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
end
% now use the k-means algorithm to cluster U row-wise
% C will be a n-by-1 matrix containing the cluster number for
% each data point

   C = kmeans(U, k);
% C = kmeans(U, k, 'start', 'cluster', 'EmptyAction', 'singleton');
             
% now convert C to a n-by-k matrix containing the k indicator
% vectors as columns
C = sparse(1:size(D, 1), C, 1);




%function [IDX,C,SUMD,K]=kmeans_opt(X,varargin)
%%% [IDX,C,SUMD,K]=kmeans_opt(X,varargin) returns the output of the k-means
%%% algorithm with the optimal number of clusters, as determined by the ELBOW
%%% method. this function treats NaNs as missing data, and ignores any rows of X that
%%% contain NaNs.
%%%
%%% [IDX]=kmeans_opt(X) returns the cluster membership for each datapoint in
%%% vector X.
%%%
%%% [IDX]=kmeans_opt(X,MAX) returns the cluster membership for each datapoint in
%%% vector X. The Elbow method will be tried from 1 to MAX number of
%%% clusters (default: square root of the number of samples)
%%% [IDX]=kmeans_opt(X,MAX,CUTOFF) returns the cluster membership for each datapoint in
%%% vector X. The Elbow method will be tried from 1 to MAX number of
%%% clusters and will choose the number which explains a fraction CUTOFF of
%%% the variance (default: 0.95)
%%% [IDX]=kmeans_opt(X,MAX,CUTOFF,REPEATS) returns the cluster membership for each datapoint in
%%% vector X. The Elbow method will be tried from 1 to MAX number of
%%% clusters and will choose the number which explains a fraction CUTOFF of
%%% the variance, taking the best of REPEATS runs of k-means (default: 3).
%%% [IDX,C]=kmeans_opt(X,varargin) returns in addition, the location of the
%%% centroids of each cluster.
%%% [IDX,C,SUMD]=kmeans_opt(X,varargin) returns in addition, the sum of
%%% point-to-cluster-centroid distances.
%%% [IDX,C,SUMD,K]=kmeans_opt(X,varargin) returns in addition, the number of
%%% clusters.
%%% sebastien.delandtsheer@uni.lu
%%% sebdelandtsheer@gmail.com
%%% Thomas.sauter@uni.lu
% [m,~]=size(X); %getting the number of samples
% if nargin>1, ToTest=cell2mat(varargin(1)); else, ToTest=ceil(sqrt(m)); end
% if nargin>2, Cutoff=cell2mat(varargin(2)); else, Cutoff=0.95; end
% if nargin>3, Repeats=cell2mat(varargin(3)); else, Repeats=3; end
% %unit-normalize
% MIN=min(X); MAX=max(X); 
% X=(X-MIN)./(MAX-MIN);
% D=zeros(ToTest,1); %initialize the results matrix
% for c=1:ToTest %for each sample
%     [~,~,dist]=kmeans(X,c,'emptyaction','drop'); %compute the sum of intra-cluster distances
%     tmp=sum(dist); %best so far
%     
%     for cc=2:Repeats %repeat the algo
%         [~,~,dist]=kmeans(X,c,'emptyaction','drop');
%         tmp=min(sum(dist),tmp);
%     end
%     D(c,1)=tmp; %collect the best so far in the results vecor
% end
% Var=D(1:end-1)-D(2:end); %calculate %variance explained
% PC=cumsum(Var)/(D(1)-D(end));
% [r,~]=find(PC>Cutoff); %find the best index
% K=1+r(1,1); %get the optimal number of clusters
% [IDX,C,SUMD]=kmeans(X,K); %now rerun one last time with the optimal number of clusters
% C=C.*(MAX-MIN)+MIN;
% end
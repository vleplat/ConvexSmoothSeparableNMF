%% The following function helps us to indentify factor matrix H from X.

%Input: X from  M~=MX where M is a "orth-separable" mix matrix.
% r is number of clusters,

%   options:
%   options.delta: delta is threshold to decide the number of selected rows of X
%   options.numX: number of selected rows of X
%   options.Type: Defines the type of spectral clustering algorithm  that should be used. 
%   'Type' - 
%      1 - Unnormalized
%      2 - Normalized according to Shi and Malik (2000)
%      3 - Normalized according to Jordan and Weiss (2002)
%   options.modeltype: decide to use nnls or alternatingONMF. Usually nnls
%   for  mix model and alternatingONMF for ONMF model
%   'modeltype'-
%      0 - ONMF
%      1 - Mix model


%Output: Klusters Ki: K1+K2+...+Kr=K where Ki stands for the index set of ith cluster. 
         %W and H such that  M~=WH,  

function [W,H,Kclusters,Xs] =alg2(M,X,r, options)

sx=sum(X,2);
%% Selection of K to build X(:,K) or X(K,K)
if ~isfield(options,'delta') && isfield(options,'numX') 
    [~,idx]=sort(sx,'descend');
    numX=options.numX;
    Tinx=idx(1:numX);  
elseif isfield(options,'delta') && ~isfield(options,'numX') 
   delta=options.delta;
   Tinx=find(sx>=delta);
elseif ~isfield(options,'delta') && ~isfield(options,'numX') 
     delta=0;
     Tinx=find(sx>=delta);
else
    error(message('Ooops, delta and numX lead to different row numbers of X. Please do not give little AI two choices')) 
end

if ~isfield(options,'type')
    options.type=1;
else
    type=options.type;
end

%% Clustering
%%%% spectral clustering on  (X(Tinx,Tinx)+X'(Tinx,Tinx)) or X(Tinx,:)*X(Tinx,:)';%%%%
if options.clustering==0
    Xs=X(Tinx,Tinx);
    Xs=(Xs+Xs');
    Xs(Xs<1e-4)=0;
    
    % Xs=X(Tinx,:)*X(Tinx,:)';
    % Xs(Xs<1e-4)=0;
    clusters = spcl(Xs,r,type);
%%%% kmeans
elseif options.clustering==1
    Xs=X(:,Tinx); 
    [IDX, ~] = kmeans(Xs', r);
end

 for i=1:r
     if options.clustering==0
        Kclusters{i}=Tinx(find(clusters(:,i)==1));
     elseif options.clustering==1
        Kclusters{i}=Tinx(find(IDX==i)); 
     end
     if options.agregation==0
        W(:,i)=mean(M(:,Kclusters{i}),2);
     elseif options.agregation==1
        W(:,i)=median(M(:,Kclusters{i}),2); 
     elseif options.agregation==2
        W{1}(:,i)=mean(M(:,Kclusters{i}),2); 
        W{2}(:,i)=median(M(:,Kclusters{i}),2); 
     end
 end
  
 if options.modeltype==0 
     % Here, we have the special case H=H0 -> ONMF
     if options.agregation==2
        [W{1},H{1}] = alternatingONMF_fast(M,W{1},1000); 
        [W{2},H{2}] = alternatingONMF_fast(M,W{2},1000); 
     else
        [W,H] = alternatingONMF_fast(M,W,1000);
     end
 else
    if options.agregation==2
        H{1}=nnlsHALSupdt_new(W{1}'*M,W{1},[],1000);
        H{2}=nnlsHALSupdt_new(W{2}'*M,W{2},[],1000);
    else
        H=nnlsHALSupdt_new(W'*M,W,[],1000); 
    end
 end

 



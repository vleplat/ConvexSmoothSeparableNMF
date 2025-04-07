% ONMF on CBCL 

load('C:/Users/Nicolas/Dropbox/Data/Faces/cbclim.mat'); 
M = M'; 

[m,n] = size(M); 
r = 49; 

K = FastSepNMF(M,49); 
[W,H] = alternatingONMF(M,M(:,K),100);

%K = randperm(n); 
%[W,H] = alternatingONMF(M,M(:,K(1:r)),100); 

affichage(H',7,19,19); 
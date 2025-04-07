% ONMF on CBCL 

load('C:/Users/Nicolas/Dropbox/Data/Hyperspectral imaging/Urban.mat'); 
A = A'; 

[m,n] = size(A); 
r = 6; 

K = FastSepNMF(A,r); 

[W,H] = alternatingONMF(A,A(:,K),100); 

affichage(H([2 1 6  3 5 4],:)',3,307,307); 
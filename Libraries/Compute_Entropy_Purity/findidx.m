% the following function aims to find the cluster from clustering matrix A
% size of m*r. 

function id=findidx(A)
[m,r]=size(A);
for i=1:r
    id{i}=find(A(:,i)==1);
end


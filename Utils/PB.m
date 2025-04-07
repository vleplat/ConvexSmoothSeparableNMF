%this function is producing an orthogonal nonnegative matrix randomly
function B=PB(m,r,Type)
%type=0, balanced
%type=1, unbanced


 switch Type
     case 0
    a=(m/r)*ones(1,r);


     case 1  
     a=rand(1,r);  a=a/sum(a);  a=floor(m*a);
        for i=1:r
          if a(i)==0
            a(i)=1;
          end
        end
          a(r)=m-sum(a(1:r-1));
 
 end


 
 B=zeros(m,r);k=0;
for i=1:r
    b=abs(rand(a(i),1));
    b=b/norm(b);
    B(k+1:k+a(i),i)=b;
   k=k+a(i);
end
%  B = B(randperm(m),:);

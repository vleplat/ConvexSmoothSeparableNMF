function entro=Entropy(x,y,n,r)
% original clustering index x;
% our clustering index y;
% n is point number
% r is clustering number (n point are cluster into r classess)
a=0;b=0;
for j=1:r 
     zy=double(y{j});
     szy=length(zy);
    for i=1:r 
     zx=double(x{i});
     z=intersect(zx,zy); sz=length(z);
     if sz~=0
     a=a+sz*log(sz/szy);
      end
    end
    b=b+a;
end
   entro=-b/(n*log(r));
 
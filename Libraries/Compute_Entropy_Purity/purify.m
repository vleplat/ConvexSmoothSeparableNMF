function purity=purify(x,y,n,r)
% original clustering index x;
% our clustering index y;
% n is point number
% r is clustering number (n point are cluster into r classess)

num=zeros(r,1);
for i=1:r
    for j=1:r
    zx=double(x{j});
    zy=double(y{i});
    z=intersect(zx,zy);
    num(j)=length(z);
    end
    cnum(i)=max(num);
end
    
purity=sum(cnum)/n;
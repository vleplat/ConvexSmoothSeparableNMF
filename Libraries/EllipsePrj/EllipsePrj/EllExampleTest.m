% Script to test a projection on ellipsoid

% Data, generate ellipse centered at origin, { x: x'*Q*x = 1 }
n = 3; % dimension, prudent for n>20
L = randn(n);
Q = L'*L;

% Given point in R^n (to be projected on ellipse)
c = 1*randn(n,1);

%%
[radii U] = EllAlg2Geo(Q, 0, -1);
x = EllPrj(c, radii, U, 0, true);

% Compute l = |x-c|
l = sqrt(sum(bsxfun(@minus, x, c).^2, 1));
% Where the distance is minimal
[lmin imin] = min(l);
fprintf('Smallest distance = %f\n', lmin);

%%
% Graphical check
% Generate many points on ellipse
A = U*diag(radii);
clf;
if n==2
    theta = linspace(0,2*pi,181);
    ellipse = A*[cos(theta); sin(theta)];
    
    plot(ellipse(1,:),ellipse(2,:));
    axis equal;
    hold on;
    % Plot c projected to the ellipse
    plot(c(1),c(2),'ok');
    for k=1:size(x,2)
        if k==imin
            linespec  = '-r.';
        else
            linespec = '-c.';
        end
        plot([x(1,k) c(1)],[x(2,k) c(2)],linespec);
    end
elseif n==3
    N = 50;
    [X Y Z] = ellipsoid(0,0,0,radii(1),radii(2),radii(3),N);
    XYZ = U*[X(:) Y(:) Z(:)]';
    X = reshape(XYZ(1,:),[N N]+1);
    Y = reshape(XYZ(2,:),[N N]+1);
    Z = reshape(XYZ(3,:),[N N]+1);
    surf(X,Y,Z,'Edgecolor','none','FaceColor','flat');
    colormap gray;
    alpha(0.5);
    axis equal;
    hold on;
    plot3(c(1),c(2),c(3),'or');
    % Plot c projected to the ellipse
    for k=1:size(x,2)
        if k==imin
            linespec = '-r.';
        else
            linespec = '-c.';
        end
        plot3([x(1,k) c(1)],[x(2,k) c(2)],[x(3,k) c(3)],linespec);
    end
else
    fprintf('Sorry I can''t plot\n');
end

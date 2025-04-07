% Script to test a projection on conic in 2D

% Data
A=randn(2);
b = randn(2,1);
c = -5;
P=randn(2,1);

q = ConicPrj(P, A, b, c);

if isempty(q)
    fprintf('Sorry, an empty conic is generated, please try again\n');
    return
end

% Plot the conic
xmin = floor(min(q(1,:)))-2;
xmax = floor(max(q(1,:)))+2;
ymin = floor(min(q(2,:)))-2;
ymax = floor(max(q(2,:)))+2;
x = linspace(xmin,xmax);
y = linspace(ymin,ymax);
[X Y] = meshgrid(x,y);
Z = A(1,1)*X.^2 + A(2,2)*Y.^2 + (A(1,2)+A(2,1))*X.*Y + b(1)*X + b(2)*Y + c;

% Graphic check
clf
contour(x,y,Z,[0 0])
axis equal;
hold on;
plot(P(1),P(2),'or-');
for k=1:size(q,2)
    plot([P(1) q(1,k)],[P(2) q(2,k)],'.r-');
end
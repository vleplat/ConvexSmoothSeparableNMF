function y = StdConicPrj(d, s, g, l, parabolatol)
% y = StdEllPrj(d, s, g, l)
%
% Find the projection of point "d" in R^n on the 'standard' conic
%      E := { x in R^n: x'*S*x + g'*x = l }, with S = diag(s).
% 'Standard' means E is aligned with Cartesian's axis
%
% INPUTS:
% - d is (n x 1) vector
% - s is (n x 1) vector, inverse of the square of the radii
% - g is (n x 1) vector, optional, zeros(n,1) by default
% - l is scalar, optional, 1 by default
%
% When all elements of S is strictly positive, E is an ellipsoid;
%      if s has positive and zeros, E is a hyper-parabola
%      otherwise E is an hyperbolic;
%
% >> y = StdEllPrj(..., parabolatol) to force any axis having radii larger
% than parabolatol*(smallest axis radii) to be infinity.
% Default value of parabolatol is 1e-3.
%
% OUTPUT:
%   y (n x p) array, all possible solutions. Each column is coordinate
%             of the solutions
%
% Method: solve the Euler Lagrange equation with respect to the
%         Lagrange multiplier, which can be written as polynomial
%         equation (from an idea by Roger Stafford)
%
% See also: EllPrj, Pprod
%
% Author: Bruno Luong <brunoluong@yahoo.com>
%   Original: 24-May-2010
%   25-May-2010: extend to generic conic
%   25-May-2010: fix a bug
%                Slight speed improvement for centered ellipsoid

if nargin<3 || isempty(g)
    g = zeros(size(d),class(d));
elseif isscalar(g)
    g = g+zeros(size(d),class(d));
end

if nargin<4 || isempty(l)
    l = 1;
end

if nargin<5 || isempty(parabolatol)
    parabolatol = 1e-3;
end

d = d(:);
s = s(:);
g = g(:);

% solve for lambda
%    d-y = lambda*(S*y + 0.5*g) % Euler Lagrange eqt
%    y'*S*y + g'*y = l

% Set of indice that is not degenerated to parabola
i1 = abs(s) > max(abs(s))*parabolatol^2;
i2 = find(~i1);
i1 = find(i1);

% This is polynome (s*lambda+1)^2
parray = [s(i1).^2, 2*s(i1), ones(size(i1))];

% Compute the products of all polynomials
[Pc Pi] = Pprod(parray);

% The first order terms: g * (-lambda*g/2 + d) * (s*lambda+1)
parray = [-0.5*g(i1).*s(i1), s(i1).*d(i1)-0.5*g(i1), d(i1)];
parray = bsxfun(@times, g(i1), parray);
P1 = 0;
for i=1:size(parray,1)
    if g(i1(i)) % we only add it needs
        P1 = P1 + conv(Pi(i,:),parray(i,:));
    end
end

% The second order terms: s * (-lambda*g/2 + d)^2
% This term always different zero when the conic is not degenerate
% in a subspace
parray = [0.25*g(i1).^2, -g(i1).*d(i1), d(i1).^2];
parray = bsxfun(@times, s(i1), parray);
P2 = 0;
for i=1:size(parray,1)
    if g(i1(i))
        P2 = P2 + conv(Pi(i,:),parray(i,:));
    else
        P2 = P2 + Pi(i,:)*parray(i,3);
    end
end

% The first order term: g * (-lambda*g/2 + d)
parray = [-0.5*g(i2), d(i2)];
parray = bsxfun(@times, g(i2), parray);
P3 = sum(parray,1);
P3(end) = P3(end) - l;
if P3(1) % ~= 0 % Bug fixed 26-May-2010
    P3 = conv(Pc, P3);
else
    P3 = P3(2)*Pc;
end 

P = polysum(P1,P2,P3);
% Find the real roots of polynomial
lambda = Proots(P);

% Compute: y = (d - lambda*g/2) ./ (lambda*s + 1)
ls = bsxfun(@times, s, lambda);
lg = bsxfun(@times, g/2, lambda);
dmlg = bsxfun(@minus, d, lg);
y = bsxfun(@rdivide, dmlg, ls+1);

end % StdEllPrj

%%
function s = polysum(varargin)
% s = polysum(P1, P2, ...)
% Sum polynomials
Plist = varargin;
lgt = cellfun('size', Plist, 2);
s = zeros(1,max(lgt));
for k=1:length(Plist)
    tail = size(s,2)+(-lgt(k)+1:0);
    s(tail) = s(tail) + Plist{k};
end
end % polysum

%%
function lambda = Proots(P)
% lambda = Proots(P)
% Find the real roots of polynomial
% Remove the nil dominant coefficients 
i1 = find(P,1,'first');
P = P(i1:end);
lambda = roots(P);
lambda = reshape(lambda, 1, []);
% Filter out complex roots
lambda = lambda(imag(lambda)==0);
end % Proots

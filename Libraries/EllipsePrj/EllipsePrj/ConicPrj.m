function x = ConicPrj(P, A, b, c, parabolatol)
%
% x = ConicPrj(P, A, b, c)
%
% Find the projection of point P in R^n on the conic
%   E = { x such that: x'*A*x + b'*x + c = 0}
%
% INPUTS:
% - P is (n x 1) vector
% - A is (n x n) matrix
% - b (n x 1) vector
% - x scalar
%
% >> x = ConicPrj(..., parabolatol) to force any axis having radii larger
% than parabolatol*(smallest axis radii) to be infinity.
% Default value of parabolatol is 1e-3.
%
% OUTPUT:
%   x (n x p) array, all possible solutions. Each column is coordinate
%   of the candidate point.
%   The condition satisfied by x is vk := (x(:,k)-P) are orthogonal to E.
%   The maximum number of candidates is 2*n, the minimum is 2.
%   It is up to user to filter out the smallest and largest distance to P
%   depending on the need.
%
% Limitation: problem might arise when using on very large dimension
% (e.g., > 50).
%
% See also: StdConicPrj, EllPrj
%
% Author: Bruno Luong <brunoluong@yahoo.com>
%   Original: 25-May-2010

if nargin<5 || isempty(parabolatol)
    parabolatol = 1e-3;
end

P = P(:);
n = size(P,1);

% Symmetrize A
A = 0.5*(A+A.');

% Factorization A = U'*diag(S)*U
% U'*U = eye(n)
[U D] = eig(A);
s = diag(real(D));
% Make sure the direction are mutually orthogonal
[U R E] = qr(U,0); %#ok
p(E) = 1:n;
U = U(:,p);

% Change coordinates
%   P = U*d
%   x = U*y
%   b = U*g
d = U'*P(:);
g = U'*b(:);
l = -c;

y = StdConicPrj(d, s, g, l, parabolatol);

% Change back the coordinates
x = U*y;

end % ConicPrj

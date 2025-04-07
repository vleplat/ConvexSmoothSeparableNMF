function [radii U x0] = EllAlg2Geo(A, b, c)
% [radii U x0] = EllAlg2Geo(A, b, c)
%
% Transform the Ellipsoid from the algebraic form
%   E = { x'*A*x + b'*x + c = 0 } where
%       A is (n x n) symmetric-definite-positive matrix (property not check)
%       b = (n x 1) vector
%       c is scalar
% to the geometric form:
%   y = U'*(x-x0)
%   z = y./radii
%   |z|^2 = sum(z.^2) = 1
% Note: The transform inverse from z to x is
%   x = x0 + U*(z.*radii)
%
% See also: EllGeo2Alg
%
% Author: Bruno Luong <brunoluong@yahoo.com>
%   Original: 24-May-2010

if isscalar(b)
    b = b + zeros(size(A,2),1,class(A));
end
% E = { (x-x0)'*Q*(x-x0) = 1 }
x0 = -0.5*(A\b);
d = x0'*A*x0-c;
if d<=0
    error('EllAlg2Geo: empty ellipsoid');
end
Q = A/d;

[U S] = svd(Q);
radii = sqrt(1./diag(S));

end % EllAlg2Geo

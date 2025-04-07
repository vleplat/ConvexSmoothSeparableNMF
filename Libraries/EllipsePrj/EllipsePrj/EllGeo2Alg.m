function [A b c] = EllGeo2Alg(radii, U, x0)
% [A b c] = EllGeo2Alg(radii, U, x0)
%
% Transform the Ellipsoid from the geometric form:
%   E = { x = x0 + U*(z.*radii) : |z| = 1 }
% to the algebraic form:
%   E = { x'*A*x + b'*x + c = 0 } where
%       A is (n x n) symmetric-definite-positive matrix (property not check)
%       b = (n x 1) vector
%       c is scalar
% Note: to convert x on E to z (in S(0,1))
%   y = U'*(x-x0)
%   z = y./radii
%
% See also: EllAlg2Geo
%
% Author: Bruno Luong <brunoluong@yahoo.com>
%   Original: 24-May-2010

radii = reshape(radii, 1, []);
W = bsxfun(@rdivide, U, radii);
A = W*W';
b = -2*A*x0;
c = x0'*A*x0-1;

end % EllGeo2Alg

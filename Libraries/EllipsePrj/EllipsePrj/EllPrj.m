function x = EllPrj(P, radii, U, x0, nocheck)
%
% x = EllPrj(P, radii, U, x0)
%
% Find the projection of point P in R^n on the ellipsoid
%   E = { x = x0 + U*(z.*radii) : |z| = 1 }, where
% - radii is the length of the ellipsoid axis. 
% - U is a n x n matrix given the orientation of the ellipsoid, i.e.,
%   U(:,k) is the kth axis of the ellipsoid corresponding to radii(k).
%   If U is not provided, ellipsoid is alligned with Cartesian axis.
% - x0 is the center coordinates of the ellipsoid. If x0 is not provided
%   origin is selected
%
% INPUTS:
% - P is (n x 1) vector
% - radii is (n x 1) vector
% - U (optional) is (n x n) matrix
% - x0 (optional) is (n x 1) vector
%
% OUTPUT:
%   x (n x p) array, all possible solutions. Each column is coordinate
%   of the candidate point.
%   The condition satisfied by x is vk := (x(:,k)-P) are orthogonal to E.
%   The maximum number of candidates is 2*n, the minimum is 2.
%   It is up to user to filter out the smallest and largest distance to P
%   depending on the need.
%
% To discard orthogonaliy check of U (expensive), call
% with TRUE flag in fifth parameter
%   EllPrj(P, radii, U, x0, true)
%
% Limitation: problem might arise when using on very large dimension
% (e.g., > 50).
%
% See also: StdConicPrj, ConicPrj
%
% Author: Bruno Luong <brunoluong@yahoo.com>
%   Original: 24-May-2010

n = size(radii,1);

radii = radii(:);
if nargin<3 || isempty(U)
    U = eye(n);
else
    if nargin<5 || ~nocheck
        % Make sure the direction are mutually orthogonal
        [U R E] = qr(U,0); %#ok
        p(E) = 1:n;
        U = U(:,p);
    end
end
if nargin<4 || isempty(x0)
    x0 = zeros(n,1,class(radii));
end
s = 1./radii.^2;

% Change coordinates
%   P = U*d+x0
%   x = U*y+x0
d = U'*(P(:)-x0(:));
y = StdConicPrj(d, s, 0, 1);
x = bsxfun(@plus, U*y, x0(:));

end % EllPrj

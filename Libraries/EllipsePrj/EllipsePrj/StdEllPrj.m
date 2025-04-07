function y = StdEllPrj(d, s)
% y = StdEllPrj(d, s)
%
% Find the projection of point D in R^n on the 'standard' ellipsoid
%      E := { x in R^n: x'*S*x = 1 }, with S = diag(s).
% 'Standard' means E is centered about the origin and aligned with
% cartedsian's axis
%
% INPUTS:
% - d is (n x 1) vector
% - s is (n x 1) vector, inverse of the square of the radii
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

d = d(:);
s = s(:);

% solve for lambda
%    d-y = lambda*S*y
%    y'*S*y = 1

% This is polynome (s*lambda+1)^2
pd = [s.^2 2*s ones(size(s))];
a = s.*d.^2;
% Compute the products of polynomials
[Pc Pi] = Pprod(pd);
P = sum(bsxfun(@times, a, Pi),1);
P = [0 0 P]-Pc;
%% Solve for roots
lambda = roots(P);
lambda = reshape(lambda, 1, []);
% Filter out complex roots
lambda = lambda(imag(lambda)==0);

% Compute: y = d / (lambda*s + 1)
ls = bsxfun(@times, lambda, s);
y = bsxfun(@rdivide, d, ls+1);

end % StdEllPrj

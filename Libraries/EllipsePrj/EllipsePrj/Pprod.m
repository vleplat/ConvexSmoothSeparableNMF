function [Pc Pi] = Pprod(pd)
% [Pc Pi] = Pprod(pd)
% Compute multiplication of n-polynomials, and also n leave-one-out
% polynomials
%
% INPUT:
%   pd (n x m) stores n polynomials (of order m-1) in each row
% OUTPUTS:
%   - Pc is product of all polynomials: PROD [ pd(j,:) for j=1,2, ..., n ]
%   - Pi is an array polynomials of product but leaving-one-out.
%     Pi(i,:) = PROD [ pd(j,:) for j#i ]
%
% Smart booking so as the complexity is O(n*log(n))
%
% See also: StdEllPrj
%
% Author: Bruno Luong <brunoluong@yahoo.com>
%   Original: 24-May-2010

n = size(pd,1);
k = nextpow2(n);

% Building the tree from bottom-to-top (leaf to root)
tree = cell(1,k+1);
% initialize leaf
tree{1} = struct('child', [], 'coef', num2cell(pd,2));

k = 1;
l = n; % size(tree{k},1);
while l>1
    tk = tree{k};
    l = ceil(l/2);
    tkp1 = struct('child', cell(l,1), 'coef', cell(l,1));
    % Take convolution by pair
    for i=1:l
        j = 2*i-1;
        if j<size(tk,1)
            tkp1(i) = struct('child', [j j+1], ...
                             'coef', conv(tk(j).coef,tk(j+1).coef));
        else % alone trailing
            tkp1(i) = struct('child', j, ...
                             'coef', tk(j).coef);
        end        
    end % for-loop
    k = k+1;
    tree{k} = tkp1(:);
end % while

% Retreive the product of all the polynomials is here
Pc = tree{end}.coef;

% Building the leave-one-out products
Pi = leaveoneout(tree, cell(n, 1), size(tree,2), 1, 1);
% Put the result in array
Pi = cat(1, Pi{:});

end % Pprod

%%
% Go through the tree recursively , and take the product of polynomial
% by leaving one-out
function Pi = leaveoneout(tree, Pi, level, ind, cp)
% tree: is the tree, unchanged
% Pi: current cell list of *all* polynomials, updated when we hit
%     the first level (leaf)
% level: where we are on the tree?
% ind: indice of the node in the tree at the current level
% cp: current cumulative product of the branch currently visited

if level==1
    Pi(ind) = {cp}; % done, return
else
    % Recursive call
    node = tree{level}(ind);
    child = node.child;
    up = level-1;
    if length(child)==1
        Pi = leaveoneout(tree, Pi, up, child, cp);
    else % length(child)==1
        Pi = leaveoneout(tree, Pi, up, child(1), ...
                         conv(cp, tree{up}(child(2)).coef));
        Pi = leaveoneout(tree, Pi, up, child(2), ...
                         conv(cp, tree{up}(child(1)).coef));
    end
end

end % leaveoneout

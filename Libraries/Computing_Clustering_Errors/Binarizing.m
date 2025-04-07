% BINARIZING allows to binarise a matrix with or without an overlap. If the
% overlap is equal to 0 then there must be only one 1 by row of H.
% Otherwise there may be multiple 1.
%
% The strategy of this function is to look in each rows of H and put the
% maximum to 1 and the other value to 0.

% -----------------
% Input
% -----------------
%   H              : (n x r) matrix to binarize
%
% -----------------
% Output
% -----------------
%   H              : (n x r) binarized matrix
%   overlap        : 0: no overlap - 1: overlap

function H = Binarizing(H,overlap)
    
    % Get the number of rows of H
    [n,~] = size(H);
    
    if overlap==0
        
        % Iterate over the rows of H
        for i=1:n
            
            % Get the position of the maximum(s) of the ith row
            maxpos = find(H(i,:)==max(H(i,:)));
            
            % Take randomly one maximum and put it to 1
            newpos = randi(length(maxpos));
            
            % Initialize the row to zero
            H(i,:) = 0;
            
            % Put the randomly chosed position to 1
            H(i,maxpos(newpos)) = 1;
        end
    else
        % Not implemented yet
    end

end


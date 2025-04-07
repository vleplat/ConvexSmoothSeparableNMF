function [Y, Out] = proj_dsm_BFGS(A, x0, n)
% Author:  created by Dejun Chu, djun.chu@gmail.com
% If you have any questions or comments, please feel free to contact us.

opts.gtol = 1e-10;   % convergence tolerance
opts.record = 0;     % 0: no print-out; else with print-out
opts.maxiter = 2000;  % max number of iterations

e = ones(n,1);
[x_BFGS, Out] = fminBFGS(x0, @dualproj, opts, A, e, n);
y_BFGS = x_BFGS(1:n); 
z_BFGS = x_BFGS(n+1:end);
Y = proj(bsxfun(@minus, bsxfun(@minus, A, y_BFGS), z_BFGS'));
Out.x = x_BFGS;


    function [g, Hd, dual] = dualproj(x,G,e,n)
        y = x(1:n); z = x(n+1:end);
        tempG = bsxfun(@minus, bsxfun(@minus, G, y), z');%
        
        [projG, Inx] = proj(tempG);
        Haa = sum(Inx,2);
        Haa(Haa==0) = 1;
        Hbb = sum(Inx);
        Hbb(Hbb==0)=1;
        Hd = [Haa; Hbb'];
        
        dual.Inx = Inx;
        dual.projG = projG(:);
        
        g = -[projG*e; projG'*e] + 1;
        
    end

    function [Z, I] = proj(C)
        Z = max(0, C);
        
        if(nargout>1)
            I = C>0;
        end
    end
end